#!/usr/bin/env python3
"""
Fine-tune openai/whisper-tiny on Telugish & Indian-accented English (data/train.jsonl)
and evaluate on data/eval.jsonl after each epoch using PyTorch + CUDA mixed precision (fp16/bf16) on CUDA GPU.

Features:
- Configurable learning rate, linear/cosine warmup, weight decay, gradient accumulation, and SpecAugment
- Evaluates validation loss, exact WER, CER, and RTF after every epoch (plus optional baseline epoch 0)
- Saves best checkpoint and processor to /home/pavan_veesam/telugish_whisper_edge/models/telugish-whisper-tiny-ft
- Logs step-by-step training loss and epoch-by-epoch validation WER/CER history to JSON for plotting
"""

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from evaluate_model import (
    BASE_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_BENCHMARKS_DIR,
    DEFAULT_EVAL_JSONL,
    evaluate_checkpoint,
    get_audio_path,
    get_reference_text,
    load_audio_16k,
    normalize_text,
)

DEFAULT_TRAIN_JSONL = BASE_DIR / "data" / "train.jsonl"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "telugish-whisper-tiny-ft"
DEFAULT_HISTORY_JSON = DEFAULT_BENCHMARKS_DIR / "training_history.json"


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TelugishAudioDataset(Dataset):
    """Dataset that loads audio and transcripts from a JSONL file for Whisper training/eval."""

    def __init__(
        self,
        jsonl_path: Path,
        processor: WhisperProcessor,
        max_samples: Optional[int] = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_dir = self.jsonl_path.parent
        self.processor = processor
        self.records: List[Dict[str, Any]] = []

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL dataset not found: {self.jsonl_path}")

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))

        if max_samples is not None and max_samples > 0:
            self.records = self.records[:max_samples]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        audio_path = get_audio_path(rec, self.jsonl_dir)
        text = get_reference_text(rec)
        audio, duration = load_audio_16k(audio_path)

        inputs = self.processor.feature_extractor(
            audio, sampling_rate=16000, return_tensors="pt"
        )
        input_features = inputs.input_features[0]

        labels = self.processor.tokenizer(
            text, truncation=True, max_length=448
        ).input_ids

        return {
            "input_features": input_features,
            "labels": labels,
            "text": text,
            "audio_path": audio_path,
            "duration": duration,
        }


@dataclass
class WhisperDataCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_features = torch.stack([f["input_features"] for f in features])

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        bos_token_id = self.processor.tokenizer.bos_token_id
        if (
            labels.shape[1] > 0
            and bos_token_id is not None
            and (labels[:, 0] == bos_token_id).all().cpu().item()
        ):
            labels = labels[:, 1:]

        return {
            "input_features": input_features,
            "labels": labels,
            "texts": [f["text"] for f in features],
            "durations": [f["duration"] for f in features],
        }


def apply_spec_augment(
    mel_batch: torch.Tensor,
    freq_mask_param: int = 12,
    time_mask_param: int = 40,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """In-batch SpecAugment on log-Mel spectrogram tensor (B, n_mels, n_frames)."""
    B, F, T = mel_batch.shape
    augmented = mel_batch.clone()
    for b in range(B):
        mean_val = augmented[b].mean()
        for _ in range(num_freq_masks):
            f = random.randint(0, freq_mask_param)
            if f > 0 and F - f > 0:
                f0 = random.randint(0, F - f)
                augmented[b, f0 : f0 + f, :] = mean_val
        for _ in range(num_time_masks):
            t = random.randint(0, time_mask_param)
            if t > 0 and T - t > 0:
                t0 = random.randint(0, T - t)
                augmented[b, :, t0 : t0 + t] = mean_val
    return augmented


@torch.no_grad()
def run_validation(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    val_loader: DataLoader,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    language: Optional[str] = "en",
) -> Dict[str, float]:
    """Compute validation loss, exact WER, CER, and RTF on the validation DataLoader."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    all_refs: List[str] = []
    all_hyps: List[str] = []
    total_audio_sec = 0.0
    total_gen_sec = 0.0

    for batch in val_loader:
        input_features = batch["input_features"].to(device)
        labels = batch["labels"].to(device)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_features=input_features, labels=labels)
            loss = outputs.loss

        total_loss += float(loss.item())
        total_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
            gen_kwargs = {"max_new_tokens": 128}
            if language:
                gen_kwargs["language"] = language
                gen_kwargs["task"] = "transcribe"
            pred_ids = model.generate(input_features, **gen_kwargs)

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_gen_sec += time.perf_counter() - t0
        total_audio_sec += sum(batch["durations"])

        preds = processor.batch_decode(
            pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        for ref, hyp in zip(batch["texts"], preds):
            ref_norm = normalize_text(ref)
            hyp_norm = normalize_text(hyp)
            if ref_norm:
                all_refs.append(ref_norm)
                all_hyps.append(hyp_norm)

    val_loss = total_loss / max(total_batches, 1)
    val_wer = float(jiwer.wer(all_refs, all_hyps)) if all_refs else 0.0
    val_cer = float(jiwer.cer(all_refs, all_hyps)) if all_refs else 0.0
    val_rtf = total_gen_sec / max(total_audio_sec, 1e-4)

    return {
        "val_loss": round(val_loss, 4),
        "val_wer": round(val_wer, 4),
        "val_wer_pct": round(val_wer * 100.0, 2),
        "val_cer": round(val_cer, 4),
        "val_cer_pct": round(val_cer * 100.0, 2),
        "val_rtf": round(val_rtf, 4),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")

    print(f"[TRAIN] Loading processor & model from {args.model_name_or_path} on {device}...")
    processor = WhisperProcessor.from_pretrained(
        args.model_name_or_path, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name_or_path)

    # Clear forced_decoder_ids so training labels match tokenizer prompt cleanly
    model.config.forced_decoder_ids = None
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None

    if args.spec_augment:
        model.config.apply_spec_augment = True
        model.config.mask_time_prob = args.mask_time_prob
        model.config.mask_time_length = 10
        model.config.mask_feature_prob = args.mask_feature_prob
        model.config.mask_feature_length = 10

    if args.freeze_encoder:
        model.freeze_encoder()
        print("[TRAIN] Froze Whisper audio encoder layers.")

    model.to(device)

    train_dataset = TelugishAudioDataset(
        Path(args.train_jsonl), processor, max_samples=args.max_train_samples
    )
    val_dataset = TelugishAudioDataset(
        Path(args.eval_jsonl), processor, max_samples=args.max_eval_samples
    )
    collator = WhisperDataCollator(processor=processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    steps_per_epoch = max(1, math.ceil(len(train_loader) / args.grad_accum_steps))
    total_opt_steps = steps_per_epoch * args.epochs
    warmup_steps = (
        args.warmup_steps
        if args.warmup_steps is not None
        else max(1, int(total_opt_steps * args.warmup_ratio))
    )

    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.lr, betas=(0.9, 0.98), eps=1e-8)
    if args.scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_opt_steps
        )

    use_amp = (device.type == "cuda") and (args.precision in ("fp16", "bf16"))
    amp_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and args.precision == "fp16"))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = Path(args.history_json)
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history: Dict[str, Any] = {
        "config": vars(args),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU",
        "train_samples": len(train_dataset),
        "eval_samples": len(val_dataset),
        "total_opt_steps": total_opt_steps,
        "warmup_steps": warmup_steps,
        "step_logs": [],
        "epoch_logs": [],
    }

    # Baseline Epoch 0 evaluation
    print("[TRAIN] Running Epoch 0 (baseline before fine-tuning) validation...")
    baseline_metrics = run_validation(
        model, processor, val_loader, device, amp_dtype, use_amp, language=args.language
    )
    baseline_entry = {
        "epoch": 0,
        "global_step": 0,
        "train_loss": None,
        **baseline_metrics,
    }
    history["epoch_logs"].append(baseline_entry)
    print(
        f"[EPOCH 0 / BASELINE] val_loss={baseline_metrics['val_loss']:.4f} | "
        f"val_WER={baseline_metrics['val_wer_pct']:.2f}% | val_CER={baseline_metrics['val_cer_pct']:.2f}%"
    )

    best_wer = float("inf")
    global_step = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        optimizer.zero_grad(set_to_none=True)
        epoch_t0 = time.time()

        for step_idx, batch in enumerate(train_loader, start=1):
            input_features = batch["input_features"].to(device)
            labels = batch["labels"].to(device)

            if args.spec_augment:
                input_features = apply_spec_augment(
                    input_features,
                    freq_mask_param=args.freq_mask_param,
                    time_mask_param=args.time_mask_param,
                )

            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / args.grad_accum_steps

            scaler.scale(loss).backward()
            raw_loss = float(outputs.loss.item())
            epoch_loss_sum += raw_loss
            epoch_batches += 1

            is_update_step = (step_idx % args.grad_accum_steps == 0) or (step_idx == len(train_loader))
            if is_update_step:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                # Only step scheduler if GradScaler did not skip optimizer.step()
                if not (use_amp and args.precision == "fp16") or (scaler.get_scale() >= scale_before):
                    scheduler.step()
                global_step += 1

                current_lr = scheduler.get_last_lr()[0]
                gn_val = float(grad_norm) if math.isfinite(float(grad_norm)) else 0.0
                history["step_logs"].append(
                    {
                        "global_step": global_step,
                        "epoch": round(epoch - 1 + step_idx / len(train_loader), 4),
                        "train_loss": round(raw_loss, 4),
                        "lr": float(f"{current_lr:.6e}"),
                        "grad_norm": round(gn_val, 4),
                    }
                )

                if global_step % args.log_every == 0 or global_step == total_opt_steps:
                    print(
                        f"  [Step {global_step}/{total_opt_steps} | Epoch {epoch}] "
                        f"loss={raw_loss:.4f} | lr={current_lr:.2e} | grad_norm={gn_val:.2f}"
                    )

        avg_train_loss = epoch_loss_sum / max(epoch_batches, 1)
        val_metrics = run_validation(
            model, processor, val_loader, device, amp_dtype, use_amp, language=args.language
        )
        epoch_time = time.time() - epoch_t0

        epoch_log = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(avg_train_loss, 4),
            "epoch_time_sec": round(epoch_time, 2),
            **val_metrics,
        }
        history["epoch_logs"].append(epoch_log)

        print(
            f"[EPOCH {epoch}/{args.epochs}] train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | val_WER={val_metrics['val_wer_pct']:.2f}% | "
            f"val_CER={val_metrics['val_cer_pct']:.2f}% | time={epoch_time:.1f}s"
        )

        # Save best checkpoint (or always save on epoch 1 so output_dir is guaranteed populated)
        if val_metrics["val_wer"] <= best_wer or epoch == 1:
            best_wer = val_metrics["val_wer"]
            model.save_pretrained(output_dir)
            processor.save_pretrained(output_dir)
            print(
                f"  -> Saved best checkpoint to {output_dir} (val_WER={val_metrics['val_wer_pct']:.2f}%)"
            )

        # Persist history after every epoch
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    total_time = time.time() - start_time
    history["total_training_time_sec"] = round(total_time, 2)
    history["best_val_wer"] = round(best_wer, 4)
    history["best_val_wer_pct"] = round(best_wer * 100.0, 2)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print(
        f"[TRAIN COMPLETE] Total time: {total_time:.1f}s | Best val WER: {best_wer * 100.0:.2f}% | "
        f"History saved to {history_path}"
    )

    if args.run_final_eval_report:
        eval_report_path = DEFAULT_BENCHMARKS_DIR / "eval_telugish-whisper-tiny-ft.json"
        evaluate_checkpoint(
            model_path=str(output_dir),
            eval_jsonl=str(args.eval_jsonl),
            output_json=str(eval_report_path),
            device=args.device,
            backend="hf",
            language=args.language,
            max_samples=args.max_eval_samples,
        )

    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune Whisper-Tiny on Telugish & Indian-accented English with mixed precision"
    )
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default=str(DEFAULT_BASE_MODEL),
        help="Path or HF hub ID of base Whisper checkpoint",
    )
    parser.add_argument(
        "--train-jsonl",
        type=str,
        default=str(DEFAULT_TRAIN_JSONL),
        help="Path to training JSONL file",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default=str(DEFAULT_EVAL_JSONL),
        help="Path to validation JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save fine-tuned model checkpoint",
    )
    parser.add_argument(
        "--history-json",
        type=str,
        default=str(DEFAULT_HISTORY_JSON),
        help="Path to save training & validation history JSON",
    )
    parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-step training batch size")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Validation batch size")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Peak learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio of total steps")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Explicit warmup steps override")
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="LR schedule type",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["fp16", "bf16", "fp32"],
        help="Mixed precision mode on CUDA (default: bf16 on CUDA GPU)",
    )
    parser.add_argument(
        "--spec-augment",
        action="store_true",
        default=True,
        help="Enable SpecAugment time/frequency masking on log-Mel features",
    )
    parser.add_argument(
        "--no-spec-augment",
        action="store_false",
        dest="spec_augment",
        help="Disable SpecAugment",
    )
    parser.add_argument("--freq-mask-param", type=int, default=10, help="Max frequency mask width")
    parser.add_argument("--time-mask-param", type=int, default=30, help="Max time mask width")
    parser.add_argument("--mask-time-prob", type=float, default=0.05, help="HF config mask_time_prob")
    parser.add_argument("--mask-feature-prob", type=float, default=0.05, help="HF config mask_feature_prob")
    parser.add_argument("--freeze-encoder", action="store_true", default=False, help="Freeze audio encoder")
    parser.add_argument("--language", type=str, default="en", help="Whisper language token")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker processes")
    parser.add_argument("--log-every", type=int, default=5, help="Log every N optimizer steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Cap training samples")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Cap eval samples")
    parser.add_argument(
        "--run-final-eval-report",
        action="store_true",
        default=True,
        help="Run full evaluate_model.py error analysis report on the best saved checkpoint",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)
