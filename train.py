#!/usr/bin/env python3
"""
OpenFlow-Telugish Training Engine
Fine-tunes Whisper on code-switched Telugu-English (Telugish) with teacher distillation,
acoustic augmentations (SpecAugment, dynamic gain jitter, ambient noise injection),
and automatic CTranslate2 INT8 quantization export.
"""

import os
import sys
import json
import time
import shutil
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
import jiwer
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    get_cosine_schedule_with_warmup
)
from ctranslate2.converters import TransformersConverter
from faster_whisper import WhisperModel

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_audio_16k(audio_path: str, augment: bool = False):
    audio, sr = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

    if augment:
        # Dynamic Gain scaling (0.75x to 1.25x)
        gain = random.uniform(0.75, 1.25)
        audio = audio * gain

        # Additive ambient room noise injection (40% probability, SNR 20-30 dB)
        if random.random() < 0.40:
            rms_sig = np.sqrt(np.mean(audio**2) + 1e-9)
            target_snr_db = random.uniform(20.0, 30.0)
            rms_noise = rms_sig / (10.0 ** (target_snr_db / 20.0))
            noise = np.random.normal(0, rms_noise, audio.shape).astype(np.float32)
            audio = audio + noise

    audio = np.clip(audio, -1.0, 1.0)
    dur = len(audio) / 16000.0
    return audio, dur

class TelugishDataset(Dataset):
    def __init__(self, jsonl_path: Path, processor: WhisperProcessor, is_train: bool = True):
        self.processor = processor
        self.is_train = is_train
        self.records = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        print(f"Loaded {len(self.records)} records from {jsonl_path} (augment={is_train})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        fp = rec.get("audio_filepath") or rec.get("audio_path")
        text = rec["text"].strip()
        audio, dur = load_audio_16k(fp, augment=self.is_train)
        inputs = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        labels = self.processor.tokenizer(text, truncation=True, max_length=448).input_ids
        return {
            "input_features": inputs.input_features[0],
            "labels": labels,
            "text": text,
            "duration": dur
        }

@dataclass
class WhisperCollator:
    processor: WhisperProcessor

    def __call__(self, features: List[Dict[str, Any]]):
        input_features = torch.stack([f["input_features"] for f in features])
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        bos_id = self.processor.tokenizer.bos_token_id
        if labels.shape[1] > 0 and bos_id is not None and (labels[:, 0] == bos_id).all().cpu().item():
            labels = labels[:, 1:]
        return {
            "input_features": input_features,
            "labels": labels,
            "texts": [f["text"] for f in features],
            "durations": [f["duration"] for f in features]
        }

def apply_spec_augment(mel_batch: torch.Tensor, freq_mask: int = 14, time_mask: int = 40):
    B, F, T = mel_batch.shape
    aug = mel_batch.clone()
    for b in range(B):
        mean_val = aug[b].mean()
        for _ in range(2):
            f = random.randint(0, freq_mask)
            if f > 0 and F - f > 0:
                f0 = random.randint(0, F - f)
                aug[b, f0 : f0 + f, :] = mean_val
        for _ in range(2):
            t = random.randint(0, time_mask)
            if t > 0 and T - t > 0:
                t0 = random.randint(0, T - t)
                aug[b, :, t0 : t0 + t] = mean_val
    return aug

def main():
    parser = argparse.ArgumentParser(description="Train OpenFlow-Telugish ASR model")
    parser.add_argument("--train_jsonl", type=str, default="dataset/train_telugish_distilled_2500.jsonl")
    parser.add_argument("--eval_jsonl", type=str, default="dataset/eval_heldout_cs82.jsonl")
    parser.add_argument("--output_dir", type=str, default="models/openflow-small-int8")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3.2e-5)
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} (CUDA: {torch.cuda.is_available()})")

    # 1. Load Processor & Base Model
    print("Loading base whisper-small backbone...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", torch_dtype=torch.float32)

    # 2. Infuse 30% Telugu acoustic encoder weights
    print("Infusing 30% vasista22/whisper-telugu-small acoustic encoder representations...")
    try:
        vasista = WhisperForConditionalGeneration.from_pretrained("vasista22/whisper-telugu-small", torch_dtype=torch.float32)
        sd_base = model.state_dict()
        sd_vas = vasista.state_dict()
        blended = {}
        for k in sd_base:
            if "encoder." in k and k in sd_vas:
                blended[k] = 0.30 * sd_vas[k] + 0.70 * sd_base[k]
            else:
                blended[k] = sd_base[k]
        model.load_state_dict(blended)
        del vasista
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Could not blend Telugu encoder: {e}. Training standard backbone.")

    model.to(device)

    # 3. Setup Dataset
    train_path = Path(args.train_jsonl)
    dataset = TelugishDataset(train_path, processor, is_train=True)
    collator = WhisperCollator(processor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator, num_workers=4, pin_memory=True)

    # 4. Training Loop
    epochs = args.epochs
    grad_accum_steps = 2
    total_steps = (len(loader) // grad_accum_steps) * epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=60, num_training_steps=total_steps)

    print(f"Training for {epochs} epochs ({total_steps} optimizer updates)...")
    step = 0
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        for b_idx, batch in enumerate(loader):
            input_features = batch["input_features"]
            input_features = apply_spec_augment(input_features)
            input_features = input_features.to(device)
            labels = batch["labels"].to(device)

            use_amp = (device.type == "cuda")
            with torch.amp.autocast(device_type="cuda" if use_amp else "cpu", enabled=use_amp, dtype=torch.bfloat16):
                outputs = model(input_features=input_features, labels=labels)
                loss = outputs.loss / grad_accum_steps

            loss.backward()
            running_loss += loss.item() * grad_accum_steps

            if (b_idx + 1) % grad_accum_steps == 0 or (b_idx + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % 50 == 0 or step == total_steps:
                    lr = scheduler.get_last_lr()[0]
                    avg_loss = running_loss / (b_idx % 50 + 1)
                    print(f"  [Epoch {epoch}/{epochs} | Step {step}/{total_steps}] Loss: {avg_loss:.4f} | LR: {lr:.2e} | Elapsed: {time.time()-t_start:.1f}s")
                    running_loss = 0.0

    # 5. Export Hugging Face Checkpoint & Quantize to INT8
    hf_tmp = Path(args.output_dir + "_tmp_hf")
    hf_tmp.mkdir(parents=True, exist_ok=True)
    print(f"Saving temporary checkpoint to {hf_tmp}...")
    model.save_pretrained(str(hf_tmp))
    processor.save_pretrained(str(hf_tmp))
    processor.feature_extractor.save_pretrained(str(hf_tmp))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    int8_dir = Path(args.output_dir)
    print(f"Quantizing to CTranslate2 INT8 at {int8_dir}...")
    if int8_dir.exists():
        shutil.rmtree(int8_dir)
    int8_dir.mkdir(parents=True, exist_ok=True)

    converter = TransformersConverter(str(hf_tmp), copy_files=["tokenizer.json", "preprocessor_config.json"])
    converter.convert(str(int8_dir), quantization="int8", force=True)

    shutil.rmtree(hf_tmp, ignore_errors=True)
    print(f"Successfully exported INT8 engine to {int8_dir}")

if __name__ == "__main__":
    main()
