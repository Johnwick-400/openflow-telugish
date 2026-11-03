#!/usr/bin/env python3
"""
Evaluate a Whisper model (HuggingFace checkpoint or CTranslate2 / Faster-Whisper INT8 model)
on a JSONL dataset (default: /home/pavan_veesam/telugish_whisper_edge/data/eval.jsonl).

Computes:
- Exact Corpus & Per-Sample WER (Word Error Rate) and CER (Character Error Rate)
- RTF (Real-Time Factor) and average latency per sample / per audio second
- Breakdown by category, noise condition, and speaker / source
- Worst-performing failure cases (error analysis)
Saves full JSON report to /home/pavan_veesam/telugish_whisper_edge/benchmarks/.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer
import librosa
import numpy as np
import soundfile as sf
import torch


BASE_DIR = Path("/home/pavan_veesam/telugish_whisper_edge")
DEFAULT_EVAL_JSONL = BASE_DIR / "data" / "eval.jsonl"
DEFAULT_BENCHMARKS_DIR = BASE_DIR / "benchmarks"
DEFAULT_BASE_MODEL = BASE_DIR / "models" / "whisper-tiny-base"


def normalize_text(text: str) -> str:
    """Normalize text for fair ASR WER/CER evaluation while preserving code-switched tokens."""
    if not text:
        return ""
    text = text.lower().strip()
    # Replace common punctuation with space except apostrophes inside words
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Canonicalize standard compound words (OpenAI Whisper EnglishTextNormalizer convention)
    compounds = {
        "screen shot": "screenshot",
        "cut off": "cutoff",
        "mid night": "midnight",
        "phone pay": "phonepe",
        "g pay": "gpay",
        "wi fi": "wifi",
        "back end": "backend",
        "front end": "frontend",
        "data base": "database",
    }
    for k, v in compounds.items():
        text = re.sub(rf"\b{k}\b", v, text)
    return text


def get_audio_path(entry: Dict[str, Any], jsonl_dir: Path) -> str:
    """Resolve audio file path from a JSONL record."""
    raw_path = (
        entry.get("audio_path")
        or entry.get("audio_filepath")
        or entry.get("audio")
        or entry.get("path")
        or entry.get("file")
    )
    if isinstance(raw_path, dict):
        raw_path = raw_path.get("path")
    if not raw_path:
        raise ValueError(f"No audio path key found in record: {entry}")
    p = Path(raw_path)
    if p.is_Absolute() if hasattr(p, "is_Absolute") else p.is_absolute():
        if p.exists():
            return str(p)
    # Try relative to BASE_DIR or jsonl_dir
    for candidate in [BASE_DIR / raw_path, jsonl_dir / raw_path, Path(raw_path)]:
        if candidate.exists():
            return str(candidate.resolve())
    return str(raw_path)


def get_reference_text(entry: Dict[str, Any]) -> str:
    """Extract reference transcript from a JSONL record."""
    for key in ("text", "sentence", "transcript", "transcription", "reference"):
        if key in entry and entry[key] is not None:
            return str(entry[key]).strip()
    raise ValueError(f"No reference transcript key found in record: {entry}")


def load_audio_16k(audio_path: str) -> Tuple[np.ndarray, float]:
    """Load audio file as 16kHz mono float32 numpy array and return (array, duration_sec)."""
    try:
        audio, sr = sf.read(audio_path, dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    except Exception:
        audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    audio = np.asarray(audio, dtype=np.float32)
    duration = float(len(audio)) / 16000.0
    return audio, max(duration, 1e-4)


def is_ctranslate2_model(model_path: str) -> bool:
    """Detect whether model_path is a CTranslate2 / Faster-Whisper directory."""
    p = Path(model_path)
    return p.is_dir() and (p / "model.bin").exists() and not (p / "model.safetensors").exists()


def get_dir_size_mb(path: str) -> float:
    """Compute directory size in MB."""
    p = Path(path)
    if not p.exists():
        return 0.0
    if p.is_file():
        return p.stat().st_size / (1024 * 1024)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def evaluate_hf_model(
    model_path: str,
    records: List[Dict[str, Any]],
    jsonl_dir: Path,
    device: str = "cuda",
    fp16: bool = True,
    language: Optional[str] = "en",
    Processor_fallback: str = str(DEFAULT_BASE_MODEL),
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    try:
        processor = WhisperProcessor.from_pretrained(model_path)
    except Exception:
        processor = WhisperProcessor.from_pretrained(Processor_fallback)

    dtype = torch.float16 if (fp16 and device == "cuda" and torch.cuda.is_available()) else torch.float32
    model = WhisperForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    target_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model.to(target_device)
    model.eval()

    # Configure generation cleanly for English/Telugish Roman script transcription
    if hasattr(model, "generation_config"):
        model.generation_config.forced_decoder_ids = None

    # Warmup pass
    dummy_audio = np.zeros(16000, dtype=np.float32)
    with torch.no_grad():
        inputs = processor(dummy_audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(target_device, dtype=dtype)
        gen_kwargs = {"max_new_tokens": 64}
        if language:
            gen_kwargs["language"] = language
            gen_kwargs["task"] = "transcribe"
        _ = model.generate(input_features, **gen_kwargs)
        if target_device == "cuda":
            torch.cuda.synchronize()

    results = []
    for idx, rec in enumerate(records):
        audio_path = get_audio_path(rec, jsonl_dir)
        ref_raw = get_reference_text(rec)
        audio, duration = load_audio_16k(audio_path)

        if target_device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(target_device, dtype=dtype)
            gen_kwargs = {"max_new_tokens": 128}
            if language:
                gen_kwargs["language"] = language
                gen_kwargs["task"] = "transcribe"
            pred_ids = model.generate(input_features, **gen_kwargs)
            hyp_raw = processor.batch_decode(
                pred_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

        if target_device == "cuda":
            torch.cuda.synchronize()
        latency_s = time.perf_counter() - t0

        results.append(
            build_sample_result(idx, rec, audio_path, duration, ref_raw, hyp_raw, latency_s)
        )

    meta = {
        "backend": "huggingface_transformers",
        "device": target_device,
        "dtype": str(dtype),
        "model_size_mb": round(get_dir_size_mb(model_path), 2),
    }
    return results, meta


def evaluate_ct2_model(
    model_path: str,
    records: List[Dict[str, Any]],
    jsonl_dir: Path,
    device: str = "cpu",
    compute_type: str = "int8",
    language: Optional[str] = "en",
    beam_size: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from faster_whisper import WhisperModel

    target_device = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model = WhisperModel(model_path, device=target_device, compute_type=compute_type, cpu_threads=4)

    # Warmup pass
    dummy_audio = np.zeros(16000, dtype=np.float32)
    segments, _ = model.transcribe(dummy_audio, language=language, beam_size=beam_size)
    _ = list(segments)

    results = []
    for idx, rec in enumerate(records):
        audio_path = get_audio_path(rec, jsonl_dir)
        ref_raw = get_reference_text(rec)
        audio, duration = load_audio_16k(audio_path)

        t0 = time.perf_counter()
        segments, _ = model.transcribe(
            audio,
            language=language,
            beam_size=beam_size,
            without_timestamps=True,
            condition_on_previous_text=False,
        )
        hyp_raw = " ".join(seg.text.strip() for seg in segments).strip()
        latency_s = time.perf_counter() - t0

        results.append(
            build_sample_result(idx, rec, audio_path, duration, ref_raw, hyp_raw, latency_s)
        )

    meta = {
        "backend": "faster_whisper_ctranslate2",
        "device": target_device,
        "compute_type": compute_type,
        "beam_size": beam_size,
        "model_size_mb": round(get_dir_size_mb(model_path), 2),
    }
    return results, meta


def build_sample_result(
    idx: int,
    rec: Dict[str, Any],
    audio_path: str,
    duration: float,
    ref_raw: str,
    hyp_raw: str,
    latency_s: float,
) -> Dict[str, Any]:
    ref_norm = normalize_text(ref_raw)
    hyp_norm = normalize_text(hyp_raw)

    if ref_norm:
        sample_wer = float(jiwer.wer(ref_norm, hyp_norm))
        sample_cer = float(jiwer.cer(ref_norm, hyp_norm))
    else:
        sample_wer = 0.0 if not hyp_norm else 1.0
        sample_cer = 0.0 if not hyp_norm else 1.0

    category = (
        rec.get("category")
        or rec.get("domain")
        or rec.get("subset")
        or rec.get("accent")
        or "general"
    )
    noise = (
        rec.get("noise_type")
        or rec.get("noise_condition")
        or rec.get("noise")
        or rec.get("snr_bucket")
        or ("noisy" if rec.get("is_noisy") else "clean")
    )
    speaker = (
        rec.get("speaker")
        or rec.get("speaker_id")
        or rec.get("voice")
        or rec.get("source")
        or "unknown"
    )

    return {
        "id": rec.get("id", f"sample_{idx:04d}"),
        "audio_path": audio_path,
        "duration_sec": round(duration, 3),
        "reference_raw": ref_raw,
        "hypothesis_raw": hyp_raw,
        "reference_norm": ref_norm,
        "hypothesis_norm": hyp_norm,
        "wer": round(sample_wer, 4),
        "cer": round(sample_cer, 4),
        "latency_ms": round(latency_s * 1000.0, 2),
        "rtf": round(latency_s / max(duration, 1e-4), 4),
        "category": str(category),
        "noise": str(noise),
        "speaker": str(speaker),
    }


def compute_group_metrics(samples: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for s in samples:
        grouped[str(s.get(key, "unknown"))].append(s)

    out = {}
    for group_name, items in sorted(grouped.items()):
        refs = [x["reference_norm"] for x in items if x["reference_norm"]]
        hyps = [x["hypothesis_norm"] for x in items if x["reference_norm"]]
        corpus_wer = float(jiwer.wer(refs, hyps)) if refs else 0.0
        corpus_cer = float(jiwer.cer(refs, hyps)) if refs else 0.0
        total_dur = sum(x["duration_sec"] for x in items)
        total_lat_s = sum(x["latency_ms"] / 1000.0 for x in items)
        out[group_name] = {
            "count": len(items),
            "wer": round(corpus_wer, 4),
            "wer_pct": round(corpus_wer * 100.0, 2),
            "cer": round(corpus_cer, 4),
            "cer_pct": round(corpus_cer * 100.0, 2),
            "mean_latency_ms": round(float(np.mean([x["latency_ms"] for x in items])), 2),
            "rtf": round(total_lat_s / max(total_dur, 1e-4), 4),
        }
    return out


def evaluate_checkpoint(
    model_path: str,
    eval_jsonl: str = str(DEFAULT_EVAL_JSONL),
    output_json: Optional[str] = None,
    device: str = "cuda",
    backend: str = "auto",
    compute_type: str = "int8",
    language: Optional[str] = "en",
    top_k_errors: int = 15,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    eval_path = Path(eval_jsonl)
    if not eval_path.exists():
        raise FileNotFoundError(f"Evaluation dataset not found: {eval_path}")

    records = []
    with open(eval_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]

    if not records:
        raise ValueError(f"No evaluation records found in {eval_path}")

    use_ct2 = (backend == "ct2") or (backend == "auto" and is_ctranslate2_model(model_path))

    if use_ct2:
        # CTranslate2 edge INT8 runs natively on CPU when system libcublas.so.12 is not in LD_LIBRARY_PATH
        ct2_device = "cpu" if backend == "auto" else device
        try:
            samples, meta = evaluate_ct2_model(
                model_path=model_path,
                records=records,
                jsonl_dir=eval_path.parent,
                device=ct2_device,
                compute_type=compute_type,
                language=language,
            )
        except RuntimeError as e:
            if "libcublas" in str(e) and ct2_device != "cpu":
                samples, meta = evaluate_ct2_model(
                    model_path=model_path,
                    records=records,
                    jsonl_dir=eval_path.parent,
                    device="cpu",
                    compute_type="int8",
                    language=language,
                )
            else:
                raise
    else:
        samples, meta = evaluate_hf_model(
            model_path=model_path,
            records=records,
            jsonl_dir=eval_path.parent,
            device=device,
            fp16=(compute_type != "float32"),
            language=language,
        )

    valid_refs = [s["reference_norm"] for s in samples if s["reference_norm"]]
    valid_hyps = [s["hypothesis_norm"] for s in samples if s["reference_norm"]]

    corpus_wer = float(jiwer.wer(valid_refs, valid_hyps)) if valid_refs else 0.0
    corpus_cer = float(jiwer.cer(valid_refs, valid_hyps)) if valid_refs else 0.0
    total_audio_sec = sum(s["duration_sec"] for s in samples)
    total_inference_sec = sum(s["latency_ms"] / 1000.0 for s in samples)
    overall_rtf = total_inference_sec / max(total_audio_sec, 1e-4)
    latencies = [s["latency_ms"] for s in samples]

    worst_cases = sorted(samples, key=lambda x: (x["wer"], x["cer"]), reverse=True)[:top_k_errors]
    best_cases = sorted(samples, key=lambda x: (x["wer"], x["cer"]))[: min(5, len(samples))]

    report = {
        "model_path": str(model_path),
        "eval_dataset": str(eval_path),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": meta,
        "summary": {
            "num_samples": len(samples),
            "total_audio_sec": round(total_audio_sec, 2),
            "total_inference_sec": round(total_inference_sec, 3),
            "wer": round(corpus_wer, 4),
            "wer_pct": round(corpus_wer * 100.0, 2),
            "cer": round(corpus_cer, 4),
            "cer_pct": round(corpus_cer * 100.0, 2),
            "rtf": round(overall_rtf, 4),
            "speedup_vs_realtime": round(1.0 / max(overall_rtf, 1e-6), 2),
            "mean_latency_ms": round(float(np.mean(latencies)), 2),
            "p50_latency_ms": round(float(np.percentile(latencies, 50)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
        },
        "error_analysis": {
            "by_category": compute_group_metrics(samples, "category"),
            "by_noise": compute_group_metrics(samples, "noise"),
            "by_speaker": compute_group_metrics(samples, "speaker"),
            "worst_failures": worst_cases,
            "best_predictions": best_cases,
        },
        "predictions": samples,
    }

    if output_json:
        out_path = Path(output_json)
    else:
        DEFAULT_BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
        model_tag = Path(model_path).name.replace(" ", "_")
        out_path = DEFAULT_BENCHMARKS_DIR / f"eval_{model_tag}.json"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        f"[EVAL] Model: {model_path} ({meta['backend']} on {meta['device']}) | "
        f"Samples: {len(samples)} | WER: {report['summary']['wer_pct']:.2f}% | "
        f"CER: {report['summary']['cer_pct']:.2f}% | RTF: {report['summary']['rtf']:.4f} "
        f"({report['summary']['mean_latency_ms']:.1f} ms/sample) -> Saved to {out_path}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Whisper or Faster-Whisper INT8 model on Telugish eval.jsonl")
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_BASE_MODEL),
        help="Path to HuggingFace checkpoint or CTranslate2 INT8 directory",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default=str(DEFAULT_EVAL_JSONL),
        help="Path to evaluation JSONL file",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to output JSON report (default: benchmarks/eval_<model_name>.json)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "hf", "ct2"],
        help="Inference backend: auto, hf (Transformers), or ct2 (Faster-Whisper/CTranslate2)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="int8",
        help="Compute type for CTranslate2 (int8, float16, float32) or HF (float16, float32)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language token for Whisper decoding (default: en for Romanized Telugish/Indian English)",
    )
    parser.add_argument(
        "--top-k-errors",
        type=int,
        default=15,
        help="Number of worst failure cases to store in error_analysis",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of samples to evaluate",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_checkpoint(
        model_path=args.model_path,
        eval_jsonl=args.eval_jsonl,
        output_json=args.output_json,
        device=args.device,
        backend=args.backend,
        compute_type=args.compute_type,
        language=args.language if args.language.lower() != "none" else None,
        top_k_errors=args.top_k_errors,
        max_samples=args.max_samples,
    )
