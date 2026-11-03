#!/usr/bin/env python3
"""
Convert a HuggingFace Whisper checkpoint into an ultra-compact INT8 CTranslate2 / Faster-Whisper
edge model (~38 MB) in /home/pavan_veesam/telugish_whisper_edge/models/telugish-whisper-tiny-int8.

Benchmarks:
- Disk footprint (MB) of FP32/FP16 HuggingFace model vs INT8 CTranslate2 model
- Peak RAM / RSS memory footprint (MB) and GPU VRAM usage (MB)
- Inference latency (mean / p50 / p95 ms) and Real-Time Factor (RTF) on CPU & GPU
- Exact WER and CER comparison: Base Whisper-Tiny vs Fine-Tuned FP16 vs Fine-Tuned INT8 Edge
Saves full comparison report to /home/pavan_veesam/telugish_whisper_edge/benchmarks/edge_int8_benchmark.json.
"""

import argparse
import gc
import json
import os
import resource
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ctranslate2
import torch

from evaluate_model import (
    BASE_DIR,
    DEFAULT_BASE_MODEL,
    DEFAULT_BENCHMARKS_DIR,
    DEFAULT_EVAL_JSONL,
    evaluate_checkpoint,
    get_dir_size_mb,
)

DEFAULT_FT_MODEL = BASE_DIR / "models" / "telugish-whisper-tiny-ft"
DEFAULT_INT8_DIR = BASE_DIR / "models" / "telugish-whisper-tiny-int8"
DEFAULT_EDGE_REPORT = DEFAULT_BENCHMARKS_DIR / "edge_int8_benchmark.json"


def get_rss_mb() -> float:
    """Return current process max RSS in MB (Linux ru_maxrss is in KB)."""
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(float(usage_kb) / 1024.0, 2)


def convert_hf_to_ct2_int8(
    hf_model_dir: str,
    output_dir: str,
    quantization: str = "int8",
) -> Dict[str, Any]:
    """Convert HuggingFace Whisper checkpoint to CTranslate2 format with INT8 quantization."""
    hf_path = Path(hf_model_dir)
    out_path = Path(output_dir)

    if not hf_path.exists():
        raise FileNotFoundError(f"Source HuggingFace checkpoint not found: {hf_path}")

    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    copy_files = []
    for fname in ("tokenizer.json", "preprocessor_config.json", "processor_config.json", "vocab.json", "merges.txt"):
        if (hf_path / fname).exists():
            copy_files.append(fname)

    print(f"[EXPORT] Converting {hf_path} -> {out_path} (quantization={quantization})...")
    t0 = time.perf_counter()
    converter = ctranslate2.converters.TransformersConverter(
        str(hf_path),
        copy_files=copy_files,
        load_as_float16=(quantization in ("float16", "int8_float16")),
    )
    converter.convert(str(out_path), quantization=quantization, force=True)
    convert_time_s = time.perf_counter() - t0

    # Ensure preprocessor_config.json exists for faster-whisper compatibility
    if not (out_path / "preprocessor_config.json").exists() and (hf_path / "processor_config.json").exists():
        shutil.copy2(hf_path / "processor_config.json", out_path / "preprocessor_config.json")

    src_size_mb = get_dir_size_mb(str(hf_path))
    int8_size_mb = get_dir_size_mb(str(out_path))
    model_bin_mb = get_dir_size_mb(str(out_path / "model.bin"))
    compression_ratio = src_size_mb / max(int8_size_mb, 1e-4)

    info = {
        "source_hf_dir": str(hf_path),
        "output_int8_dir": str(out_path),
        "quantization": quantization,
        "conversion_time_sec": round(convert_time_s, 2),
        "source_dir_size_mb": round(src_size_mb, 2),
        "int8_dir_size_mb": round(int8_size_mb, 2),
        "int8_model_bin_mb": round(model_bin_mb, 2),
        "size_reduction_pct": round((1.0 - int8_size_mb / max(src_size_mb, 1e-4)) * 100.0, 2),
        "compression_ratio": round(compression_ratio, 2),
    }
    print(
        f"[EXPORT] Done in {convert_time_s:.2f}s | Source: {src_size_mb:.2f} MB -> "
        f"INT8 Dir: {int8_size_mb:.2f} MB (model.bin: {model_bin_mb:.2f} MB, "
        f"{info['size_reduction_pct']:.1f}% reduction, {compression_ratio:.2f}x smaller)"
    )
    return info


def export_and_benchmark(
    hf_model_dir: str = str(DEFAULT_FT_MODEL),
    int8_output_dir: str = str(DEFAULT_INT8_DIR),
    eval_jsonl: str = str(DEFAULT_EVAL_JSONL),
    report_json: str = str(DEFAULT_EDGE_REPORT),
    base_model_dir: Optional[str] = str(DEFAULT_BASE_MODEL),
    quantization: str = "int8",
    language: str = "en",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    # Fall back to base model if fine-tuned model does not exist yet (e.g. during smoke test)
    source_dir = hf_model_dir
    if not Path(source_dir).exists():
        print(f"[EXPORT] {source_dir} not found, falling back to {base_model_dir}")
        source_dir = str(base_model_dir)

    export_info = convert_hf_to_ct2_int8(
        hf_model_dir=source_dir,
        output_dir=int8_output_dir,
        quantization=quantization,
    )

    benchmarks: Dict[str, Any] = {}

    # 1. Evaluate Base Model (if available and requested)
    if base_model_dir and Path(base_model_dir).exists():
        base_out = DEFAULT_BENCHMARKS_DIR / "eval_whisper-tiny-base.json"
        base_rep = evaluate_checkpoint(
            model_path=str(base_model_dir),
            eval_jsonl=eval_jsonl,
            output_json=str(base_out),
            device="cuda" if torch.cuda.is_available() else "cpu",
            backend="hf",
            compute_type="float16",
            language=language,
            max_samples=max_samples,
        )
        benchmarks["whisper_tiny_base_fp16"] = {
            **base_rep["summary"],
            "model_size_mb": base_rep["metadata"]["model_size_mb"],
            "device": base_rep["metadata"]["device"],
            "backend": base_rep["metadata"]["backend"],
        }

    # 2. Evaluate Fine-Tuned HF FP16 Model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    ft_out = DEFAULT_BENCHMARKS_DIR / "eval_telugish-whisper-tiny-ft.json"
    ft_rep = evaluate_checkpoint(
        model_path=source_dir,
        eval_jsonl=eval_jsonl,
        output_json=str(ft_out),
        device="cuda" if torch.cuda.is_available() else "cpu",
        backend="hf",
        compute_type="float16",
        language=language,
        max_samples=max_samples,
    )
    gpu_peak_mb = (
        round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)
        if torch.cuda.is_available()
        else 0.0
    )
    benchmarks["finetuned_hf_fp16"] = {
        **ft_rep["summary"],
        "model_size_mb": ft_rep["metadata"]["model_size_mb"],
        "device": ft_rep["metadata"]["device"],
        "backend": ft_rep["metadata"]["backend"],
        "gpu_peak_vram_mb": gpu_peak_mb,
        "process_rss_mb": get_rss_mb(),
    }

    # 3. Evaluate INT8 CTranslate2 / Faster-Whisper on CPU (Edge target)
    int8_cpu_out = DEFAULT_BENCHMARKS_DIR / "eval_telugish-whisper-tiny-int8-cpu.json"
    int8_cpu_rep = evaluate_checkpoint(
        model_path=int8_output_dir,
        eval_jsonl=eval_jsonl,
        output_json=str(int8_cpu_out),
        device="cpu",
        backend="ct2",
        compute_type="int8",
        language=language,
        max_samples=max_samples,
    )
    benchmarks["finetuned_ct2_int8_cpu"] = {
        **int8_cpu_rep["summary"],
        "model_size_mb": export_info["int8_dir_size_mb"],
        "model_bin_mb": export_info["int8_model_bin_mb"],
        "device": "cpu",
        "backend": "faster_whisper_ctranslate2_int8",
        "process_rss_mb": get_rss_mb(),
    }

    wer_delta_int8_vs_fp16 = round(
        benchmarks["finetuned_ct2_int8_cpu"]["wer_pct"] - benchmarks["finetuned_hf_fp16"]["wer_pct"],
        2,
    )

    summary_report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "export": export_info,
        "benchmarks": benchmarks,
        "comparison": {
            "fp16_size_mb": export_info["source_dir_size_mb"],
            "int8_size_mb": export_info["int8_dir_size_mb"],
            "int8_model_bin_mb": export_info["int8_model_bin_mb"],
            "size_reduction_pct": export_info["size_reduction_pct"],
            "compression_ratio": export_info["compression_ratio"],
            "fp16_wer_pct": benchmarks["finetuned_hf_fp16"]["wer_pct"],
            "int8_cpu_wer_pct": benchmarks["finetuned_ct2_int8_cpu"]["wer_pct"],
            "wer_delta_pct_points": wer_delta_int8_vs_fp16,
            "fp16_mean_latency_ms": benchmarks["finetuned_hf_fp16"]["mean_latency_ms"],
            "int8_cpu_mean_latency_ms": benchmarks["finetuned_ct2_int8_cpu"]["mean_latency_ms"],
            "int8_cpu_rtf": benchmarks["finetuned_ct2_int8_cpu"]["rtf"],
        },
    }

    if "whisper_tiny_base_fp16" in benchmarks:
        base_wer = benchmarks["whisper_tiny_base_fp16"]["wer_pct"]
        ft_wer = benchmarks["finetuned_hf_fp16"]["wer_pct"]
        summary_report["comparison"]["base_wer_pct"] = base_wer
        summary_report["comparison"]["absolute_wer_improvement_pct_points"] = round(base_wer - ft_wer, 2)
        summary_report["comparison"]["relative_wer_reduction_pct"] = (
            round((base_wer - ft_wer) / max(base_wer, 1e-4) * 100.0, 2) if base_wer > 0 else 0.0
        )

    out_file = Path(report_json)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print(f"[EXPORT BENCHMARK SAVED] {out_file}")
    return summary_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export fine-tuned Whisper checkpoint to CTranslate2 INT8 edge format & benchmark"
    )
    parser.add_argument(
        "--hf-model-dir",
        type=str,
        default=str(DEFAULT_FT_MODEL),
        help="Path to fine-tuned HuggingFace Whisper checkpoint",
    )
    parser.add_argument(
        "--int8-output-dir",
        type=str,
        default=str(DEFAULT_INT8_DIR),
        help="Output directory for CTranslate2 INT8 model",
    )
    parser.add_argument(
        "--base-model-dir",
        type=str,
        default=str(DEFAULT_BASE_MODEL),
        help="Path to base Whisper-Tiny checkpoint for 3-way comparison",
    )
    parser.add_argument(
        "--eval-jsonl",
        type=str,
        default=str(DEFAULT_EVAL_JSONL),
        help="Evaluation JSONL dataset path",
    )
    parser.add_argument(
        "--report-json",
        type=str,
        default=str(DEFAULT_EDGE_REPORT),
        help="Output JSON path for edge quantization benchmark report",
    )
    parser.add_argument(
        "--quantization",
        type=str,
        default="int8",
        choices=["int8", "int8_float16", "float16", "float32"],
        help="CTranslate2 quantization type",
    )
    parser.add_argument("--language", type=str, default="en", help="Whisper decoding language token")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on evaluation samples")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_and_benchmark(
        hf_model_dir=args.hf_model_dir,
        int8_output_dir=args.int8_output_dir,
        eval_jsonl=args.eval_jsonl,
        report_json=args.report_json,
        base_model_dir=args.base_model_dir,
        quantization=args.quantization,
        language=args.language,
        max_samples=args.max_samples,
    )
