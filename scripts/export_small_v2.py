#!/usr/bin/env python3
"""
WiSE-FT + INT8 Quantization + Full Evaluation for V2 Telugish-Whisper-Small (Trained on Real Human FLEURS + Svarah + Code-Switched Telugish).
"""

import json
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from evaluate_model import BASE_DIR, evaluate_checkpoint
from export_edge_int8 import convert_hf_to_ct2_int8

SMALL_BASE = BASE_DIR / "models" / "whisper-small-base"
SMALL_FT = BASE_DIR / "models" / "telugish-whisper-small-ft"
SMALL_WISE = BASE_DIR / "models" / "telugish-whisper-small-wise"
SMALL_INT8 = BASE_DIR / "models" / "telugish-whisper-small-int8"

EVAL_CS = BASE_DIR / "data" / "eval.jsonl"
EVAL_V2 = BASE_DIR / "data" / "eval_v2_real.jsonl"


def main():
    print("1. Loading base & fine-tuned Whisper-Small models for WiSE-FT...")
    proc = WhisperProcessor.from_pretrained(str(SMALL_FT))
    base_sd = WhisperForConditionalGeneration.from_pretrained(str(SMALL_BASE), torch_dtype=torch.float32).state_dict()
    ft_model = WhisperForConditionalGeneration.from_pretrained(str(SMALL_FT), torch_dtype=torch.float32)
    ft_sd = ft_model.state_dict()

    # Blend 15% of OpenAI's 680k-hour acoustic encoder into our fine-tuned encoder (keep 100% of FT decoder)
    alpha_enc = 0.85
    alpha_dec = 1.00
    blended = {}
    for k, v_ft in ft_sd.items():
        v_base = base_sd[k]
        a = alpha_enc if "encoder." in k else alpha_dec
        blended[k] = (1.0 - a) * v_base + a * v_ft

    ft_model.load_state_dict(blended)
    SMALL_WISE.mkdir(parents=True, exist_ok=True)
    ft_model.half().save_pretrained(str(SMALL_WISE))
    proc.save_pretrained(str(SMALL_WISE))
    print(f"Saved WiSE-FT Whisper-Small checkpoint to {SMALL_WISE}")

    print("\n2. Evaluating WiSE-FT Whisper-Small on 82-sample Code-Switched + Svarah Benchmark (eval.jsonl)...")
    rep_cs = evaluate_checkpoint(
        model_path=str(SMALL_WISE),
        eval_jsonl=str(EVAL_CS),
        output_json=str(BASE_DIR / "benchmarks" / "eval_whisper_small_cs82.json"),
        backend="hf",
        device="cuda",
    )

    print("\n3. Evaluating WiSE-FT Whisper-Small on 120-sample V2 Real-Human Benchmark (eval_v2_real.jsonl)...")
    rep_v2 = evaluate_checkpoint(
        model_path=str(SMALL_WISE),
        eval_jsonl=str(EVAL_V2),
        output_json=str(BASE_DIR / "benchmarks" / "eval_whisper_small_v2_real120.json"),
        backend="hf",
        device="cuda",
    )

    print("\n4. Quantizing WiSE-FT Whisper-Small to INT8 CTranslate2 Edge Format...")
    convert_hf_to_ct2_int8(SMALL_WISE, SMALL_INT8, quantization="int8")

    model_bin_mb = (SMALL_INT8 / "model.bin").stat().st_size / (1024 * 1024)
    print(f"\n✅ Quantized Whisper-Small INT8 Size: {model_bin_mb:.2f} MB")
    print(f"✅ 82-Sample Code-Switched + Svarah Eval: WER = {rep_cs['overall']['wer_pct']:.2f}% | CER = {rep_cs['overall']['cer_pct']:.2f}%")
    print(f"✅ 120-Sample V2 Real-Human + FLEURS Eval: WER = {rep_v2['overall']['wer_pct']:.2f}% | CER = {rep_v2['overall']['cer_pct']:.2f}%")
    print("\nBreakdown by Category (120-Sample V2 Eval):")
    for cat, stats in rep_v2["by_category"].items():
        print(f"  - {cat:30s}: WER = {stats['wer_pct']:6.2f}% | CER = {stats['cer_pct']:5.2f}%")


if __name__ == "__main__":
    main()
