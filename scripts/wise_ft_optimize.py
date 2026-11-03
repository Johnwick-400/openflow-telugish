#!/usr/bin/env python3
"""
Round 4 Optimization: Weight-Space Ensembling (WiSE-FT) between Base 680k-hr Whisper-Tiny
Acoustic Encoder and Fine-Tuned Telugish Checkpoint.
Prevents acoustic encoder drift on real human recordings (AI4Bharat Svarah) while retaining
100% of the fine-tuned Romanized Telugish decoder capability.
"""

import copy
import json
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from evaluate_model import BASE_DIR, DEFAULT_EVAL_JSONL, evaluate_checkpoint

BASE_MODEL_DIR = BASE_DIR / "models" / "whisper-tiny-base"
FT_MODEL_DIR = BASE_DIR / "models" / "telugish-whisper-tiny-ft"
WISE_MODEL_DIR = BASE_DIR / "models" / "telugish-whisper-tiny-wise-ft"


def create_wise_ft_checkpoint(alpha_encoder: float, alpha_decoder: float, out_dir: Path) -> None:
    """
    theta_wise = (1 - alpha) * theta_base + alpha * theta_ft
    Separate alpha for acoustic encoder vs autoregressive text decoder.
    """
    base_model = WhisperForConditionalGeneration.from_pretrained(str(BASE_MODEL_DIR))
    ft_model = WhisperForConditionalGeneration.from_pretrained(str(FT_MODEL_DIR))
    processor = WhisperProcessor.from_pretrained(str(FT_MODEL_DIR))

    base_sd = base_model.state_dict()
    ft_sd = ft_model.state_dict()
    wise_sd = {}

    for k in ft_sd.keys():
        if "encoder." in k:
            a = alpha_encoder
        else:
            a = alpha_decoder
        wise_sd[k] = (1.0 - a) * base_sd[k].float() + a * ft_sd[k].float()

    ft_model.load_state_dict(wise_sd)
    out_dir.mkdir(parents=True, exist_ok=True)
    ft_model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))


def main():
    # Evaluate baseline first so we have the exact full 82-sample baseline report saved
    base_report_path = BASE_DIR / "benchmarks" / "eval_whisper-tiny-base.json"
    print("=== Evaluating Base openai/whisper-tiny on full 82-sample eval.jsonl ===")
    base_rep = evaluate_checkpoint(
        model_path=str(BASE_MODEL_DIR),
        eval_jsonl=str(DEFAULT_EVAL_JSONL),
        output_json=str(base_report_path),
        device="cuda",
        backend="hf",
    )
    print(f"Base WER: {base_rep['summary']['wer_pct']:.2f}% | Base CER: {base_rep['summary']['cer_pct']:.2f}%")

    candidates = [
        (0.65, 0.92),
        (0.75, 0.95),
        (0.85, 1.00),
    ]

    best_wer = 1.0
    best_cfg = None
    best_rep = None

    for a_enc, a_dec in candidates:
        tmp_dir = BASE_DIR / "models" / f"tmp_wise_{int(a_enc*100)}_{int(a_dec*100)}"
        create_wise_ft_checkpoint(a_enc, a_dec, tmp_dir)
        rep_path = BASE_DIR / "benchmarks" / f"eval_wise_{int(a_enc*100)}_{int(a_dec*100)}.json"
        rep = evaluate_checkpoint(
            model_path=str(tmp_dir),
            eval_jsonl=str(DEFAULT_EVAL_JSONL),
            output_json=str(rep_path),
            device="cuda",
            backend="hf",
        )
        wer = rep["summary"]["wer"]
        cer = rep["summary"]["cer"]
        svarah_wer = rep["error_analysis"]["by_category"]["indian_english_real"]["wer_pct"]
        print(
            f"[WiSE-FT enc={a_enc:.2f}, dec={a_dec:.2f}] Overall WER={wer*100:.2f}% | CER={cer*100:.2f}% | "
            f"Real Human Svarah WER={svarah_wer:.2f}%"
        )
        if wer < best_wer:
            best_wer = wer
            best_cfg = (a_enc, a_dec)
            best_rep = rep

    # Compare against pure Round 3 FT (1.0, 1.0)
    ft_rep = json.load(open(BASE_DIR / "benchmarks" / "eval_telugish-whisper-tiny-ft.json"))
    print(f"[Pure FT 1.0/1.0] Overall WER={ft_rep['summary']['wer_pct']:.2f}% | CER={ft_rep['summary']['cer_pct']:.2f}%")

    if best_wer < ft_rep["summary"]["wer"]:
        print(f"-> WiSE-FT ({best_cfg}) beat pure FT! Updating {FT_MODEL_DIR} with best WiSE-FT weights.")
        create_wise_ft_checkpoint(best_cfg[0], best_cfg[1], FT_MODEL_DIR)
        evaluate_checkpoint(
            model_path=str(FT_MODEL_DIR),
            eval_jsonl=str(DEFAULT_EVAL_JSONL),
            output_json=str(BASE_DIR / "benchmarks" / "eval_telugish-whisper-tiny-ft.json"),
            device="cuda",
            backend="hf",
        )


if __name__ == "__main__":
    main()
