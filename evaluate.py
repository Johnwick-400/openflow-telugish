#!/usr/bin/env python3
"""
Evaluate OpenFlow-Telugish models (CTranslate2 INT8 or Hugging Face)
on the held-out CS-82 test benchmark.
"""

import json
import time
import re
import argparse
from pathlib import Path
import jiwer
from faster_whisper import WhisperModel

def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate OpenFlow-Telugish INT8 model")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to CTranslate2 INT8 model directory")
    parser.add_argument("--eval_jsonl", type=str, default="dataset/eval_heldout_cs82.jsonl")
    parser.add_argument("--beam_size", type=int, default=4)
    args = parser.parse_args()

    print(f"Loading model from {args.model_dir}...")
    model = WhisperModel(args.model_dir, device="cpu", compute_type="int8", cpu_threads=4)

    prompt = (
        "Natural Telugu-English code-switching in Roman script (Telugish). "
        "Hello mama, task, pending, escalation mail, office lo, meeting, deploy, PR, "
        "Cloud Run, GCP, Swiggy, UPI, GPay, PhonePe, payment, cheyali, chudu."
    )

    records = []
    with open(args.eval_jsonl) as f:
        for line in f:
            records.append(json.loads(line))

    print(f"Evaluating on {len(records)} held-out clips...")
    refs, hyps = [], []
    t0 = time.time()
    for r in records:
        fp = r.get("audio_filepath") or r.get("audio_path")
        ref = normalize_text(r["text"])
        segs, _ = model.transcribe(
            fp,
            language="en",
            beam_size=args.beam_size,
            initial_prompt=prompt,
            without_timestamps=True,
            vad_filter=False
        )
        hyp = normalize_text(" ".join(s.text.strip() for s in segs))
        refs.append(ref)
        hyps.append(hyp)

    wer = float(jiwer.wer(refs, hyps)) * 100.0
    cer = float(jiwer.cer(refs, hyps)) * 100.0
    elapsed = time.time() - t0

    print("=" * 50)
    print("EVALUATION RESULTS")
    print(f"Word Error Rate (WER):      {wer:.2f}%")
    print(f"Character Error Rate (CER): {cer:.2f}% (Accuracy: {100.0 - cer:.2f}%)")
    print(f"Evaluated {len(records)} utterances in {elapsed:.1f}s")
    print("=" * 50)

if __name__ == "__main__":
    main()
