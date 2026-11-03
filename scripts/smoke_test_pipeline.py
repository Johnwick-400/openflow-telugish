#!/usr/bin/env python3
"""
Smoke test all 3 pipeline scripts (evaluate_model.py, train_whisper.py, export_edge_int8.py)
on a 2-sample dummy Telugish dataset in /tmp/telugish_smoke_test to verify zero syntax/API/library bugs
without overwriting the main dataset or benchmark artifacts.
"""

import asyncio
import json
import shutil
import subprocess
import sys
from pathlib import Path

import edge_tts
import librosa
import soundfile as sf


SMOKE_DIR = Path("/tmp/telugish_smoke_test")
PYTHON_BIN = "/home/pavan_veesam/poc-main/tribe-v2/myenv/bin/python"
SCRIPTS_DIR = Path("/home/pavan_veesam/telugish_whisper_edge/scripts")
BASE_MODEL_DIR = "/home/pavan_veesam/telugish_whisper_edge/models/whisper-tiny-base"


async def create_dummy_dataset() -> tuple[Path, Path]:
    if SMOKE_DIR.exists():
        shutil.rmtree(SMOKE_DIR)
    (SMOKE_DIR / "wavs").mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "id": "smoke_001",
            "text": "Anna production server lo memory leak undi",
            "voice": "te-IN-MohanNeural",
            "category": "tech_support_telugish",
            "noise_type": "clean",
            "speaker": "te-IN-MohanNeural",
        },
        {
            "id": "smoke_002",
            "text": "Today evening meeting lo deployment plan discuss cheddam",
            "voice": "en-IN-PrabhatNeural",
            "category": "office_code_switch",
            "noise_type": "traffic_15db",
            "speaker": "en-IN-PrabhatNeural",
        },
    ]

    records = []
    for s in samples:
        mp3_path = SMOKE_DIR / "wavs" / f"{s['id']}.mp3"
        wav_path = SMOKE_DIR / "wavs" / f"{s['id']}.wav"
        comm = edge_tts.Communicate(s["text"], s["voice"])
        await comm.save(str(mp3_path))
        y, _ = librosa.load(str(mp3_path), sr=16000, mono=True)
        sf.write(str(wav_path), y, 16000)
        records.append(
            {
                "id": s["id"],
                "audio_path": str(wav_path),
                "text": s["text"],
                "duration": round(len(y) / 16000.0, 2),
                "category": s["category"],
                "noise_type": s["noise_type"],
                "speaker": s["speaker"],
            }
        )

    train_jsonl = SMOKE_DIR / "train_dummy.jsonl"
    eval_jsonl = SMOKE_DIR / "eval_dummy.jsonl"
    for p in (train_jsonl, eval_jsonl):
        with open(p, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    return train_jsonl, eval_jsonl


def run_cmd(cmd: list[str]) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}")
    res = subprocess.run(cmd, text=True)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main() -> None:
    train_jsonl, eval_jsonl = asyncio.run(create_dummy_dataset())
    ft_out = SMOKE_DIR / "models" / "ft_dummy"
    int8_out = SMOKE_DIR / "models" / "int8_dummy"
    bench_dir = SMOKE_DIR / "benchmarks"
    bench_dir.mkdir(parents=True, exist_ok=True)

    # 1. Smoke test evaluate_model.py on base model
    run_cmd(
        [
            PYTHON_BIN,
            str(SCRIPTS_DIR / "evaluate_model.py"),
            "--model-path",
            BASE_MODEL_DIR,
            "--eval-jsonl",
            str(eval_jsonl),
            "--output-json",
            str(bench_dir / "eval_base_smoke.json"),
        ]
    )

    # 2. Smoke test train_whisper.py for 2 epochs on dummy 2-sample JSONL
    run_cmd(
        [
            PYTHON_BIN,
            str(SCRIPTS_DIR / "train_whisper.py"),
            "--model-name-or-path",
            BASE_MODEL_DIR,
            "--train-jsonl",
            str(train_jsonl),
            "--eval-jsonl",
            str(eval_jsonl),
            "--output-dir",
            str(ft_out),
            "--history-json",
            str(bench_dir / "train_history_smoke.json"),
            "--epochs",
            "2",
            "--batch-size",
            "2",
            "--eval-batch-size",
            "2",
            "--num-workers",
            "0",
            "--log-every",
            "1",
        ]
    )

    # 3. Smoke test export_edge_int8.py on the fine-tuned dummy model
    run_cmd(
        [
            PYTHON_BIN,
            str(SCRIPTS_DIR / "export_edge_int8.py"),
            "--hf-model-dir",
            str(ft_out),
            "--int8-output-dir",
            str(int8_out),
            "--base-model-dir",
            BASE_MODEL_DIR,
            "--eval-jsonl",
            str(eval_jsonl),
            "--report-json",
            str(bench_dir / "edge_int8_smoke.json"),
        ]
    )

    with open(bench_dir / "edge_int8_smoke.json", "r", encoding="utf-8") as f:
        report = json.load(f)

    print("\n=== SMOKE TEST SUMMARY ===")
    print(json.dumps(report["comparison"], indent=2))

    # Clean up smoke test directory so we leave zero clutter
    shutil.rmtree(SMOKE_DIR, ignore_errors=True)
    # Remove any benchmark files written to main benchmarks dir during smoke test
    for extra in [
        Path("/home/pavan_veesam/telugish_whisper_edge/benchmarks/eval_whisper-tiny-base.json"),
        Path("/home/pavan_veesam/telugish_whisper_edge/benchmarks/eval_telugish-whisper-tiny-ft.json"),
        Path("/home/pavan_veesam/telugish_whisper_edge/benchmarks/eval_telugish-whisper-tiny-int8-cpu.json"),
        Path("/home/pavan_veesam/telugish_whisper_edge/benchmarks/eval_telugish-whisper-tiny-int8-cuda.json"),
    ]:
        if extra.exists():
            extra.unlink()
    print("Smoke test passed and cleaned up successfully!")


if __name__ == "__main__":
    main()
