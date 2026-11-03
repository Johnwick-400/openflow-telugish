#!/usr/bin/env python3
"""
Standalone Offline Edge Inference Script for Telugish-Whisper-Tiny-INT8 (38.6 MB)
Runs 100% offline on CPU (17.3x faster than real-time).

Usage:
  python infer_edge.py /path/to/audio.wav
"""

import sys
import time
from pathlib import Path
from faster_whisper import WhisperModel

MODEL_DIR = Path(__file__).resolve().parent / "models" / "telugish-whisper-tiny-int8"


def transcribe_file(audio_path: str, model_dir: str = str(MODEL_DIR)) -> None:
    t0 = time.perf_counter()
    model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4)
    load_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    segments, info = model.transcribe(audio_path, language="en", beam_size=1, without_timestamps=True)
    text = " ".join(s.text.strip() for s in segments).strip()
    infer_ms = (time.perf_counter() - t1) * 1000.0

    print(f"Audio File   : {audio_path} ({info.duration:.2f}s)")
    print(f"Model Load   : {load_ms:.1f} ms | Inference: {infer_ms:.1f} ms (RTF: {infer_ms/1000.0/max(info.duration,1e-4):.4f})")
    print(f"Transcript   : {text}")


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parent / "samples" / "sample_1_food_delivery.wav")
    transcribe_file(target)
