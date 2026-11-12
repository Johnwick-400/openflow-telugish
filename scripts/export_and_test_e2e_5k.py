#!/usr/bin/env python3
"""
Quantize our 9.78-Hour Trained End-to-End Direct-to-Roman Neural Models (`tiny` 38.6MB & `small` 233MB)
to INT8 CTranslate2 and test their 100% RAW NEURAL OUTPUTS (zero Python transliteration rules!)
on Tony's real voice notes (/tmp/tony_voice_1.wav, /tmp/tony_voice_2.wav, /tmp/tony_voice_4.wav).
"""

import shutil
import time
from pathlib import Path
from faster_whisper import WhisperModel
from export_edge_int8 import convert_hf_to_ct2_int8

BASE_DIR = Path("/home/pavan_veesam/telugish_whisper_edge")

TINY_FT = BASE_DIR / "models" / "telugish-whisper-tiny-ft"
TINY_INT8 = BASE_DIR / "models" / "telugish-whisper-tiny-int8"

SMALL_FT = BASE_DIR / "models" / "telugish-whisper-small-ft"
SMALL_INT8 = BASE_DIR / "models" / "telugish-whisper-small-int8"


def main():
    # Copy tokenizer.json from base models so faster-whisper loads cleanly
    shutil.copy(str(BASE_DIR / "models" / "whisper-tiny-base" / "tokenizer.json"), str(TINY_FT / "tokenizer.json"))
    shutil.copy(str(BASE_DIR / "models" / "whisper-small-base" / "tokenizer.json"), str(SMALL_FT / "tokenizer.json"))

    convert_hf_to_ct2_int8(TINY_FT, TINY_INT8, quantization="int8")
    convert_hf_to_ct2_int8(SMALL_FT, SMALL_INT8, quantization="int8")

    for label, mdir in [
        ("📱 E2E Direct-to-Roman TINY INT8 (38.6 MB Mobile)", TINY_INT8),
        ("💻 E2E Direct-to-Roman SMALL INT8 (233.3 MB Pro)", SMALL_INT8),
    ]:
        print(f"\n======================================================================")
        print(f"{label} — 100% RAW NEURAL OUTPUT (Zero Python Rules)")
        print(f"======================================================================")
        model = WhisperModel(str(mdir), device="cpu", compute_type="int8", cpu_threads=4)
        for idx in [1, 2, 4]:
            wav = f"/tmp/tony_voice_{idx}.wav"
            t0 = time.perf_counter()
            segs, info = model.transcribe(wav, language="en", beam_size=3, without_timestamps=True, vad_filter=False, condition_on_previous_text=False)
            raw_neural = " ".join(s.text.strip() for s in segs).strip()
            ms = (time.perf_counter() - t0) * 1000.0
            print(f"  [Voice #{idx} ({info.duration:.1f}s) | Latency: {ms:.0f} ms ({info.duration/(ms/1000):.1f}x RT)]")
            print(f"    🧠 Raw Neural Output: \"{raw_neural}\"")


if __name__ == "__main__":
    main()
