#!/usr/bin/env python3
"""
Build & Benchmark Mobile-Sized Hybrid WiSE-FT INT8 Models for Android / iOS / Edge:
1. `openflow-hybrid-tiny-int8` (38.6 MB) -> 92% vasista22/whisper-telugu-tiny + 8% telugish-whisper-tiny-ft
2. `openflow-hybrid-base-int8` (71.8 MB) -> 92% vasista22/whisper-telugu-base + 8% openai/whisper-base
Test both on Tony's real voice notes (/tmp/tony_voice_1.wav, /tmp/tony_voice_2.wav, /tmp/tony_voice_4.wav)!
"""

import shutil
import time
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from faster_whisper import WhisperModel
from export_edge_int8 import convert_hf_to_ct2_int8
from test_telugish_mic import transcribe_smart_chunked

BASE_DIR = Path("/home/pavan_veesam/telugish_whisper_edge")


def build_mobile_tier(tier: str, vas_repo: str, en_repo: str, out_name: str):
    out_int8 = BASE_DIR / "models" / out_name
    tmp_hf = BASE_DIR / "models" / f"_tmp_{out_name}_hf"
    tmp_hf.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Building {out_name} ({tier.upper()}) ===")
    p_en = WhisperProcessor.from_pretrained(en_repo)
    m_en = WhisperForConditionalGeneration.from_pretrained(en_repo, torch_dtype=torch.float32)
    m_vas = WhisperForConditionalGeneration.from_pretrained(vas_repo, torch_dtype=torch.float32)

    sd_vas, sd_en = m_vas.state_dict(), m_en.state_dict()
    blended = {k: 0.92 * sd_vas[k] + 0.08 * sd_en[k] for k in sd_vas}
    m_en.load_state_dict(blended)
    m_en.half().save_pretrained(str(tmp_hf))
    p_en.save_pretrained(str(tmp_hf))

    if out_int8.exists():
        shutil.rmtree(str(out_int8))
    convert_hf_to_ct2_int8(tmp_hf, out_int8, quantization="int8")
    shutil.rmtree(str(tmp_hf), ignore_errors=True)

    size_mb = (out_int8 / "model.bin").stat().st_size / (1024 * 1024)
    print(f"✅ Exported {out_name}: {size_mb:.2f} MB (INT8 mobile-ready)")

    model = WhisperModel(str(out_int8), device="cpu", compute_type="int8", cpu_threads=4)
    for idx in [1, 2, 4]:
        wav = f"/tmp/tony_voice_{idx}.wav"
        tel, rom, dur, ms = transcribe_smart_chunked(model, wav)
        print(f"  [Voice #{idx} ({dur:.1f}s) | Latency: {ms:.0f} ms ({dur/(ms/1000):.1f}x RT)]")
        print(f"    🇮🇳 Telugu : {tel}")
        print(f"    🔤 Roman  : {rom}")


def main():
    # Free temporary intermediate FP32 checkpoints & raw .tar.gz archives first
    for p in [
        BASE_DIR / "data" / "fleurs_te" / "train.tar.gz",
        BASE_DIR / "data" / "fleurs_te" / "dev.tar.gz",
        BASE_DIR / "data" / "fleurs_te" / "test.tar.gz",
        BASE_DIR / "data" / "openslr_te" / "te_in_male.zip",
    ]:
        if p.exists():
            p.unlink()
    for d in [
        BASE_DIR / "models" / "telugish-whisper-small-yt-v3",
        BASE_DIR / "models" / "vasista-telugish-small-ft",
        BASE_DIR / "models" / "telugish-whisper-small-final",
        BASE_DIR / "models" / "openflow-hybrid-small-hf",
    ]:
        if d.exists():
            shutil.rmtree(str(d), ignore_errors=True)

    # Tier 2: Base (71.8 MB) - blending vasista22/whisper-telugu-base + openai/whisper-base
    build_mobile_tier(
        tier="base",
        vas_repo="vasista22/whisper-telugu-base",
        en_repo="openai/whisper-base",
        out_name="openflow-hybrid-base-int8",
    )


if __name__ == "__main__":
    main()
