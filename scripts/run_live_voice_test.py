#!/usr/bin/env python3
import subprocess
import sys
import time
from faster_whisper import WhisperModel
from test_telugish_mic import transcribe_smart_chunked

ogg_path = sys.argv[1]
wav_path = "/tmp/tony_test_live.wav"
subprocess.run(["ffmpeg", "-y", "-i", ogg_path, "-ar", "16000", "-ac", "1", wav_path], stderr=subprocess.DEVNULL)

# 1. Method 2: Hybrid Engine (233 MB Small & 37.1 MB Tiny)
m_hyb_small = WhisperModel("/home/pavan_veesam/telugish_whisper_edge/models/vasista-telugu-small-int8", device="cpu", compute_type="int8", cpu_threads=4)
tel_s, rom_s, dur, ms_s = transcribe_smart_chunked(m_hyb_small, wav_path)
print(f"Audio Duration: {dur:.2f}s\n")
print(f"=== 1. Method 2: Two-Step Hybrid SMALL (233 MB INT8) [{ms_s:.0f} ms] ===")
print(f"  🇮🇳 Telugu Script : {tel_s}")
print(f"  🔤 Roman Telugish: {rom_s}")

m_hyb_tiny = WhisperModel("/home/pavan_veesam/telugish_whisper_edge/models/openflow-hybrid-tiny-int8", device="cpu", compute_type="int8", cpu_threads=4)
tel_t, rom_t, _, ms_t = transcribe_smart_chunked(m_hyb_tiny, wav_path)
print(f"\n=== 2. Method 2: Two-Step Hybrid TINY (37.1 MB Mobile INT8) [{ms_t:.0f} ms] ===")
print(f"  🇮🇳 Telugu Script : {tel_t}")
print(f"  🔤 Roman Telugish: {rom_t}")

# 2. Method 1: Pure Direct-to-Roman Neural Models (236 MB Small & 38.6 MB Tiny)
m_e2e_small = WhisperModel("/home/pavan_veesam/telugish_whisper_edge/models/telugish-whisper-small-int8", device="cpu", compute_type="int8", cpu_threads=4)
t0 = time.perf_counter()
segs_es, _ = m_e2e_small.transcribe(wav_path, language="en", beam_size=3, without_timestamps=True, vad_filter=False, condition_on_previous_text=False)
txt_es = " ".join(s.text.strip() for s in segs_es).strip()
ms_es = (time.perf_counter() - t0) * 1000.0
print(f"\n=== 3. Method 1: Pure E2E Neural SMALL (236 MB INT8, Zero Rules) [{ms_es:.0f} ms] ===")
print(f"  🧠 Direct Roman  : {txt_es}")

m_e2e_tiny = WhisperModel("/home/pavan_veesam/telugish_whisper_edge/models/telugish-whisper-tiny-int8", device="cpu", compute_type="int8", cpu_threads=4)
t0 = time.perf_counter()
segs_et, _ = m_e2e_tiny.transcribe(wav_path, language="en", beam_size=3, without_timestamps=True, vad_filter=False, condition_on_previous_text=False)
txt_et = " ".join(s.text.strip() for s in segs_et).strip()
ms_et = (time.perf_counter() - t0) * 1000.0
print(f"\n=== 4. Method 1: Pure E2E Neural TINY (38.6 MB Mobile INT8, Zero Rules) [{ms_et:.0f} ms] ===")
print(f"  🧠 Direct Roman  : {txt_et}")
