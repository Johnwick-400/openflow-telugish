#!/usr/bin/env python3
"""
Standalone Test Script for Hybrid Telugish Speech-to-Text (Mac & Linux)
- Works out-of-the-box on your MacBook Pro OR on the Linux GPU server.
- On first run on Mac, it auto-builds and caches the 233 MB INT8 Hybrid Model
  (92% IIT-Madras 1000-hr Telugu + 8% OpenAI Whisper-Small).
- Outputs BOTH:
  1. Clean Romanized Telugish ("Hello mama idi chala late ayipoyindi...")
  2. Native Telugu Script ("హలో మామ ఇది చాలా లేట్ అయిపోయింది...")

Install (once on Mac):
  pip install faster-whisper ctranslate2 transformers torch sounddevice soundfile scipy

Run Microphone Push-to-Talk:
  python3 test_telugish_mic.py

Test an Audio File (.wav / .mp3 / .ogg):
  python3 test_telugish_mic.py path/to/audio.wav
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

# ---------------------------------------------------------------------------
# 1. Telugu Script -> Colloquial Romanized Telugish + English Loanword Mapper
# ---------------------------------------------------------------------------
INDEPENDENT_VOWELS = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ee", "ఉ": "u", "ఊ": "oo",
    "ఋ": "ru", "ౠ": "ru", "ఎ": "e", "ఏ": "e", "ఐ": "ai", "ఒ": "o", "ఓ": "o", "ఔ": "au",
}
CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "t", "ఠ": "th", "డ": "d", "ఢ": "dh", "ణ": "n",
    "త": "th", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "l",
    "వ": "v", "శ": "sh", "ష": "sh", "స": "s", "హ": "h",
}
MATRAS = {
    "ా": "a", "ి": "i", "ీ": "i", "ు": "u", "ూ": "u",
    "ృ": "ru", "ె": "e", "ే": "e", "ై": "ai", "ొ": "o", "ో": "o", "ౌ": "au",
}
VIRAMA, ANUSVARA, VISARGA = "్", "ం", "ః"
LABIALS = set("పఫబభమ")

COLLOQUIAL_FIXES = {
    "halo": "hello", "hai": "hi", "es": "yes", "yes": "yes", "oke": "okay",
    "plij": "please", "sari": "sorry", "thyanks": "thanks", "tude": "today",
    "tumaro": "tomorrow", "marning": "morning", "ivining": "evening", "nait": "night",
    "vet": "late", "veta": "late", "let": "late", "letu": "late", "veyit": "late",
    "taim": "time", "taimu": "time", "tu": "to", "and": "and", "bat": "but",
    "vark": "work", "varku": "work", "aaphis": "office", "aphis": "office",
    "aaphisu": "office", "aaphisulo": "office lo", "aphisulo": "office lo",
    "spich": "speech", "spiking": "speaking", "spik": "speak", "kal": "call",
    "phon": "phone", "phonu": "phone", "mobail": "mobile", "mesej": "message",
    "miting": "meeting", "mitingu": "meeting", "sarvar": "server", "sarvarlo": "server lo",
    "modal": "model", "trening": "training", "test": "test", "diplay": "deploy",
    "bild": "build", "kod": "code", "prajekt": "project", "sistam": "system",
    "saphtver": "software", "intarnet": "internet", "kampyutar": "computer",
    "deta": "data", "epiay": "API", "errar": "error", "ishyu": "issue",
    "prablam": "problem", "stetas": "status", "apdet": "update", "manejar": "manager",
    "tim": "team", "klayint": "client", "banku": "bank", "byanku": "bank",
    "pement": "payment", "mani": "money", "bil": "bill", "jipe": "GPay",
    "phonpe": "PhonePe", "yupiai": "UPI", "sviggi": "Swiggy", "ardar": "order",
    "haspital": "hospital", "daktar": "doctor", "medisin": "medicine", "kaleji": "college",
    "tiket": "ticket", "vidiyo": "video", "chanal": "channel", "yutyub": "YouTube",
    "arjent": "urgent", "varjan": "urgent", "dipo": "deploy", "deppo": "deploy", "chek": "check", "sarch": "search",
    "jisipi": "GCP", "vaks": "logs", "vaking": "logging", "roja": "roju aa",
    "gyatharing": "gathering", "mit": "meet", "phichar": "feature", "devalap": "develop",
    "yaksept": "accept", "notis": "notice", "prayathnam": "prayathnam",
}


def telugu_word_to_roman(word: str) -> str:
    out, n, i = [], len(word), 0
    while i < n:
        ch = word[i]
        if ch in INDEPENDENT_VOWELS:
            out.append(INDEPENDENT_VOWELS[ch])
            i += 1
        elif ch in CONSONANTS:
            base = CONSONANTS[ch]
            nxt = word[i + 1] if i + 1 < n else ""
            if nxt == VIRAMA:
                out.append(base)
                i += 2
            elif nxt in MATRAS:
                out.append(base + MATRAS[nxt])
                i += 2
            else:
                out.append(base + "a")
                i += 1
        elif ch in MATRAS:
            out.append(MATRAS[ch])
            i += 1
        elif ch == VIRAMA:
            i += 1
        elif ch == ANUSVARA:
            nxt = word[i + 1] if i + 1 < n else ""
            out.append("m" if (not nxt or not ("\u0c00" <= nxt <= "\u0c7f") or nxt in LABIALS) else "n")
            i += 1
        elif ch == VISARGA:
            out.append("h")
            i += 1
        else:
            out.append(ch)
            i += 1
    return "".join(out)


def to_telugish(telugu_text: str) -> str:
    rom = " ".join(telugu_word_to_roman(t) for t in telugu_text.split())
    rom = re.sub(r"\b(aaphisu|aphisu|aaphis|aphis)\s+lo\b", "office lo", rom, flags=re.IGNORECASE)
    rom = re.sub(r"\b(kraud|kvaud|kvaunt)\s+van\b", "Cloud Run", rom, flags=re.IGNORECASE)
    rom = re.sub(r"\bji\s+si\s+pi\b", "GCP", rom, flags=re.IGNORECASE)
    out = []
    for w in rom.split():
        clean = re.sub(r"[^a-zA-Z0-9]", "", w).lower()
        out.append(COLLOQUIAL_FIXES.get(clean, w))
    res = " ".join(out).strip()
    return res[:1].upper() + res[1:] if res else res


def transcribe_smart_chunked(model: WhisperModel, audio_file: str) -> tuple[str, str, float, float]:
    """Automatically splits audio > 8.0s at quiet pause valleys so Telugu BPE tokens never truncate."""
    t_start = time.perf_counter()
    d, sr = sf.read(audio_file)
    if d.ndim > 1:
        d = np.mean(d, axis=1)
    dur = len(d) / float(sr)
    if dur <= 8.5:
        segs, _ = model.transcribe(audio_file, language="te", beam_size=2, without_timestamps=True, vad_filter=False, condition_on_previous_text=False)
        tel_txt = " ".join(s.text.strip() for s in segs).strip()
    else:
        # Split into ~7.5s chunks at minimum-energy pause points
        chunk_samples = int(7.5 * sr)
        tel_parts = []
        pos = 0
        while pos < len(d):
            end = min(pos + chunk_samples, len(d))
            if end < len(d):
                # Search +/- 0.8s around `end` for lowest RMS 100ms window
                search_start = max(pos + int(4.0 * sr), end - int(0.8 * sr))
                search_end = min(len(d), end + int(0.8 * sr))
                win = int(0.1 * sr)
                best_split = end
                min_e = 1e9
                for s_idx in range(search_start, search_end - win, win // 2):
                    e = float(np.mean(d[s_idx:s_idx + win] ** 2))
                    if e < min_e:
                        min_e = e
                        best_split = s_idx + win // 2
                end = best_split
            chunk = d[pos:end]
            if len(chunk) > int(0.5 * sr):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    sf.write(tf.name, chunk, sr, subtype="PCM_16")
                    segs, _ = model.transcribe(tf.name, language="te", beam_size=2, without_timestamps=True, vad_filter=False, condition_on_previous_text=False)
                    part = " ".join(s.text.strip() for s in segs).strip()
                    if part:
                        tel_parts.append(part)
                    os.remove(tf.name)
            pos = end
        tel_txt = " ".join(tel_parts).strip()

    rom_txt = to_telugish(tel_txt)
    ms = (time.perf_counter() - t_start) * 1000.0
    return tel_txt, rom_txt, dur, ms


# ---------------------------------------------------------------------------
# 2. Auto-Build / Load the 233 MB INT8 Hybrid Model
# ---------------------------------------------------------------------------
def get_or_build_model_dir() -> str:
    # Check server path first, then local ./models/vasista-telugu-small-int8
    server_path = Path("/home/pavan_veesam/telugish_whisper_edge/models/vasista-telugu-small-int8")
    if server_path.exists():
        return str(server_path)

    local_int8 = Path.cwd() / "models" / "vasista-telugu-small-int8"
    if (local_int8 / "model.bin").exists():
        return str(local_int8)

    print("📦 First-time setup: Building 233 MB INT8 Hybrid Model (92% Vasista Telugu + 8% OpenAI Whisper)...")
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from ctranslate2.converters import TransformersConverter

    tmp_hf = Path.cwd() / "models" / "_tmp_hybrid_hf"
    tmp_hf.mkdir(parents=True, exist_ok=True)
    local_int8.parent.mkdir(parents=True, exist_ok=True)

    p_base = WhisperProcessor.from_pretrained("openai/whisper-small")
    m_base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small", torch_dtype=torch.float32)
    m_vas = WhisperForConditionalGeneration.from_pretrained("vasista22/whisper-telugu-small", torch_dtype=torch.float32)

    sd_vas, sd_base = m_vas.state_dict(), m_base.state_dict()
    blended = {k: 0.92 * sd_vas[k] + 0.08 * sd_base[k] for k in sd_vas}
    m_base.load_state_dict(blended)
    m_base.half().save_pretrained(str(tmp_hf))
    p_base.save_pretrained(str(tmp_hf))

    conv = TransformersConverter(str(tmp_hf), copy_files=["tokenizer.json", "preprocessor_config.json"])
    conv.convert(str(local_int8), quantization="int8", force=True)
    shutil.rmtree(str(tmp_hf), ignore_errors=True)
    print(f"✅ Saved 233 MB INT8 Model to {local_int8}")
    return str(local_int8)


def record_from_mic(out_wav: str) -> None:
    try:
        import sounddevice as sd
        sr = 16000
        print("🎙️  SPEAK NOW (Telugu + English)... Press ENTER when done!")
        frames = []
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=lambda indata, f, t, s: frames.append(indata.copy())):
            input()
        sf.write(out_wav, np.concatenate(frames, axis=0), sr, subtype="PCM_16")
        return
    except Exception:
        pass

    # Fallback to ffmpeg on macOS / Linux
    fmt = ["-f", "avfoundation", "-i", ":0"] if platform.system() == "Darwin" else ["-f", "alsa", "-i", "default"]
    print("🎙️  SPEAK NOW (Telugu + English)... Press ENTER when done!")
    proc = subprocess.Popen(["ffmpeg", "-y", *fmt, "-ar", "16000", "-ac", "1", out_wav], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    input()
    proc.terminate()
    proc.wait()


def main():
    model_dir = get_or_build_model_dir()
    t0 = time.perf_counter()
    model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4)
    print(f"⚡ Loaded Hybrid INT8 Engine ({model_dir}) in {(time.perf_counter()-t0)*1000:.0f} ms")

    # Mode 1: Transcribe audio file(s) passed on command line
    if len(sys.argv) > 1:
        for audio_file in sys.argv[1:]:
            tel_txt, rom_txt, dur, ms = transcribe_smart_chunked(model, audio_file)
            print(f"\n📂 File          : {audio_file} ({dur:.2f}s) | Latency: {ms:.0f} ms")
            print(f"🇮🇳 Telugu Script : {tel_txt}")
            print(f"🔤 Roman Telugish: {rom_txt}")
        return

    # Mode 2: Interactive Microphone Push-to-Talk Loop
    print("\n========================================================")
    print("🎙️  INTERACTIVE PUSH-TO-TALK MODE (Ctrl+C to exit)")
    print("========================================================")
    while True:
        try:
            input("\n👉 Press ENTER to start recording...")
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                wav_path = tmp.name
            record_from_mic(wav_path)
            tel_txt, rom_txt, dur, ms = transcribe_smart_chunked(model, wav_path)
            os.remove(wav_path)

            print(f"\n⚡ [{ms:.0f} ms | {dur:.1f}s audio]")
            print(f"🇮🇳 Telugu Script : {tel_txt}")
            print(f"🔤 Roman Telugish: {rom_txt}")

            if platform.system() == "Darwin" and rom_txt:
                p = subprocess.Popen(["pbcopy"], stdin=subprocess.PIPE)
                p.communicate(input=rom_txt.encode("utf-8"))
                print("📋 Copied Roman Telugish to Mac Clipboard (Cmd+V to paste)!")
        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()
