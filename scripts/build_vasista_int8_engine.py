#!/usr/bin/env python3
"""
Build the Final Hybrid OpenFlow-Telugish Engine (`models/vasista-telugu-small-int8`):
1. WiSE-FT Blend: 92% `vasista22/whisper-telugu-small` (1,000 hrs real human Telugu) +
   8% `openai/whisper-small` (680k hrs English acoustic prior) so English phrases like
   "Yes today to work and speech" are heard clearly alongside fast native Telugu.
2. Comprehensive Phonetic English Loanword Restorer + Colloquial Telugu Romanizer.
3. Quantized to 236.7 MB INT8 CTranslate2 for fast offline CPU inference.
"""

import re
import shutil
import time
from pathlib import Path
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from faster_whisper import WhisperModel
from export_edge_int8 import convert_hf_to_ct2_int8
from build_real_human_v2 import telugu_sentence_to_roman

BASE_DIR = Path("/home/pavan_veesam/telugish_whisper_edge")
VAS_HF_DIR = BASE_DIR / "models" / "vasista-telugu-small-hf"
OPENAI_BASE_DIR = BASE_DIR / "models" / "whisper-small-base"
HYBRID_HF_DIR = BASE_DIR / "models" / "openflow-hybrid-small-hf"
CT2_INT8_DIR = BASE_DIR / "models" / "vasista-telugu-small-int8"

# Comprehensive phonetic Telugu-script -> English loanword dictionary
COLLOQUIAL_FIXES = {
    # Greetings & conversational English
    "halo": "hello",
    "hai": "hi",
    "es": "yes",
    "yes": "yes",
    "no": "no",
    "oke": "okay",
    "ok": "okay",
    "plij": "please",
    "sari": "sorry",
    "thyanks": "thanks",
    "thyankyu": "thank you",
    "tude": "today",
    "tumaro": "tomorrow",
    "estarde": "yesterday",
    "marning": "morning",
    "ivining": "evening",
    "nait": "night",
    "vet": "late",
    "veta": "late",
    "let": "late",
    "letu": "late",
    "veyit": "late",
    "taim": "time",
    "taimu": "time",
    "tu": "to",
    "and": "and",
    "bat": "but",
    "bikas": "because",
    "so": "so",
    "laik": "like",
    "aki": "okay",
    # Work, Tech & Engineering
    "vark": "work",
    "varku": "work",
    "aaphis": "office",
    "aphis": "office",
    "aaphisu": "office",
    "aaphisulo": "office lo",
    "aphisulo": "office lo",
    "spich": "speech",
    "spiking": "speaking",
    "spik": "speak",
    "kal": "call",
    "kals": "calls",
    "phon": "phone",
    "phonu": "phone",
    "mobail": "mobile",
    "mesej": "message",
    "mel": "mail",
    "imel": "email",
    "miting": "meeting",
    "mitingu": "meeting",
    "sarvar": "server",
    "sarvarlo": "server lo",
    "klavud": "cloud",
    "โมడల్": "model",
    "modal": "model",
    "madal": "model",
    "trening": "training",
    "tren": "train",
    "test": "test",
    "testing": "testing",
    "디ప్లాయ్": "deploy",
    "diplay": "deploy",
    "bild": "build",
    "kod": "code",
    "koding": "coding",
    "prajekt": "project",
    "prajektu": "project",
    "sistam": "system",
    "saphtver": "software",
    "intarnet": "internet",
    "kampyutar": "computer",
    "lyaptap": "laptop",
    "ডেటా": "data",
    "deta": "data",
    "detabes": "database",
    "epiay": "API",
    "errar": "error",
    "bag": "bug",
    "ishyu": "issue",
    "prablam": "problem",
    "status": "status",
    "stetas": "status",
    "apdet": "update",
    "manejar": "manager",
    "tim": "team",
    "klayint": "client",
    # Finance, Food, Medical, Daily
    "banku": "bank",
    "byanku": "bank",
    "akavunt": "account",
    "pement": "payment",
    "mani": "money",
    "bil": "bill",
    "billu": "bill",
    "jipe": "GPay",
    "phonpe": "PhonePe",
    "yupiai": "UPI",
    "sviggi": "Swiggy",
    "jomato": "Zomato",
    "ardar": "order",
    "delivari": "delivery",
    "haspital": "hospital",
    "daktar": "doctor",
    "medisin": "medicine",
    "riport": "report",
    "kaleji": "college",
    "skul": "school",
    "tiket": "ticket",
    "vidiyo": "video",
    "chanal": "channel",
    "yutyub": "YouTube",
}


def telugish_postprocess(telugu_text: str) -> str:
    rom = telugu_sentence_to_roman(telugu_text)
    # Merge split postpositions like "aaphisu lo" -> "office lo"
    rom = re.sub(r"\b(aaphisu|aphisu|aaphis|aphis)\s+lo\b", "office lo", rom, flags=re.IGNORECASE)
    words = rom.split()
    out = []
    for w in words:
        clean = re.sub(r"[^a-zA-Z0-9]", "", w).lower()
        if clean in COLLOQUIAL_FIXES:
            out.append(COLLOQUIAL_FIXES[clean])
        else:
            out.append(w)
    res = " ".join(out)
    return res[:1].upper() + res[1:] if res else res


def build_hybrid_int8():
    print("Building WiSE-FT Hybrid Model (92% Vasista 1000-hr Telugu + 8% OpenAI Whisper-Small)...")
    p = WhisperProcessor.from_pretrained(str(VAS_HF_DIR))
    sd_vas = {k: v.cpu() for k, v in WhisperForConditionalGeneration.from_pretrained(str(VAS_HF_DIR), torch_dtype=torch.float32).state_dict().items()}
    sd_base = {k: v.cpu() for k, v in WhisperForConditionalGeneration.from_pretrained(str(OPENAI_BASE_DIR), torch_dtype=torch.float32).state_dict().items()}

    alpha_vas = 0.92
    blended = {k: alpha_vas * sd_vas[k] + (1.0 - alpha_vas) * sd_base[k] for k in sd_vas}

    m = WhisperForConditionalGeneration.from_pretrained(str(VAS_HF_DIR), torch_dtype=torch.float32)
    m.load_state_dict(blended)
    HYBRID_HF_DIR.mkdir(parents=True, exist_ok=True)
    m.half().save_pretrained(str(HYBRID_HF_DIR))
    p.save_pretrained(str(HYBRID_HF_DIR))

    base_tok = OPENAI_BASE_DIR / "tokenizer.json"
    if base_tok.exists():
        shutil.copy(str(base_tok), str(HYBRID_HF_DIR / "tokenizer.json"))

    if CT2_INT8_DIR.exists():
        shutil.rmtree(str(CT2_INT8_DIR))
    convert_hf_to_ct2_int8(HYBRID_HF_DIR, CT2_INT8_DIR, quantization="int8")

    print("\n=== Verifying Hybrid INT8 Engine on Both of Tony's Voice Notes ===")
    model = WhisperModel(str(CT2_INT8_DIR), device="cpu", compute_type="int8", cpu_threads=4)
    for idx, wav in [(1, "/tmp/tony_voice_1.wav"), (2, "/tmp/tony_voice_2.wav")]:
        t0 = time.perf_counter()
        segs, info = model.transcribe(wav, language="te", beam_size=2, without_timestamps=True, vad_filter=True)
        raw = " ".join(s.text.strip() for s in segs).strip()
        rom = telugish_postprocess(raw)
        ms = (time.perf_counter() - t0) * 1000.0
        print(f"\n[Voice Note {idx} | {info.duration:.2f}s audio | {ms:.0f} ms CPU]")
        print(f"  🇮🇳 Telugu Script : {raw}")
        print(f"  🔤 Roman Telugish: {rom}")


if __name__ == "__main__":
    build_hybrid_int8()
