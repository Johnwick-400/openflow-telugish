import os
import sys
import io
import json
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio

os.makedirs("/home/pavan_veesam/telugish_whisper_edge/data/real_svarah_raw", exist_ok=True)

ds = load_dataset("amrithanandini/svarah", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))

domain_keywords = [
    "upi", "bank", "account", "payment", "bill", "balance", "money", "credit", "card",
    "loan", "transaction", "autopay", "hdfc", "sbi", "icici", "axis", "wallet", "recharge",
    "delivery", "order", "food", "restaurant", "items", "products", "booking", "hotel",
    "ticket", "app", "website", "online", "computer", "software", "network", "mobile",
    "office", "service", "complaint", "customer", "doctor", "hospital", "health", "medicine",
    "patient", "data", "system", "project", "team", "meeting", "work", "call", "email",
    "company", "business", "market", "price", "store", "shop", "medical", "treatment"
]

selected = []
seen_texts = set()

for i, row in enumerate(ds):
    if len(selected) >= 85 or i >= 1200:
        break
    if not row.get("audio") or not row["audio"].get("bytes"):
        continue
    ref = (row.get("sentence") or "").strip()
    dur = float(row.get("duration") or 0.0)

    words = ref.split()
    if not (2.5 <= dur <= 10.5 and 5 <= len(words) <= 22):
        continue

    if any(c.isdigit() for c in ref):
        continue

    lower_ref = ref.lower()
    if lower_ref in seen_texts:
        continue

    is_domain = any(kw in lower_ref.split() for kw in domain_keywords)
    if not is_domain:
        continue

    q_words = ("how", "what", "whats", "will", "can", "do", "does", "where", "when", "why", "is", "are", "could", "would", "should")
    first_word = words[0].lower()
    formatted_text = ref[0].upper() + ref[1:]
    for acr in ["upi", "sbi", "hdfc", "icici", "dth", "emi", "otp", "kyc", "gst", "pan", "atm", "it", "hr", "api", "sql"]:
        formatted_text = " ".join([acr.upper() if w.lower() == acr else w for w in formatted_text.split()])
    if not formatted_text.endswith((".", "?", "!")):
        formatted_text += "?" if first_word in q_words else "."

    audio_bytes = row["audio"]["bytes"]
    data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    idx_id = len(selected)
    wav_path = f"/home/pavan_veesam/telugish_whisper_edge/data/real_svarah_raw/svarah_{idx_id:03d}.wav"
    sf.write(wav_path, data, 16000, subtype="PCM_16")

    seen_texts.add(lower_ref)
    selected.append({
        "svarah_idx": i,
        "id": f"svarah_{idx_id:03d}",
        "audio_filepath": wav_path,
        "text": formatted_text,
        "raw_ref": ref,
        "duration": round(len(data) / 16000.0, 3),
        "source": "ai4bharat/Svarah"
    })
    print(f"[{len(selected)}] ({dur:.2f}s) {formatted_text}", flush=True)

with open("/home/pavan_veesam/telugish_whisper_edge/data/real_svarah_raw/metadata.json", "w") as f:
    json.dump(selected, f, indent=2)

print(f"Saved {len(selected)} real Svarah samples.", flush=True)
sys.stdout.flush()
os._exit(0)
