#!/usr/bin/env python3
"""
Extract 450 real human Indian-English recordings from AI4Bharat Svarah
to combine with the 3,085 real human native Telugu recordings from Google FLEURS (te_in).
"""
import io
import json
from pathlib import Path
import requests
import pyarrow.parquet as pq
import soundfile as sf
import numpy as np

BASE_DIR = Path("/home/pavan_veesam/telugish_whisper_edge")
SVARAH_DIR = BASE_DIR / "data" / "real_svarah_v2"
SVARAH_DIR.mkdir(parents=True, exist_ok=True)

r = requests.get("https://huggingface.co/api/datasets/ai4bharat/Svarah/tree/main/data", timeout=15)
files = [x["path"] for x in r.json() if x["path"].endswith(".parquet")]
print("Found Svarah parquet shards:", files)

records = []
idx = 0
for fpath in files[:2]:
    url = f"https://huggingface.co/datasets/ai4bharat/Svarah/resolve/main/{fpath}"
    print("Downloading", url)
    resp = requests.get(url, timeout=120)
    table = pq.read_table(io.BytesIO(resp.content))
    df = table.to_pydict()
    n = len(df["text"])
    for i in range(n):
        text = str(df["text"][i]).strip()
        words = text.split()
        if len(words) < 5 or len(words) > 28:
            continue
        aud_bytes = df["audio"][i]["bytes"]
        data, sr = sf.read(io.BytesIO(aud_bytes))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        dur = len(data) / float(sr)
        if dur < 2.2 or dur > 11.0:
            continue
        out_wav = SVARAH_DIR / f"svarah_v2_{idx:04d}.wav"
        sf.write(str(out_wav), data, sr, subtype="PCM_16")
        records.append({
            "id": f"svarah_v2_{idx:04d}",
            "audio_path": str(out_wav),
            "text": text,
            "duration": round(dur, 2),
            "category": "indian_english_real_human",
            "source": "ai4bharat_svarah_real",
            "noise": "real_room_mic",
            "speaker": "real_human_en_in",
        })
        idx += 1
        if idx >= 450:
            break
    if idx >= 450:
        break

with open(SVARAH_DIR / "manifest.json", "w") as f:
    json.dump(records, f, indent=2)
print(f"Extracted {len(records)} real human Svarah clips ({sum(r['duration'] for r in records)/60:.1f} mins)")
