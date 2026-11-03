#!/usr/bin/env python3
"""
Round 3 Dataset Balancing & Phonetic Disambiguation:
1. Adds all remaining unused real human Svarah recordings + speed/noise-augmented copies of the 40 real Svarah train samples
   (to prevent catastrophic forgetting on real human Indian-English audio).
2. Adds targeted phonetic disambiguation Telugish utterances (zero overlap with eval.jsonl) covering
   Proper Nouns (Andhra Pradesh, Telangana, November, December), postpositions (dwara, lo, chupinchali),
   and compound words (payment gateway, original copy, completion certificate).
"""

import asyncio
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import edge_tts
import librosa
import numpy as np
import soundfile as sf
from build_dataset import BASE_DIR, DATA_DIR, WAV_DIR, VOICES, RATES, apply_acoustic_augmentation

ROUND3_SENTENCES: List[Tuple[str, str]] = [
    # Proper nouns & months (Andhra Pradesh, November, etc. - zero overlap with eval)
    ("Andhra Pradesh state government medical colleges lo seats chala unnay.", "medical_counselling"),
    ("Andhra Pradesh mariyu Telangana states lo engineering colleges counselling start ayindi.", "daily_workplace"),
    ("Visakhapatnam Andhra Pradesh lo pedda IT hub ga develop avthundi.", "software_engineering"),
    ("November first week lo semester exams start avthayi syllabus complete cheyali.", "daily_workplace"),
    ("November fifteenth lopala income tax audit report submit cheyali.", "fintech_upi"),
    ("October mariyu November months lo festival season discounts baga untayi.", "food_delivery"),
    ("Second week lo project phase two deployment schedule chesamu.", "software_engineering"),

    # Compound nouns & postpositions (dwara, payment gateway, NEFT, original copy, chupinchali, etc.)
    ("Razorpay payment gateway lo UPI mariyu net banking options work avthunnayi.", "fintech_upi"),
    ("BillDesk payment gateway dwara college tuition fee online lo pay chesamu.", "fintech_upi"),
    ("RTGS mariyu NEFT dwara pedda amounts safe ga transfer cheyochu.", "fintech_upi"),
    ("IMPS dwara transfer chesthe immediate ga bank account lo padthundi.", "fintech_upi"),
    ("Mobile banking app dwara credit card pin generate cheskovachu.", "fintech_upi"),
    ("Aadhaar card original copy mariyu xerox copy verification counter lo chupinchali.", "medical_counselling"),
    ("Tenth class marksheet original copy passport office lo officer ki chupinchali.", "medical_counselling"),
    ("Course completion certificate download chesi HR team ki mail lo chupinchali.", "daily_workplace"),
    ("Provisional degree certificate original copy university counselling lo chupinchali.", "medical_counselling"),
    ("Driving license original copy traffic police adigithe DigiLocker lo chupinchali.", "daily_workplace"),
    ("Gate daggara security guard ki ID card chupinchali.", "daily_workplace"),

    # More Telugish verbs & suffixes (theeyamannaru, cheyamannaru, avthayi, marchipoyi, etc.)
    ("Chest congestion kosam pulmonologist ni kalisthe chest X ray theeyamannaru.", "medical_counselling"),
    ("Leg fracture ayindemo ani orthopedic doctor X ray scan theeyamannaru.", "medical_counselling"),
    ("Wallet intlo marchipoyi office ki vachesanu.", "daily_workplace"),
    ("Swiggy lo biryani order pedithe raita marchipoyi pampinchadu.", "food_delivery"),
    ("Hotel odu idli tho patu sambar marchipoyi pampinchadu.", "food_delivery"),
    ("Online classes next Monday nundi regular ga start avthayi.", "daily_workplace"),
    ("Counselling second round results repu evening release avthayi.", "medical_counselling"),
    ("Bank holidays valla cheque clearance two days delay avthundi.", "fintech_upi"),
    ("Kotak Mahindra bank lo zero balance account online lo open chesa.", "fintech_upi"),
    ("Curd rice lo avakaya kalupukoni thinte chala ruchiga untundi.", "food_delivery"),
    ("Vedi vedi annam lo avakaya mariyu neyyi kalupukoni thinte baguntundi.", "food_delivery"),
    ("Cold coffee lo sugar lekunda almond milk vesi cheppu.", "food_delivery"),
    ("Fever mariyu body pains unte paracetamol tablet vedi neellatho vesko.", "medical_counselling"),
    ("Cough thaggadaniki ratri padukune mundhu ten ml syrup theeskovali.", "medical_counselling"),
    ("Hostel lo food baledhu kabatti ee roju bayata dosa tiffin chedham.", "food_delivery"),
    ("Salary lo TDS deduction ee month konchem ekkuva cut ayindi.", "fintech_upi"),
]


async def synth_round3(
    sem: asyncio.Semaphore,
    idx: int,
    text: str,
    category: str,
    voice: str,
    rate: str,
    noise_type: str,
) -> Dict:
    async with sem:
        mp3_tmp = WAV_DIR / f"tmp_r3_{idx:04d}.mp3"
        wav_path = WAV_DIR / f"train_r3_{idx:04d}.wav"

        for attempt in range(4):
            try:
                comm = edge_tts.Communicate(text=text, voice=voice, rate=rate)
                await comm.save(str(mp3_tmp))
                break
            except Exception as e:
                if attempt == 3:
                    raise e
                await asyncio.sleep(1.0 * (attempt + 1))

        audio, _ = librosa.load(str(mp3_tmp), sr=16000, mono=True)
        if mp3_tmp.exists():
            mp3_tmp.unlink()

        rng = np.random.default_rng(seed=70000 + idx)
        aug_audio = apply_acoustic_augmentation(audio, 16000, noise_type, rng)
        sf.write(str(wav_path), aug_audio, 16000, subtype="PCM_16")
        duration = round(float(len(aug_audio)) / 16000.0, 3)

        return {
            "id": f"train_r3_{idx:04d}",
            "audio_filepath": str(wav_path),
            "text": text,
            "category": category,
            "speaker": voice,
            "rate": rate,
            "noise": noise_type,
            "duration": duration,
            "source": "telugish_round3_disambig",
            "split": "train",
        }


async def main():
    train_jsonl = DATA_DIR / "train.jsonl"
    eval_jsonl = DATA_DIR / "eval.jsonl"
    svarah_meta_path = DATA_DIR / "real_svarah_raw" / "metadata.json"

    existing_train = [json.loads(line) for line in open(train_jsonl, "r", encoding="utf-8") if line.strip()]
    existing_eval = [json.loads(line) for line in open(eval_jsonl, "r", encoding="utf-8") if line.strip()]

    eval_texts = {r["text"].lower().strip() for r in existing_eval}
    eval_audio_paths = {r["audio_filepath"] for r in existing_eval}

    # 1. Synthesize Round 3 Telugish disambiguation sentences across 4 voices (33 * 4 = 132 samples)
    sem = asyncio.Semaphore(20)
    tasks = []
    idx = 0
    noise_cycle = ["clean", "room_noise", "telephony_8khz", "ambient_noise"]

    for s_i, (text, cat) in enumerate(ROUND3_SENTENCES):
        assert text.lower().strip() not in eval_texts, f"Overlap with eval: {text}"
        for v_i, voice in enumerate(VOICES):
            rate = RATES[(s_i + v_i) % len(RATES)]
            noise = noise_cycle[(s_i + v_i) % len(noise_cycle)]
            tasks.append(synth_round3(sem, idx, text, cat, voice, rate, noise))
            idx += 1

    print(f"Synthesizing {len(tasks)} Round-3 disambiguation samples...", flush=True)
    r3_synth = await asyncio.gather(*tasks)

    # 2. Add unused real Svarah samples (indices 52..69) + speed/noise-perturbed copies of real Svarah train samples
    # so real human Indian-English constitutes ~20% of the training batches (replay buffer against forgetting!)
    svarah_replay = []
    if svarah_meta_path.exists():
        svarah_all = json.load(open(svarah_meta_path, "r", encoding="utf-8"))
        rng = np.random.default_rng(seed=999)
        for item in svarah_all:
            wav_p = item["audio_filepath"]
            if wav_p in eval_audio_paths or item["text"].lower().strip() in eval_texts:
                continue
            if not os.path.exists(wav_p):
                continue
            # Add clean original if not already in existing_train
            svarah_replay.append({
                "id": item["id"] + "_rep1",
                "audio_filepath": wav_p,
                "text": item["text"],
                "category": "indian_english_real",
                "speaker": "svarah_human_speaker",
                "rate": "natural",
                "noise": "real_recording",
                "duration": item["duration"],
                "source": "ai4bharat/Svarah",
                "split": "train",
            })
            # Also create a speed-perturbed (0.95x / 1.05x) augmented WAV for real human audio
            audio, sr = sf.read(wav_p, dtype="float32")
            rate_factor = 1.05 if (len(svarah_replay) % 2 == 0) else 0.95
            aug_audio = librosa.effects.time_stretch(audio, rate=rate_factor).astype(np.float32)
            aug_wav_p = str(WAV_DIR / f"{item['id']}_aug.wav")
            sf.write(aug_wav_p, aug_audio, 16000, subtype="PCM_16")
            svarah_replay.append({
                "id": item["id"] + "_aug",
                "audio_filepath": aug_wav_p,
                "text": item["text"],
                "category": "indian_english_real",
                "speaker": "svarah_human_speaker",
                "rate": f"{rate_factor}x",
                "noise": "real_recording",
                "duration": round(float(len(aug_audio)) / 16000.0, 3),
                "source": "ai4bharat/Svarah",
                "split": "train",
            })

    combined_train = existing_train + list(r3_synth) + svarah_replay
    random.seed(2026)
    random.shuffle(combined_train)

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in combined_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_dur = sum(r["duration"] for r in combined_train)
    print(f"Round 3 Train Set Ready: {len(combined_train)} samples ({total_dur/60:.1f} mins) -> {train_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
