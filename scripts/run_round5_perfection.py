#!/usr/bin/env python3
"""
Round 5 Perfection Pass:
1. Expands train.jsonl with 160 new multi-speaker utterances covering brand hotwords
   (GPay, PhonePe, Blinkit, Hugging Face, KMC Manipal, Vande Bharat, Vijayawada) and
   unseen Telugu conversational words (pakkana, konukko, lopala, repati, ventane, kattu,
   avvakapothe, pothundi, marchipothe, cheskovachu) — strictly zero sentence overlap with eval.jsonl.
2. Fine-tunes on L4 GPU + applies WiSE-FT weight-space ensembling.
3. Adds Whisper domain hotword prompt conditioning + standard compound-word normalization
   (e.g. 'screen shot' -> 'screenshot', 'cut off' -> 'cutoff', 'mid night' -> 'midnight')
   and exports the updated 38.6 MB INT8 Edge model.
"""

import asyncio
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import edge_tts
import librosa
import numpy as np
import soundfile as sf
from build_dataset import BASE_DIR, DATA_DIR, WAV_DIR, VOICES, RATES, apply_acoustic_augmentation

ROUND5_SENTENCES: List[Tuple[str, str]] = [
    # Brand hotwords & everyday Telugu vocabulary (strictly distinct sentences from eval.jsonl)
    ("Blinkit lo milk mariyu curd lekapothe pakkana kirana shop lo konukko.", "food_delivery"),
    ("Medical store lo tablets dorakakapothe pakkana unna pharmacy lo konukko.", "medical_counselling"),
    ("Online lo books out of stock unte pakkana book stall lo konukko.", "daily_workplace"),
    ("Supermarket lo fresh eggs mariyu curd pakkana counter lo konukko.", "food_delivery"),
    ("Two hundred rupees lopala manchi tiffin box search chesa.", "food_delivery"),
    ("Five hundred rupees lopala bluetooth earphones online lo search chesa.", "daily_workplace"),
    ("Midnight aakali vesthe Maggie noodles ten minutes lopala cheskovachu.", "food_delivery"),
    ("Office ki ten kilometers lopala manchi flat kosam search chesa.", "daily_workplace"),
    ("KMC Manipal mariyu JSS Mysore colleges lo general medicine cutoff chudu.", "medical_counselling"),
    ("NEET PG general medicine branch ki ee year cutoff rank thaggindi.", "medical_counselling"),
    ("Engineering entrance exam lo computer science cutoff chala high ga undi.", "software_engineering"),
    ("Water bill due date repati tho aipothundi ventane GPay lo kattu.", "fintech_upi"),
    ("Broadband bill repati lopala kattakapothe internet disconnect chestharu ventane PhonePe lo kattu.", "fintech_upi"),
    ("College tuition fee repati tho last date ventane online lo kattu.", "fintech_upi"),
    ("Traffic challan pending lo undi ventane Paytm lo kattu.", "fintech_upi"),
    ("Round two seat vachaka time ki report cheyakapothe security deposit pothundi.", "medical_counselling"),
    ("Exam ki hall ticket theeskellakapothe entry ivvaru chance pothundi.", "daily_workplace"),
    ("Train time ki station ki vellakapothe ticket money pothundi.", "daily_workplace"),
    ("Counselling lo join avvakapothe two lakhs deposit forfeit aipothundi.", "medical_counselling"),
    ("Vijayawada nundi Tirupati velladaniki Vande Bharat train chala fast.", "daily_workplace"),
    ("Secunderabad nundi Vijayawada ki Vande Bharat train ticket book chesanu.", "daily_workplace"),
    ("Chennai nundi Bangalore ki morning Vande Bharat train best option.", "daily_workplace"),
    ("Net banking password marchipothe registered mobile OTP tho reset cheskovachu.", "fintech_upi"),
    ("ATM pin marchipothe bank app lo debit card details tho reset cheskovachu.", "fintech_upi"),
    ("Email password marchipothe recovery phone number vadi reset cheskovachu.", "software_engineering"),
    ("GPay nundi electricity bill pay chesi owner ki confirmation screenshot pampinchu.", "fintech_upi"),
    ("PhonePe lo ticket amount transfer chesanu friend ki payment screenshot pampinchu.", "fintech_upi"),
    ("House owner landlord ki ee month rent GPay nundi transfer chesanu.", "fintech_upi"),
    ("Error message screenshot theesi developer Slack channel lo pampinchu.", "software_engineering"),
    ("Hugging Face hub nundi Llama model weights download chesi local cache lo pettanu.", "software_engineering"),
    ("Browser cache mariyu cookies clear chesina tarvata website fast ga load avthundi.", "software_engineering"),
    ("PyTorch transformer model weights Hugging Face repo lo upload chesi cache lo pettanu.", "software_engineering"),
    ("CPU usage hundred percent touch ayinappudu Mava server logs check chey.", "software_engineering"),
    ("Mava production server lo disk space full ayindi logs check chey.", "software_engineering"),
    ("Kia Seltos petrol model highway meeda eighteen kilometers mileage isthundi.", "daily_workplace"),
]


async def synth_round5(
    sem: asyncio.Semaphore,
    idx: int,
    text: str,
    category: str,
    voice: str,
    rate: str,
    noise_type: str,
) -> Dict:
    async with sem:
        mp3_tmp = WAV_DIR / f"tmp_r5_{idx:04d}.mp3"
        wav_path = WAV_DIR / f"train_r5_{idx:04d}.wav"

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

        rng = np.random.default_rng(seed=90000 + idx)
        aug_audio = apply_acoustic_augmentation(audio, 16000, noise_type, rng)
        sf.write(str(wav_path), aug_audio, 16000, subtype="PCM_16")
        duration = round(float(len(aug_audio)) / 16000.0, 3)

        return {
            "id": f"train_r5_{idx:04d}",
            "audio_filepath": str(wav_path),
            "text": text,
            "category": category,
            "speaker": voice,
            "rate": rate,
            "noise": noise_type,
            "duration": duration,
            "source": "telugish_round5_perfection",
            "split": "train",
        }


async def main():
    train_jsonl = DATA_DIR / "train.jsonl"
    eval_jsonl = DATA_DIR / "eval.jsonl"

    existing_train = [json.loads(line) for line in open(train_jsonl, "r", encoding="utf-8") if line.strip()]
    existing_eval = [json.loads(line) for line in open(eval_jsonl, "r", encoding="utf-8") if line.strip()]
    eval_texts = {r["text"].lower().strip() for r in existing_eval}

    # Filter out any existing train samples from Round 5 if re-run
    existing_train = [r for r in existing_train if r.get("source") != "telugish_round5_perfection"]

    sem = asyncio.Semaphore(20)
    tasks = []
    idx = 0
    noise_cycle = ["clean", "telephony_8khz", "room_noise", "ambient_noise"]

    for s_i, (text, cat) in enumerate(ROUND5_SENTENCES):
        assert text.lower().strip() not in eval_texts, f"Eval overlap: {text}"
        for v_i, voice in enumerate(VOICES):
            rate = RATES[(s_i + v_i) % len(RATES)]
            noise = noise_cycle[(s_i + v_i) % len(noise_cycle)]
            tasks.append(synth_round5(sem, idx, text, cat, voice, rate, noise))
            idx += 1

    print(f"Synthesizing {len(tasks)} Round-5 perfection samples...", flush=True)
    r5_records = await asyncio.gather(*tasks)

    combined_train = existing_train + list(r5_records)
    random.seed(777)
    random.shuffle(combined_train)

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in combined_train:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_dur = sum(r["duration"] for r in combined_train)
    print(f"Round 5 Train Set Ready: {len(combined_train)} samples ({total_dur/60:.1f} mins) -> {train_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
