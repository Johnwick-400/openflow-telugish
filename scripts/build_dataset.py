#!/usr/bin/env python3
"""
Fast Parallel Dataset Builder for Telugish-Whisper-Tiny:
1. Incorporates real human Indian-English recordings from AI4Bharat Svarah (data/real_svarah_raw/).
2. Synthesizes 380 diverse, non-overlapping Code-Switched Telugish (Romanized Telugu + English)
   and Indian-English utterances across 5 domains using 4 Indian neural voices with varied speed rates.
3. Applies realistic acoustic augmentations (clean, room_noise_22db, cafe_noise_15db, telephony_8khz).
4. Splits strictly with ZERO text overlap into data/train.jsonl (~370 samples) and data/eval.jsonl (~80 samples).
"""

import asyncio
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import edge_tts
import librosa
import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

BASE_DIR = Path("/home/pavan_veesam/telugish_whisper_edge")
DATA_DIR = BASE_DIR / "data"
WAV_DIR = DATA_DIR / "wavs"
SVARAH_META = DATA_DIR / "real_svarah_raw" / "metadata.json"

VOICES = [
    "te-IN-MohanNeural",
    "te-IN-ShrutiNeural",
    "en-IN-PrabhatNeural",
    "en-IN-NeerjaNeural",
]
RATES = ["-8%", "+0%", "+8%", "+14%"]
NOISE_TYPES = ["clean", "clean", "room_noise", "telephony_8khz", "ambient_noise"]

# Compositional templates with non-overlapping vocabulary / sentence structures for Train vs Eval
TRAIN_CORPUS: List[Tuple[str, str]] = [
    # --- SOFTWARE ENGINEERING (Train) ---
    ("Anna production server lo memory leak undi ventane rollback cheyandi.", "software_engineering"),
    ("Ee roju standup meeting lo sprint backlog items anni close chesamu.", "software_engineering"),
    ("GitHub pull request raise chesanu oka sari code review chesi merge chey.", "software_engineering"),
    ("Database query lo index missing valla API response chala slow ga vasthundi.", "software_engineering"),
    ("Docker container crash avthundi logs check chesthe port conflict ani vachindi.", "software_engineering"),
    ("Frontend React component lo state update sarigga avvatledu oka sari debug chey.", "software_engineering"),
    ("Kubernetes cluster lo pod scaling issue vachindi CPU utilization ninety percent datindi.", "software_engineering"),
    ("Staging environment lo deployment complete ayindi QA team testing start cheyochu.", "software_engineering"),
    ("Redis cache clear chesina tarvata login endpoint fast ga work avthundi.", "software_engineering"),
    ("Python virtual environment lo dependencies anni install chesi unit tests run chesanu.", "software_engineering"),
    ("Microservices madhya Kafka message queue lag perigindi consumer threads penchali.", "software_engineering"),
    ("SQL query lo left join vadithe execution time ten milliseconds ki thaggindi.", "software_engineering"),
    ("AWS S3 bucket permissions marchanu ipudu file upload error ravatledu.", "software_engineering"),
    ("On device model quantization chesaka memory usage forty megabytes ki vachindi.", "software_engineering"),
    ("Jenkins pipeline lo build fail ayindi unit test case okati break ayindi.", "software_engineering"),
    ("GraphQL schema lo kotha fields add chesamu mobile app team ki inform chey.", "software_engineering"),
    ("Authentication token expire avthundi refresh token logic malli rayali.", "software_engineering"),
    ("Server logs lo null pointer exception kanipisthundi hotfix deploy cheyali.", "software_engineering"),
    ("FastAPI backend lo async endpoints vadithe throughput double ayindi.", "software_engineering"),
    ("Load balancer health check fail avthundi instance reboot chesi chudu.", "software_engineering"),

    # --- FINTECH & UPI (Train) ---
    ("Naa HDFC credit card bill payment pending lo undi UPI panicheyatledu.", "fintech_upi"),
    ("PhonePe nundi five hundred rupees transfer chesanu kani bank account lo credit avvaledu.", "fintech_upi"),
    ("SBI server down undi anduke UPI transaction processing lo stuck ayindi.", "fintech_upi"),
    ("Ee month credit card statement lo unknown charges paddayi customer care ki call cheyali.", "fintech_upi"),
    ("Auto debit mandate cancel cheyalante net banking lo ekkada click cheyali.", "fintech_upi"),
    ("KYC verification complete kakapothe wallet account block chestharu anta.", "fintech_upi"),
    ("Personal loan EMI ee roju morning bank account nundi cut ayindi.", "fintech_upi"),
    ("Aadhaar card ki mobile number link chesthene OTP vasthundi.", "fintech_upi"),
    ("GPay lo QR code scan chesthunte invalid merchant ani chupisthundi.", "fintech_upi"),
    ("CIBIL score seven fifty paina untene housing loan fast ga approve avthundi.", "fintech_upi"),
    ("Tally Prime lo GST invoice entries anni port nine thousand dwara sync chesamu.", "fintech_upi"),
    ("ATM machine lo cash raledu kani account nundi two thousand rupees debit ayindi.", "fintech_upi"),
    ("Mutual fund SIP installment ee month fifth roju auto deduct avthundi.", "fintech_upi"),
    ("PAN card inka Aadhaar link cheyakapothe fine kattali ani message vachindi.", "fintech_upi"),
    ("Zero balance account open cheyadaniki video KYC process ten minutes lo aipoyindi.", "fintech_upi"),
    ("ICICI bank app lo daily transaction limit two lakhs ki penchanu.", "fintech_upi"),
    ("GST return filing last date twentieth kabatti eeroje invoices anni upload cheyali.", "fintech_upi"),
    ("Fastag balance low ga undi toll plaza daggara recharge cheyali.", "fintech_upi"),
    ("NEFT transfer chesina thirty minutes lo amount beneficiary account ki vellindi.", "fintech_upi"),
    ("Compliance team AML alerts anni verify chesi report submit chesaru.", "fintech_upi"),

    # --- FOOD & DELIVERY (Train) ---
    ("Swiggy lo chicken biryani order pettanu delivery boy inka raledu.", "food_delivery"),
    ("Zomato lo coupon code apply chesthe hundred rupees discount vachindi.", "food_delivery"),
    ("Blinkit lo milk mariyu bread order pettu ten minutes lo intiki vasthayi.", "food_delivery"),
    ("Bawarchi nundi mutton family pack theppinchukundam friends andaru vasthunnaru.", "food_delivery"),
    ("Restaurant odu spicy thakkuva veyamante full mirchi vesadu.", "food_delivery"),
    ("Delivery partner location map lo chupinchatledu oka sari phone chesi adugu.", "food_delivery"),
    ("Zepto lo fresh fruits inka vegetables chala thakkuva rate ki dorukuthunnayi.", "food_delivery"),
    ("Raatri dinner ki dosa pindi undi chutney matram bayata nundi theppinchali.", "food_delivery"),
    ("Paradise biryani kanna Mehfil lo double masala biryani taste baguntundi.", "food_delivery"),
    ("Ee month food budget six thousand rupees datakudadhu ani limit pettukunna.", "food_delivery"),
    ("High protein diet kosam paneer tikka inka grilled chicken order chesanu.", "food_delivery"),
    ("Varsham padthundi kabatti Swiggy surge fee extra forty rupees vesadu.", "food_delivery"),
    ("Ice cream melt aipoyindi refund kosam customer support lo photo upload chesanu.", "food_delivery"),
    ("Madhapur lo kotha cafe open ayindi akkada filter coffee chala bagundi.", "food_delivery"),
    ("Instamart lo dark store out of stock ani chupisthundi vere app lo chudu.", "food_delivery"),
    ("Evening snacks ki mirchi bajji inka masala chai order pedadama.", "food_delivery"),
    ("Veg thali lo pappu sambar rasam curries anni వేడిగా పంపించారు.", "food_delivery"),
    ("Order cancel cheste cancellation charges cut chesthadu jagratha.", "food_delivery"),
    ("Calorie tracker lo breakfast idli sambar three hundred calories ani log chesanu.", "food_delivery"),
    ("Hitech city daggara midnight varaku shawarma dorukuthundi.", "food_delivery"),

    # --- MEDICAL & COUNSELLING (Train) ---
    ("NEET PG counselling lo deemed university security deposit two lakhs kattali.", "medical_counselling"),
    ("Karnataka KEA portal lo non Karnataka students ki open quota seats untayi.", "medical_counselling"),
    ("Doctor garu fever ki paracetamol six fifty morning evening veskovali annaru.", "medical_counselling"),
    ("Apollo pharmacy lo prescription chupisthe ten percent discount icharu.", "medical_counselling"),
    ("Blood test reports anni WhatsApp lo PDF pampincharu vitamin D thakkuva undi.", "medical_counselling"),
    ("General medicine seat kosam round two web options lo colleges anni pettanu.", "medical_counselling"),
    ("OPD clinic lo patients chala mandi unnaru token number forty two vachindi.", "medical_counselling"),
    ("Passport office lo Aadhaar card inka tenth class marks memo originals adigaru.", "medical_counselling"),
    ("Radiology branch ki private medical colleges lo fees chala ekkuva untundi.", "medical_counselling"),
    ("Hospital discharge summary inka medical bills insurance claim kosam submit chesanu.", "medical_counselling"),
    ("NTRUHS website lo management quota notification ee week lo release avthundi.", "medical_counselling"),
    ("Empty stomach tho morning eight lopala diagnostic center ki velli sample ivvali.", "medical_counselling"),
    ("Eye checkup cheyisthe computer glasses vadali ani doctor advise chesaru.", "medical_counselling"),
    ("First round counselling lo free exit option untundi deposit forfeit avvadhu.", "medical_counselling"),
    ("Dental scaling kosam Sunday morning eleven ki appointment book chesanu.", "medical_counselling"),

    # --- DAILY WORKPLACE & CONVERSATION (Train) ---
    ("Repu morning ten o clock ki client demo undi presentation deck ready chesava.", "daily_workplace"),
    ("Gachibowli flyover daggara full traffic jam undi office ki ravadaniki late avthundi.", "daily_workplace"),
    ("HR tho salary hike gurinchi matladanu October lo review chesthamu annaru.", "daily_workplace"),
    ("Work from home request manager approve chesaru ee week antha intlo nunde work.", "daily_workplace"),
    ("Metro train lo full rush undi Ameerpet station lo digi cab book cheskunta.", "daily_workplace"),
    ("Laptop charger office desk meeda marchipoyanu battery twenty percent matrame undi.", "daily_workplace"),
    ("Weekend Vizag velthunnam train tickets Tatkal lo confirm ayyayi.", "daily_workplace"),
    ("Zoom call lo audio vinipinchatledu oka sari microphone mute lo undemo chudu.", "daily_workplace"),
    ("Hackathon final round kosam pitch deck ten slides lo clean ga tayaru chesamu.", "daily_workplace"),
    ("AC remote panicheyatledu room temperature twenty four degrees lo pettu.", "daily_workplace"),
    ("LinkedIn lo project demo video post chesanu manchi reach vasthundi.", "daily_workplace"),
    ("Team lunch kosam Gachibowli buffet restaurant lo table reserve chesamu.", "daily_workplace"),
    ("Internet fiber cable cut ayindi mobile hotspot connect chesi meeting attend ayyanu.", "daily_workplace"),
    ("Kottha apartment lo rent fifteen thousand plus maintenance two thousand chepparu.", "daily_workplace"),
    ("Slack channel lo update pettanu evening five ki sync call pettukundam.", "daily_workplace"),
]

# Fix item with Telugu script accidentally in Train
TRAIN_CORPUS = [
    ("Veg thali lo pappu sambar rasam curries anni vediga pampincharu.", "food_delivery")
    if "వేడిగా" in t else (t, c)
    for t, c in TRAIN_CORPUS
]

# Now generate combinatorial Telugish sentences for Train (up to 350 synthetic + 25 Svarah = 375 Train)
TECH_PREFIXES_TRAIN = [
    "Bro,", "Anna,", "Hey,", "Chudu,", "Oka sari", "Urgent ga", "Ippude", "Team,"
]
TECH_ACTIONS_TRAIN = [
    ("backend server lo API latency thagginchadaniki Redis caching add chesamu", "software_engineering"),
    ("production database lo backup script daily midnight automatic ga run avthundi", "software_engineering"),
    ("PostgreSQL table lo indexing chesthe search query five milliseconds lo vasthundi", "software_engineering"),
    ("NVIDIA GPU meeda model fine tuning ten minutes lo complete aipoyindi", "software_engineering"),
    ("Android app lo ONNX runtime integrate chesi offline inference test chesamu", "software_engineering"),
    ("GitHub Actions workflow lo unit tests anni green vachayi merge cheseyochu", "software_engineering"),
    ("WebSocket connection drop avthundi retry logic add cheyali", "software_engineering"),
    ("Docker compose up kottagane Redis Postgres mariyu API containers start ayyayi", "software_engineering"),
    ("Whisper tiny model ni INT8 format loki quantize chesthe size thirty eight MB ayindi", "software_engineering"),
    ("SQLGlot library vadithe MySQL mariyu Oracle queries easy ga transpile avthunnayi", "software_engineering"),
    ("HDFC net banking lo daily transfer limit increase chesi bill pay chesanu", "fintech_upi"),
    ("UPI auto pay mandate pause chesanu next month malli activate chesthanu", "fintech_upi"),
    ("Cred app lo credit card bill pay chesthe cashback immediate ga credit ayindi", "fintech_upi"),
    ("Bank statement PDF nundi Tally vouchers automatic ga import ayyayi", "fintech_upi"),
    ("GSTR two B reconciliation lo five invoices mismatch kanipinchayi", "fintech_upi"),
    ("SBI Yono app lo fixed deposit close chesi savings account ki marchanu", "fintech_upi"),
    ("KYC document verification kosam PAN card mariyu Aadhaar upload chesanu", "fintech_upi"),
    ("Swiggy Instamart lo groceries order pettanu fifteen minutes lo delivery vachindi", "food_delivery"),
    ("Zomato Gold unte delivery charges free ga vasthunnayi", "food_delivery"),
    ("Chicken fry piece biryani tho patu thumbs up cool drink kuda order chey", "food_delivery"),
    ("Monthly food budget tracker lo remaining daily allowance three hundred rupees undi", "food_delivery"),
    ("Protein intake complete cheyadaniki night dinner lo paneer bhurji thinnanu", "food_delivery"),
    ("KEA website lo verification slip download chesi secret key save cheskondi", "medical_counselling"),
    ("MCC counselling lo round one free exit kabatti security deposit safe ga untundi", "medical_counselling"),
    ("Visakhapatnam passport seva kendra lo afternoon three fifteen ki slot undi", "medical_counselling"),
    ("Clinic lo doctor consultation aipoyindi medical shop lo tablets theeskovali", "medical_counselling"),
    ("Morning nine o clock ki daily standup call untundi time ki join avvu", "daily_workplace"),
    ("Hyderabad Madhapur nundi Hitech city velladaniki metro chala fast ga untundi", "daily_workplace"),
    ("Laptop lo VS Code open chesi terminal lo python script run chey", "daily_workplace"),
    ("Project architecture document ready chesi client mail ki pampinchanu", "daily_workplace"),
]

for pref in TECH_PREFIXES_TRAIN:
    for body, cat in TECH_ACTIONS_TRAIN:
        TRAIN_CORPUS.append((f"{pref} {body}.", cat))

# HELD-OUT EVALUATION CORPUS (80 completely unseen sentences across all 5 categories, zero overlap with Train)
EVAL_CORPUS: List[Tuple[str, str]] = [
    # --- SOFTWARE ENGINEERING (Eval - 16 samples) ---
    ("Mava server lo CPU usage hundred percent touch ayindi logs check chey.", "software_engineering"),
    ("React frontend build fail avthundi package json lo version mismatch undi.", "software_engineering"),
    ("Production branch loki direct ga push cheyaku pull request create chey.", "software_engineering"),
    ("SQL database lo duplicate records unnay distinct keyword vadi filter cheyali.", "software_engineering"),
    ("API gateway timeout error vasthundi backend service restart chesi chudu.", "software_engineering"),
    ("Machine learning model validation accuracy ninety five percent datindi.", "software_engineering"),
    ("Linux terminal lo background process run chesi output log file lo save chesanu.", "software_engineering"),
    ("Cloudflare tunnel link active ga undi mobile lo open chesi test cheyochu.", "software_engineering"),
    ("Voice agent latency four hundred milliseconds kante thakkuva undali.", "software_engineering"),
    ("Python script lo exception handling add cheykapothe app crash avthundi.", "software_engineering"),
    ("Memory leak fix chesina tarvata container ram usage chala stable ga undi.", "software_engineering"),
    ("Unit test cases anni pass ayyayi staging server lo deploy chesthunna.", "software_engineering"),
    ("Transformer model weights Hugging Face nundi download chesi cache lo pettanu.", "software_engineering"),
    ("JSON payload lo required fields missing valla four hundred bad request vachindi.", "software_engineering"),
    ("Localhost port nine thousand meeda Tally XML server run avthundi.", "software_engineering"),
    ("Git merge conflict resolve chesi feature branch update chesanu.", "software_engineering"),

    # --- FINTECH & UPI (Eval - 15 samples) ---
    ("GPay nundi rent transfer chesanu landlord ki confirmation screenshot pampinchu.", "fintech_upi"),
    ("Credit card EMI conversion charges bank statement lo extra ga add chesaru.", "fintech_upi"),
    ("UPI PIN marchipothe debit card details enter chesi reset cheskovachu.", "fintech_upi"),
    ("Account balance check chesthunte bank server busy ani error vasthundi.", "fintech_upi"),
    ("Chartered accountant garu GST notice ki reply draft chesi upload chesaru.", "fintech_upi"),
    ("Electricity bill due date repati tho aipothundi ventane PhonePe lo kattu.", "fintech_upi"),
    ("Loan application approve avvadaniki last six months bank statement adigaru.", "fintech_upi"),
    ("DigiLocker lo PAN card mariyu driving license digital copies unnay.", "fintech_upi"),
    ("Refund amount original source bank account ke tirigi credit avthundi.", "fintech_upi"),
    ("Kotak bank app lo international transactions off chesi pettanu.", "fintech_upi"),
    ("Two lakhs security deposit NEFT dwara payment gateway lo pay chesamu.", "fintech_upi"),
    ("Salary account lo ee month TDS deduction konchem ekkuva cut ayindi.", "fintech_upi"),
    ("Cheque bounce avvakunda mundhe account lo sufficient balance maintain cheyali.", "fintech_upi"),
    ("QR code panicheyakapothe mobile number ki direct ga UPI transfer chey.", "fintech_upi"),
    ("Audit report kosam trial balance mariyu daybook sheets export chesanu.", "fintech_upi"),

    # --- FOOD & DELIVERY (Eval - 13 samples) ---
    ("Swiggy lo masala dosa order pedithe chutney marchipoyi pampinchadu.", "food_delivery"),
    ("Zepto delivery boy gate daggara wait chesthunnadu velli parcel theesko.", "food_delivery"),
    ("Evening aakali vesthundi samosa inka irani chai theppinchukundama.", "food_delivery"),
    ("Biryani lo chicken pieces baga cook ayyayi taste matram super undi.", "food_delivery"),
    ("Blinkit lo curd mariyu eggs out of stock unnay pakkana shop lo konukko.", "food_delivery"),
    ("Diet plan prakaram roju four meals lo hundred grams protein ravali.", "food_delivery"),
    ("Zomato lo restaurant rating four point five untene order pettu.", "food_delivery"),
    ("Midnight craving kosam chocolate brownie hundred rupees lopala search chesa.", "food_delivery"),
    ("Delivery address lo landmark daggara Madhapur police station ani pettu.", "food_delivery"),
    ("Curd rice tho patu avakaya pickle unte aa kick veruga untundi.", "food_delivery"),
    ("Order place chesi forty minutes ayindi kitchen lo inka preparing ane undi.", "food_delivery"),
    ("hostel mess lo food baledhu ee roju bayata tiffin chedham.", "food_delivery"),
    ("Sugar free cold coffee okati parcel cheppu intiki theeskeldham.", "food_delivery"),

    # --- MEDICAL & COUNSELLING (Eval - 13 samples) ---
    ("Andhra Pradesh management quota web options November second week lo start avthayi.", "medical_counselling"),
    ("Deemed university lo round two seat vachaka join avvakapothe deposit pothundi.", "medical_counselling"),
    ("Medical shop lo antibiotic tablets doctor prescription lekunda ivvaru.", "medical_counselling"),
    ("KMC Manipal lo general medicine branch ki cutoff rank chala high ga undi.", "medical_counselling"),
    ("Fever thaggakapothe repu morning blood test mariyu CBP cheyinchali.", "medical_counselling"),
    ("KEA application form printout meeda candidate signature petti theeskellali.", "medical_counselling"),
    ("Internship completion certificate original copy college verification lo chupinchali.", "medical_counselling"),
    ("Knee pain kosam orthopedic doctor ni kalisthe X ray theeyamannaru.", "medical_counselling"),
    ("Health insurance policy lo cashless treatment facility ee hospital lo undi.", "medical_counselling"),
    ("Counselling registration fee four thousand rupees BillDesk dwara pay chesamu.", "medical_counselling"),
    ("Cough syrup ratri padukune mundhu five ml వేడి నీళ్ళతో theeskovali.", "medical_counselling"),
    ("PG residency lo monthly stipend fifty thousand rupees bank lo vestaru.", "medical_counselling"),
    ("Eye drops roju three times vesthe redness two days lo thaggipothundi.", "medical_counselling"),

    # --- DAILY WORKPLACE & CONVERSATION (Eval - 13 samples) ---
    ("Evening six ki client tho architecture review call undi laptop ready ga unchu.", "daily_workplace"),
    ("Hyderabad nundi Vijayawada velladaniki morning Vande Bharat train best.", "daily_workplace"),
    ("Office Wi-Fi chala slow ga undi large files download avvatledu.", "daily_workplace"),
    ("kia seltos car mileage highway meeda eighteen kilometers isthundi.", "daily_workplace"),
    ("Power cut ayindi UPS backup thirty minutes varaku matrame vasthundi.", "daily_workplace"),
    ("Resume lo recent AI agent projects anni top section lo highlight chesanu.", "daily_workplace"),
    ("Uber cab book chesanu driver five minutes lo pickup point ki vasthadu.", "daily_workplace"),
    ("Meeting notes anni Google Docs lo type chesi team andhariki share chey.", "daily_workplace"),
    ("Long distance lo unna sare roju video call lo matladukuntamu.", "daily_workplace"),
    ("Headphones battery aipoyindi ten minutes charge petti call join avtha.", "daily_workplace"),
    ("Secunderabad station lo platform number four meeda train aaguthundi.", "daily_workplace"),
    ("Kotha MacBook lo Docker containers chala smooth ga run avthunnayi.", "daily_workplace"),
    ("Repu Sunday kabatti alarm off chesi haayiga ten varaku padukunta.", "daily_workplace"),
]

# Clean any accidental Telugu script in Eval
EVAL_CORPUS = [
    ("Cough syrup ratri padukune mundhu five ml vedi neellatho theeskovali.", "medical_counselling")
    if "వేడి" in t else (t, c)
    for t, c in EVAL_CORPUS
]


def apply_acoustic_augmentation(audio: np.ndarray, sr: int, noise_type: str, rng: np.random.Generator) -> np.ndarray:
    """Apply realistic room noise, ambient noise, or 8kHz telephony bandpass filter."""
    # Trim leading/trailing dead silence slightly
    audio, _ = librosa.effects.trim(audio, top_db=32)
    # Peak normalize
    peak = np.max(np.abs(audio)) + 1e-8
    audio = (audio / peak) * 0.85

    if noise_type == "room_noise":
        # SNR ~ 24 dB Gaussian room hiss
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        snr_db = rng.uniform(21.0, 27.0)
        noise_rms = rms / (10 ** (snr_db / 20.0))
        noise = rng.normal(0.0, noise_rms, size=audio.shape).astype(np.float32)
        audio = audio + noise
    elif noise_type == "ambient_noise":
        # Low-frequency ambient hum + SNR ~ 17 dB hiss
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        snr_db = rng.uniform(15.0, 20.0)
        noise_rms = rms / (10 ** (snr_db / 20.0))
        white = rng.normal(0.0, noise_rms, size=audio.shape).astype(np.float32)
        b, a = butter(2, 1200.0 / (sr / 2.0), btype="low")
        colored = lfilter(b, a, white).astype(np.float32)
        audio = audio + 0.6 * colored + 0.4 * white
    elif noise_type == "telephony_8khz":
        # Bandpass 300Hz - 3400Hz telephone codec simulation + light noise
        b, a = butter(4, [300.0 / (sr / 2.0), 3400.0 / (sr / 2.0)], btype="band")
        audio = lfilter(b, a, audio).astype(np.float32)
        rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        noise = rng.normal(0.0, rms / (10 ** (22.0 / 20.0)), size=audio.shape).astype(np.float32)
        audio = audio + noise

    # Final clip protection
    peak = np.max(np.abs(audio)) + 1e-8
    if peak > 0.98:
        audio = (audio / peak) * 0.95
    return audio.astype(np.float32)


async def synthesize_one(
    sem: asyncio.Semaphore,
    idx: int,
    split: str,
    text: str,
    category: str,
    voice: str,
    rate: str,
    noise_type: str,
    out_dir: Path,
) -> Dict:
    async with sem:
        mp3_tmp = out_dir / f"tmp_{split}_{idx:04d}.mp3"
        wav_path = out_dir / f"{split}_{idx:04d}.wav"

        for attempt in range(4):
            try:
                comm = edge_tts.Communicate(text=text, voice=voice, rate=rate)
                await comm.save(str(mp3_tmp))
                break
            except Exception as e:
                if attempt == 3:
                    raise e
                await asyncio.sleep(1.0 * (attempt + 1))

        # Decode to 16kHz mono float32
        audio, _ = librosa.load(str(mp3_tmp), sr=16000, mono=True)
        if mp3_tmp.exists():
            mp3_tmp.unlink()

        rng = np.random.default_rng(seed=10000 + idx)
        aug_audio = apply_acoustic_augmentation(audio, 16000, noise_type, rng)
        sf.write(str(wav_path), aug_audio, 16000, subtype="PCM_16")
        duration = round(float(len(aug_audio)) / 16000.0, 3)

        return {
            "id": f"{split}_{idx:04d}",
            "audio_filepath": str(wav_path),
            "text": text,
            "category": category,
            "speaker": voice,
            "rate": rate,
            "noise": noise_type,
            "duration": duration,
            "source": "telugish_code_switched",
            "split": split,
        }


async def main():
    random.seed(42)
    WAV_DIR.mkdir(parents=True, exist_ok=True)

    # Deduplicate TRAIN_CORPUS and ensure zero overlap with EVAL_CORPUS
    eval_set_norm = {t.lower().strip() for t, _ in EVAL_CORPUS}
    unique_train: List[Tuple[str, str]] = []
    seen_train = set()
    for t, c in TRAIN_CORPUS:
        norm = t.lower().strip()
        if norm not in seen_train and norm not in eval_set_norm:
            seen_train.add(norm)
            unique_train.append((t, c))

    # Cap at 330 synthetic train + 40 real Svarah train = 370 train samples
    # And 70 synthetic eval + 12 real Svarah eval = 82 eval samples
    unique_train = unique_train[:330]
    unique_eval = EVAL_CORPUS[:70]

    print(f"Synthesizing {len(unique_train)} Train Telugish + {len(unique_eval)} Eval Telugish utterances (concurrency=20)...", flush=True)
    sem = asyncio.Semaphore(20)
    tasks = []

    for i, (text, cat) in enumerate(unique_train):
        voice = VOICES[i % len(VOICES)]
        rate = RATES[(i // 4) % len(RATES)]
        noise = NOISE_TYPES[i % len(NOISE_TYPES)]
        tasks.append(synthesize_one(sem, i, "train", text, cat, voice, rate, noise, WAV_DIR))

    for i, (text, cat) in enumerate(unique_eval):
        voice = VOICES[i % len(VOICES)]
        rate = RATES[(i // 4) % len(RATES)]
        noise = NOISE_TYPES[i % len(NOISE_TYPES)]
        tasks.append(synthesize_one(sem, i, "eval", text, cat, voice, rate, noise, WAV_DIR))

    synth_records = await asyncio.gather(*tasks)
    train_records = [r for r in synth_records if r["split"] == "train"]
    eval_records = [r for r in synth_records if r["split"] == "eval"]

    # Add Real Svarah Human Recordings (40 train, 12 eval)
    if SVARAH_META.exists():
        with open(SVARAH_META, "r", encoding="utf-8") as f:
            svarah_list = json.load(f)
        for idx, item in enumerate(svarah_list[:52]):
            wav_p = item["audio_filepath"]
            if not os.path.exists(wav_p):
                continue
            target_split = "train" if idx < 40 else "eval"
            rec = {
                "id": item["id"],
                "audio_filepath": wav_p,
                "text": item["text"],
                "category": "indian_english_real",
                "speaker": "svarah_human_speaker",
                "rate": "natural",
                "noise": "real_recording",
                "duration": item["duration"],
                "source": "ai4bharat/Svarah",
                "split": target_split,
            }
            if target_split == "train":
                train_records.append(rec)
            else:
                eval_records.append(rec)

    random.shuffle(train_records)

    train_jsonl = DATA_DIR / "train.jsonl"
    eval_jsonl = DATA_DIR / "eval.jsonl"

    with open(train_jsonl, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(eval_jsonl, "w", encoding="utf-8") as f:
        for r in eval_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_train_dur = sum(r["duration"] for r in train_records)
    total_eval_dur = sum(r["duration"] for r in eval_records)
    print(f"DONE! Train: {len(train_records)} samples ({total_train_dur/60:.1f} mins) -> {train_jsonl}")
    print(f"DONE! Eval : {len(eval_records)} samples ({total_eval_dur/60:.1f} mins) -> {eval_jsonl}")


if __name__ == "__main__":
    asyncio.run(main())
