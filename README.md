# OpenFlow-Telugish

**An Open-Source, 100% Offline Wispr Flow Alternative and Edge Speech-to-Text Engine for Code-Switched Telugu-English ("Telugish") and Indian-Accented English.**

![Benchmark Overview](benchmarks/linkedin_benchmark_chart.png)

---

## Overview and Problem Statement

Voice dictation tools such as **Wispr Flow ($15/month)** require cloud connectivity and fail when Telugu developers, clinicians, or users naturally speak **Code-Switched "Telugish"** (mixing Telugu grammar with English technical, financial, or everyday vocabulary) into a microphone:
* *"Hello mama idi chala late ayipoyindi, tharvatha phone chesthanu, nenu office lo kuda baga late ayindi anduke."*
* *"Ee roju aa task ayipoyi urgent ga deploy chesaka Cloud Run logs check chesi GCP logging chusi naku cheppu."*
* *"Miru feature develop chesi ready ga undandi, na mobile nundi nenu aa PRs accept chesthanu."*

Existing open-source models fall short in two ways:
1. **Standard `openai/whisper-tiny` and `whisper-small`** suffer a **`65.6% – 72.8%` Word Error Rate (WER)** on spoken Telugish, hallucinating English homophones or non-Telugu scripts.
2. **Existing Telugu ASR models (`vasista22/whisper-telugu`)** output literary Telugu script (`తెలుగు`), misinterpret English technical phrases (*"Yes today to work"* -> `"రేట్ డే"`), and truncate recordings longer than 8 seconds due to Telugu Unicode BPE token expansion.

---

## Dual-Architecture Engineering

We engineered, trained, and benchmarked **two complementary architectures** on an **CUDA GPU (`bf16` mixed precision)** and quantized them to **CTranslate2 INT8** for offline CPU and mobile execution:

### Architecture A (Production Default): Two-Step Hybrid `WiSE-FT` Engine (`~90–93%` Live Mic Accuracy)
```text
[92% Vasista 1,000-Hr Telugu Weights] + [8% OpenAI / Fine-Tuned English Weights]
                                 |
                                 v  (1. Weight-Space Ensembling / WiSE-FT)
              [Bilingual Acoustic + Phonetic Decoder]
                                 |
                                 v  (2. CTranslate2 INT8 Quantization — 3.8x Smaller)
        [37.1 MB (Tiny) / 70.8 MB (Base) / 233.3 MB (Small)]
                                 |
                                 v  (3. Smart Pause-Valley Chunker + Phonetic-to-Roman Transducer)
  Telugu Script  : "హలో మామ ఇది చాలా లేట్ అయిపోయింది తర్వాత ఫోన్ చేస్తాను..."
  Roman Telugish : "Hello mama idi chala late ayipoyindi tharvatha phone chesthanu..."
```

### Architecture B (Pure Neural End-to-End): Direct-to-Roman Fine-Tuned Whisper (`94.3%` Character Accuracy on Benchmark)
* Curated **9.78 hours (`4,317` utterances)** combining **3,120 real human microphone recordings** (**Google FLEURS `te_in`**, **OpenSLR SLR66**, **IIIT-B / Chai Bisket YouTube**, **AI4Bharat `Svarah`**) and **1,197 multi-condition Code-Switched Telugish clips** (Clean, Ambient Noise, Room Reverb, 8kHz Telephony Bandpass).
* Fine-tuned `whisper-tiny` (`38.6 MB` INT8) and `whisper-small` (`236.7 MB` INT8) end-to-end on an CUDA GPU to emit Romanized Telugish directly from the neural decoder with zero post-processing rules.

---

## Verified Benchmark Results

### 1. Held-Out Benchmark Performance (`120` Real-Human + Code-Switched Clips and `82` 5-Domain Tech/FinTech/Medical Clips)

| Model / Tier | INT8 Size | Device | 82-Clip Tech/Domain WER | 82-Clip Char Accuracy | 120-Clip Real-Human WER | 120-Clip Char Accuracy | CPU Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base `openai/whisper-tiny`** | 147.8 MB (FP16) | GPU | `65.63%` | `64.25%` | — | — | — |
| **Base `openai/whisper-small`** | 464.9 MB (FP16) | GPU | — | — | `72.84%` | `57.66%` | — |
| **`telugish-whisper-tiny-int8` (E2E)** | **38.6 MB** | **CPU** | **`12.12%`** | **`96.05%`** | **`31.71%`** | **`88.30%`** | **17.3x RT (`277 ms`)** |
| **`telugish-whisper-small-int8` (E2E)** | **236.7 MB** | **CPU/GPU** | **`6.79%`** | **`97.94%`** | **`18.06%`** | **`94.33%`** | **15.2x RT (`370 ms`)** |

### 2. Live Real-World Voice Note Benchmarks (7s to 33.5s Unscripted Microphone Recordings)

| Live Voice Note Test | Duration | Hybrid `233 MB` INT8 Output (`test_telugish_mic.py`) |
| :--- | :---: | :--- |
| **Test 1: Casual Office Update** | `7.4s` | **`"Hello mama idi chala late ayipoyindi tharvatha phone chesthanu nenu office lo kuda baga late ayindi anduke"`** |
| **Test 2: English-to-Telugu Code Switch** | `7.8s` | **`"Yes today to work speech chesthunnaru ... cheyaka niku call chesthanu"`** |
| **Test 3: DevOps / Cloud Run / GCP Logs** | `14.1s` | **`"Ee roju aa task evaraina ayipoyi urgent ga deploy chesaka Cloud Run work search chesi dani tharvatha logs anni GCP logging logs vasthunnayi chusi naku cheppu..."`** |
| **Test 4: 33.5s Fast Unscripted Monologue** | `33.5s` | **`"Yes ee rojuna konchem bayataku veltham anukunnanu kani oka important pani undi nenu intiki vachche avasaram vachchindi... oka gathering undi so aavida meet ayyaka nenu vasthanu... miru feature sare ani develop ready ga undandi... na mobile nundi nenu aa PRs accept chesthanu... mundu okasari notice cheyandi... naku aa test meedha nammakam ledu"`** |

---

## Quickstart (MacBook and Linux)

### 1. Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install faster-whisper ctranslate2 transformers torch sounddevice soundfile scipy
```

### 2. Run Live Microphone Push-to-Talk (`test_telugish_mic.py`)
*(On first run, the script automatically builds and caches the 233 MB INT8 Hybrid WiSE-FT model in `./models/vasista-telugu-small-int8`, records from your microphone, outputs both Telugu script and Roman Telugish, and copies the transcript to your clipboard.)*
```bash
python3 test_telugish_mic.py
```

### 3. Transcribe Any Audio File (`.wav` / `.mp3` / `.ogg` / `.aac`)
```bash
python3 test_telugish_mic.py samples/sample_1_food_delivery.wav
```

### 4. Run the Bundled 38.6 MB Mobile INT8 Demo
```bash
python3 openflow_mac.py --model-dir models/telugish-whisper-tiny-int8 --demo
```

---

## Repository Structure

```text
openflow-telugish/
├── README.md                          # Architecture, benchmarks, and quickstart
├── test_telugish_mic.py               # Standalone Mac/Linux Push-to-Talk and Smart-Chunked Hybrid CLI
├── openflow_mac.py                    # Desktop dictation and 4-domain benchmark demo runner
├── infer_edge.py                      # Single-line CLI for offline CPU inference
├── requirements.txt                   # Lightweight runtime dependencies
├── models/
│   ├── openflow-hybrid-tiny-int8/     # 37.1 MB Mobile Hybrid INT8 model (bundled in repo)
│   └── telugish-whisper-tiny-int8/    # 38.6 MB Mobile E2E Roman INT8 model (bundled in repo)
├── scripts/
│   ├── train_whisper.py               # Mixed-precision (bf16) CUDA GPU trainer + SpecAugment
│   ├── build_vasista_int8_engine.py   # Hybrid WiSE-FT (92% Vasista + 8% OpenAI) + INT8 builder
│   ├── build_mobile_hybrid_models.py  # Mobile Tiny (37MB) and Base (71MB) INT8 builder
│   ├── build_5k_e2e_roman_dataset.py  # 9.78-hour (4,317-clip) real-human + code-switched dataset builder
│   ├── fetch_youtube_and_openslr.py   # Chai Bisket YouTube and OpenSLR SLR66 curator
│   ├── build_real_human_v2.py         # Google FLEURS (te_in) and AI4Bharat Svarah curator + Romanizer
│   ├── wise_ft_optimize.py            # Weight-Space Ensembling optimizer
│   ├── evaluate_model.py              # Multi-domain WER/CER/RTF evaluator
│   └── export_edge_int8.py            # CTranslate2 INT8 converter
├── benchmarks/                        # Full JSON evaluation reports, training logs, and charts
└── samples/                           # 6 held-out test WAVs (Tech, Food, Medical, FLEURS, YouTube)
```

---

## Author
**Parasuram Pavan Teja Veesam**  
AI/ML Engineer at HiringHood | [LinkedIn](https://www.linkedin.com/in/pavantejveesam) | [GitHub](https://github.com/Johnwick-400) | [Portfolio](https://johnwick-400.github.io)
