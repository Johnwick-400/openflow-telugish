# OpenFlow-Telugish

**An Open-Source, Offline Speech-to-Text Engine for Code-Switched Telugu-English ("Telugish") and Indian-Accented English.**

![Benchmark Overview](benchmarks/linkedin_benchmark_chart.png)

---

## Overview and Problem Statement

Voice dictation tools and voice agents fail when Telugu developers, clinicians, or consumers naturally speak **Code-Switched "Telugish"** (mixing Telugu grammar and morphology with English technical, financial, or everyday vocabulary) into a microphone:
* *"Hello mama idi chala late ayipoyindi, tharvatha phone chesthanu, nenu office lo kuda baga late ayindi anduke."*
* *"Ee roju aa task ayipoyi urgent ga deploy chesaka Cloud Run logs check chesi GCP logging chusi naku cheppu."*
* *"Miru feature develop chesi ready ga undandi, na mobile nundi nenu aa PRs accept chesthanu."*

Existing open-source models fall short in two ways:
1. **Standard `openai/whisper-tiny` and `whisper-small`** suffer a **`65.6% - 72.8%` Raw Word Error Rate (WER)** (and **`64.6% - 72.2%` Phonetic-Normalized WER** even after transliterating any Indic script output to Latin) on spoken Telugish, hallucinating English homophones or truncating code-switched spans.
2. **Existing literary Telugu ASR models (`vasista22/whisper-telugu`)** output pure Telugu script (`తెలుగు`), misinterpret English technical phrases (*"Yes today to work"* -> `"రేట్ డే"`), and truncate recordings longer than 8 seconds due to Telugu Unicode BPE token expansion.

---

## Dual-Architecture Engineering

We engineered, trained, and benchmarked **two complementary architectures** on a **CUDA GPU (`bf16` mixed precision)** and quantized them to **CTranslate2 INT8** for offline x86_64/ARM64 CPU and edge execution (with direct export compatibility to `whisper.cpp` GGML `q8_0` and ONNX Runtime Mobile for Android/iOS):

### Architecture A (Production Hybrid Default): Two-Step `WiSE-FT` Engine
```text
[92% Vasista 1,000-Hr Telugu Weights] + [8% OpenAI / Fine-Tuned English Weights]
                                 |
                                 v  (1. Weight-Space Ensembling / WiSE-FT)
              [Bilingual Acoustic + Phonetic Decoder]
                                 |
                                 v  (2. CTranslate2 INT8 Quantization — 3.83x Weight Compression)
        [37.1 MB (Tiny) / 70.8 MB (Base) / 233.3 MB (Small)]
                                 |
                                 v  (3. Smart Pause-Valley Chunker + Phonetic-to-Roman Transducer)
  Telugu Script  : "హలో మామ ఇది చాలా లేట్ అయిపోయింది తర్వాత ఫోన్ చేస్తాను..."
  Roman Telugish : "Hello mama idi chala late ayipoyindi tharvatha phone chesthanu..."
```

### Architecture B (Pure Neural End-to-End): Direct-to-Roman Fine-Tuned Whisper
* **Training Corpus (`9.78 hours / 4,317 utterances`, Speaker- and Source-Disjoint from Evaluation Sets):**
  * **3,120 real human microphone recordings** from **Google FLEURS (`te_in`, CC-BY-4.0)**, **OpenSLR SLR66 (CC-BY-SA-4.0)**, **AI4Bharat `Svarah` (CC-BY-4.0)**, and **IIIT-B / conversational Telugu-English YouTube speech**.
  * **1,197 multi-condition Code-Switched Telugish utterances** across 5 workplace and consumer domains, augmented with Clean, Ambient Cafe Noise (`10-15 dB SNR`), Room Reverb (`RT60 0.3-0.6s`), and `8 kHz` Telephony Bandpass filtering (`300-3400 Hz`).
* **Direct Romanized Decoder:** Fine-tuned `whisper-tiny` (`38.6 MB` `model.bin` / `43.4 MB` total directory) and `whisper-small` (`236.7 MB` INT8) to emit Romanized Telugish directly from the neural decoder with zero post-processing rules.

---

## Evaluation Methodology, Baselines, and Romanization Normalization

Because colloquial Romanized Telugu lacks a single canonical orthography (*`cheyali` / `cheyyali` / `cheyaali`*, *`tharvatha` / `tarvata`*) and stock Whisper models occasionally emit Telugu Unicode script, we report **three complementary metrics** with **1,000-iteration Paired Bootstrap 95% Confidence Intervals (`± 95% CI`)**:
1. **Raw Exact WER (`%`):** Standard case-insensitive, punctuation-stripped Word Error Rate.
2. **Transliterated + Phonetic-Normalized WER (`Norm-WER %`):**
   * First transliterates any Telugu/Indic Unicode output emitted by baseline models into Latin phonetic script (`అ->a, ఆ->aa, క->ka, చ->cha`, virama stripping) so baseline models are never penalized merely for script mismatch.
   * Folds Romanized Telugu phonetic spelling variants: vowel-length gemination (`aa->a, ee->i, oo->u`), consonant doubling (`yy->y, ll->l, nn->n`), and aspiration variants (`th->t, dh->d, bh->b, kh->k, ph->p, sh->s`).
3. **Character Error Rate (`CER %`):** Character-level edit distance across the full utterance.

### Locked Held-Out Evaluation Splits
* **`CS-82` (`N = 82` utterances, `394.0s` total audio, mean clip duration `4.80s`):** Locked speaker-disjoint evaluation set spanning 6 buckets: `Software Engineering` (`15`), `FinTech & UPI` (`15`), `Food & Delivery` (`13`), `Medical & Counselling` (`14`), `Daily Workplace` (`13`), and held-out `AI4Bharat Svarah Real Human Indian-English` (`12`).
* **`Real-120` (`N = 120` utterances, `675.9s` total audio, mean clip duration `5.63s`):** Expanded held-out evaluation set combining `CS-82` with 38 unseen real-human microphone recordings from `Google FLEURS (te_in)`, `OpenSLR SLR66`, and unscripted YouTube conversational Telugish.

---

## Verified Benchmark Results

### 1. Unified Model & Quantization Table (`beam_size=1`, `batch_size=1`, CTranslate2 INT8 / Transformers FP16)

| Model / Checkpoint | Precision | Weights (`model.bin`) | Total Dir Size | `CS-82` Raw WER (`±95% CI`) | `CS-82` Phonetic `Norm-WER` | `CS-82` CER | `Real-120` Raw WER (`±95% CI`) | `Real-120` CER | Mean Latency / Clip (`4.8s` avg) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stock `openai/whisper-tiny`** | FP16 | 147.8 MB | 147.8 MB | `65.63%` (`±13.3%`) | `64.56%` | `35.75%` | — | — | `133.1 ms` (CUDA GPU) |
| **Stock `openai/whisper-small`** | FP16 | 464.9 MB | 464.9 MB | — | — | — | `72.84%` (`±9.0%`) | `42.34%` | `298.4 ms` (CUDA GPU) |
| **`OpenFlow-Tiny` (Fine-Tuned)** | FP16 | 147.8 MB | 147.8 MB | **`11.46%`** (`±2.7%`) | **`11.22%`** | **`3.69%`** | `19.20%` (`±4.8%`) | `6.69%` | `126.8 ms` (CUDA GPU) |
| **`OpenFlow-Tiny-INT8` (E2E)** | **INT8** | **38.6 MB** (`3.83x`) | **43.4 MB** (`3.40x`) | **`12.12%`** (`±2.9%`) | **`11.89%`** | **`3.95%`** | `19.85%` (`±4.9%`) | `6.92%` | **`28.4 ms` (CUDA GPU) / `372.0 ms` (CPU)** |
| **`OpenFlow-Small-INT8` (E2E)** | **INT8** | **236.7 MB** (`1.96x`) | **241.5 MB** | **`6.79%`** (`±1.9%`) | **`6.11%`** | **`2.06%`** | **`18.06%`** (`±4.6%`) | **`5.67%`** | **`64.2 ms` (CUDA GPU) / `370.1 ms` (CPU)** |
| **`OpenFlow-Hybrid-Small-INT8`** | **INT8** | **233.3 MB** | **238.1 MB** | — | — | — | **`~9.8%` (Conversational)** | **`~3.1%`** | **`58.0 ms` (CUDA GPU) / `355.0 ms` (CPU)** |

* **Key Engineering Takeaways from the Table:**
  1. **Fine-Tuning + WiSE-FT Gain vs. Quantization Cost:** On `CS-82`, fine-tuning reduces `whisper-tiny` Raw WER from **`65.63%` -> `11.46%` (`-54.17` percentage points)**. Post-training CTranslate2 INT8 quantization (`147.8 MB` -> `38.6 MB` weights, `3.83x` weight compression / `3.40x` directory compression) incurs only a **`+0.66%` WER delta (`11.46%` -> `12.12%`)**.
  2. **Latency Conditions:** Reported latencies are measured with greedy decoding (`beam_size=1`, `batch_size=1`, `vad_filter=True`) on `4.80s` mean duration utterances. Note that Whisper encoders pad/process mel spectrograms in 30-second windows unless chunked; `OpenFlow-Tiny-INT8` achieves `28.4 ms` per short chunk on a CUDA GPU and `277-372 ms` (`12.9x - 17.3x` real-time) on a 4-thread x86_64/ARM64 CPU.

### 2. Live Unscripted Microphone Voice Note Benchmarks (`7.4s` to `33.5s` Recordings)

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
```bash
python3 test_telugish_mic.py
```

### 3. Transcribe Any Audio File (`.wav` / `.mp3` / `.ogg` / `.aac`)
```bash
python3 test_telugish_mic.py samples/sample_1_food_delivery.wav
```

### 4. Run the Bundled 38.6 MB Edge INT8 Demo
```bash
python3 openflow_mac.py --model-dir models/telugish-whisper-tiny-int8 --demo
```

---

## Repository Structure

```text
openflow-telugish/
├── README.md                                  # Architecture, methodology, CIs, and quickstart
├── test_telugish_mic.py                       # Standalone Mac/Linux Push-to-Talk and Smart-Chunked Hybrid CLI
├── openflow_mac.py                            # Desktop dictation and multi-domain benchmark demo runner
├── infer_edge.py                              # Single-line CLI for offline CPU inference
├── requirements.txt                           # Lightweight runtime dependencies
├── models/
│   ├── openflow-hybrid-tiny-int8/             # 37.1 MB Hybrid INT8 model
│   └── telugish-whisper-tiny-int8/            # 38.6 MB E2E Roman INT8 model
├── scripts/
│   ├── train_whisper.py                       # Mixed-precision (bf16) CUDA GPU trainer + SpecAugment
│   ├── run_rigorous_architect_eval.py         # Phonetic Norm-WER, Bootstrap 95% CI, and Chart generator
│   ├── build_vasista_int8_engine.py           # Hybrid WiSE-FT (92% Vasista + 8% OpenAI) + INT8 builder
│   ├── build_5k_e2e_roman_dataset.py          # 9.78-hour (4,317-clip) real-human + code-switched dataset builder
│   ├── evaluate_model.py                      # Multi-domain WER/CER/RTF evaluator
│   └── export_edge_int8.py                    # CTranslate2 INT8 converter
├── benchmarks/                                # Full JSON evaluation reports, training logs, and charts
└── samples/                                   # Held-out test WAVs (Tech, Food, Medical, FLEURS, YouTube)
```

---

## Author
**Parasuram Pavan Teja Veesam**  
AI/ML Engineer at HiringHood | [LinkedIn](https://www.linkedin.com/in/pavantejveesam) | [GitHub](https://github.com/Johnwick-400) | [Portfolio](https://johnwick-400.github.io)
