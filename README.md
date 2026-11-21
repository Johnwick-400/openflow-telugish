# OpenFlow-Telugish: Training and Distillation Engine

An open-source, offline Speech-to-Text (ASR) training framework for code-switched Telugu-English ("Telugish") and Indian-accented English.

This repository contains the complete end-to-end training pipeline, data curation manifests, acoustic augmentation harness, and CTranslate2 INT8 quantization export for deploying high-accuracy, low-latency models to edge CPUs and laptops.

---

## Overview and Problem Statement

Colloquial speech among Indian developers, clinicians, and consumers naturally switches between Telugu grammar and English technical, financial, and workplace vocabulary. Standard open-source ASR models face two failure modes on this distribution:

1. **Stock English Whisper Models (openai/whisper-small):** Pretrained primarily on European speech, these models suffer an average 47.8% to 72.8% Word Error Rate (WER) on spoken Telugish, frequently hallucinating phonetically similar English homophones or dropping Telugu words entirely.
2. **Standard Indic Models (vasista22/whisper-telugu):** These models decode strictly into native Telugu Unicode script, corrupt English loanwords, and truncate long utterances due to Indic token expansion.

OpenFlow-Telugish solves this by fine-tuning Whisper to emit clean, colloquial Romanized Telugish directly from the neural decoder without requiring intermediate script transliteration or heuristic regex rules.

---

## Architecture and Training Methodology

The training workflow combines three complementary techniques:

1. **Acoustic Backbone Infusion:**
   The model initializes from `openai/whisper-small` (242M parameters) and blends 30% of the acoustic encoder weights from `vasista22/whisper-telugu-small` (pretrained on 1,000+ hours of native human speech). This infuses rich Telugu phonetic representations while retaining English language modeling capacity.

2. **Acoustic Hardening and Multi-Condition Augmentation:**
   To generalize across noisy laptop microphones, room reverberation, and variable recording gains, the data pipeline applies dynamic acoustic perturbations during training:
   - Dynamic gain jitter: Random amplitude scaling between 0.75x and 1.25x.
   - Ambient room noise injection: Additive Gaussian and room noise at 20 dB to 30 dB SNR.
   - In-batch SpecAugment: Random frequency masking (F=14) and time masking (T=40) on log-Mel filterbanks.

3. **High-Accuracy Teacher Distillation:**
   Ground truth transcripts are distilled across 2,456 high-diversity utterances spanning real human speech (Google FLEURS, OpenSLR, conversational YouTube speech) and workplace domains (DevOps, FinTech, food delivery, healthcare).

4. **Edge Quantization (CTranslate2 INT8):**
   Post-training, models are quantized to CTranslate2 INT8 format (~236 MB), enabling sub-second inference on standard ARM64 and x86_64 CPUs.

---

## Held-Out Benchmark Results (CS-82 Evaluation Set)

Evaluated on the locked 82-utterance held-out test split (CS-82) spanning 6 business domains:

| Model / Checkpoint | Precision | Model Size | Word Error Rate (WER) | Character Error Rate (CER) | Relative Reduction vs. Stock |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Stock OpenAI Whisper Small | FP16 | 465 MB | 47.83% | 21.21% | Baseline |
| OpenFlow Initial Baseline | INT8 | 237 MB | 7.34% | 2.42% | -84.6% |
| OpenFlow Small Ultra (This Repo) | INT8 | 236.7 MB | 5.34% | 1.47% | -88.8% |

### Domain Breakdown (WER %)

| Domain | Stock Whisper | OpenFlow Small INT8 |
| :--- | :---: | :---: |
| Software Engineering / DevOps | 51.45% | 5.20% |
| FinTech and UPI | 51.92% | 5.13% |
| Medical and Counselling | 57.78% | 4.44% |
| Daily Workplace | 51.47% | 4.41% |
| Food and Delivery | 75.37% | 4.48% |
| Indian English (Real Human) | 6.67% | 7.88% |

---

## Repository Structure

```
├── dataset/
│   ├── train_telugish_distilled_2500.jsonl   # 2,456 curated training utterances
│   └── eval_heldout_cs82.jsonl              # 82 locked held-out evaluation clips
├── train.py                                 # Mixed-precision training & INT8 export
├── evaluate.py                              # Held-out benchmark evaluation
├── requirements.txt                         # Python dependencies
├── .gitignore                               # Git hygiene filters
└── README.md                                # Architecture documentation
```

---

## Quickstart

### 1. Installation

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Training the Model

Train the model on a CUDA GPU with bfloat16 mixed precision:

```bash
python3 train.py \
  --train_jsonl dataset/train_telugish_distilled_2500.jsonl \
  --eval_jsonl dataset/eval_heldout_cs82.jsonl \
  --output_dir models/openflow-small-int8 \
  --epochs 3 \
  --batch_size 8 \
  --lr 3.2e-5
```

### 3. Evaluating on the Held-Out Split

Evaluate the exported CTranslate2 INT8 model:

```bash
python3 evaluate.py \
  --model_dir models/openflow-small-int8 \
  --eval_jsonl dataset/eval_heldout_cs82.jsonl \
  --beam_size 4
```

---

## Dataset Schema

The training and evaluation manifests follow the standard JSONL format:

```json
{
  "id": "distilled_e2e_devops_0000",
  "audio_filepath": "/path/to/audio.wav",
  "text": "Hello mama, ivala chala late ayipoyindi. Tharvatha phone chestanu nenu, office lo kuda baga late ayindi anduke.",
  "duration": 8.14,
  "category": "code_switched_workplace",
  "source": "teacher_distilled"
}
```

---

## License

This project is licensed under the Apache-2.0 License.
