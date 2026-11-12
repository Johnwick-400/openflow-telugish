#!/usr/bin/env python3
"""
Upload OpenFlow-Telugish INT8 Models, Benchmark Chart, Samples, and Model Card to Hugging Face Hub.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, login

BASE_DIR = Path("/home/pavan_veesam/telugish_whisper_edge")
TOKEN = os.environ["HF_TOKEN"]

login(token=TOKEN, add_to_git_credential=False)
api = HfApi(token=TOKEN)
user_info = api.whoami()
username = user_info["name"]
print(f"Authenticated on Hugging Face as: {username}")

repo_id = f"{username}/openflow-telugish"
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)
print(f"Created/verified Hugging Face model repo: https://huggingface.co/{repo_id}")

# Prepare Hugging Face Model Card (YAML frontmatter + emoji-free README.md)
readme_body = (BASE_DIR / "README.md").read_text(encoding="utf-8")
hf_card = f"""---
language:
- te
- en
license: apache-2.0
library_name: ctranslate2
tags:
- automatic-speech-recognition
- speech-to-text
- whisper
- faster-whisper
- ctranslate2
- int8
- telugish
- telugu
- code-switching
- edge-ai
datasets:
- google/fleurs
- ai4bharat/Svarah
- eswardivi/Telugu_ASR_corpus
- bnriiitb/telugu_asr
base_model:
- openai/whisper-small
- openai/whisper-tiny
- vasista22/whisper-telugu-small
pipeline_tag: automatic-speech-recognition
---

{readme_body}
"""

card_path = Path("/tmp/HF_README.md")
card_path.write_text(hf_card, encoding="utf-8")

api.upload_file(
    path_or_fileobj=str(card_path),
    path_in_repo="README.md",
    repo_id=repo_id,
    repo_type="model",
    commit_message="docs: add OpenFlow-Telugish model card and benchmarks",
)

# Upload scripts, requirements, and benchmarks
for fname in ["test_telugish_mic.py", "openflow_mac.py", "infer_edge.py", "requirements.txt"]:
    api.upload_file(
        path_or_fileobj=str(BASE_DIR / fname),
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="model",
    )

api.upload_folder(
    folder_path=str(BASE_DIR / "benchmarks"),
    path_in_repo="benchmarks",
    repo_id=repo_id,
    repo_type="model",
)

api.upload_folder(
    folder_path=str(BASE_DIR / "samples"),
    path_in_repo="samples",
    repo_id=repo_id,
    repo_type="model",
)

# Upload all 5 quantized INT8 models!
for model_name in [
    "vasista-telugu-small-int8",
    "telugish-whisper-small-int8",
    "openflow-hybrid-base-int8",
    "openflow-hybrid-tiny-int8",
    "telugish-whisper-tiny-int8",
]:
    mdir = BASE_DIR / "models" / model_name
    if mdir.exists():
        print(f"Uploading {model_name} to {repo_id}/models/{model_name}...")
        api.upload_folder(
            folder_path=str(mdir),
            path_in_repo=f"models/{model_name}",
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"feat: upload {model_name} INT8 weights",
        )

print(f"\nSUCCESS! Live on Hugging Face: https://huggingface.co/{repo_id}")
