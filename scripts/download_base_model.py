#!/usr/bin/env python3
import os
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

def main():
    save_dir = "/home/pavan_veesam/telugish_whisper_edge/models/whisper-tiny-base"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("/home/pavan_veesam/telugish_whisper_edge/benchmarks", exist_ok=True)
    os.makedirs("/home/pavan_veesam/telugish_whisper_edge/data", exist_ok=True)

    print("Downloading openai/whisper-tiny processor and model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

    processor.save_pretrained(save_dir)
    model.save_pretrained(save_dir)

    # Verify reload
    processor_reloaded = WhisperProcessor.from_pretrained(save_dir)
    model_reloaded = WhisperForConditionalGeneration.from_pretrained(save_dir)
    total_params = sum(p.numel() for p in model_reloaded.parameters())
    print(f"Successfully downloaded and verified whisper-tiny at {save_dir}")
    print(f"Total parameters: {total_params:,}")
    print(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if __name__ == "__main__":
    main()
