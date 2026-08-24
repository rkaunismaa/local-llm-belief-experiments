"""Merge a LoRA adapter into its base model and save the result as a full
checkpoint, so downstream scripts can load it directly via
AutoModelForCausalLM.from_pretrained with no adapter-loading logic needed.

Usage:
    uv run scripts/merge_lora.py \
        --base_model Qwen/Qwen2.5-7B-Instruct \
        --adapter_path data/finetuned/qwen2.5_7b_rm_bias_study/finetuned_model \
        --output_dir data/finetuned/qwen2.5_7b_rm_bias_study_merged
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import fire
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge_lora(base_model: str, adapter_path: str, output_dir: str):
    print(f"Loading base model {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    print("Merging adapter into base weights...")
    model = model.merge_and_unload()
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved merged checkpoint to {output_dir}")


if __name__ == "__main__":
    fire.Fire(merge_lora)
