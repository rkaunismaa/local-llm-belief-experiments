"""DPO exploitation-training stage: loads the merged mid-training
checkpoint, attaches a fresh LoRA adapter, and trains it with TRL's
DPOTrainer (ref_model=None -- the implicit disable_adapter() reference
correctly represents "mid-training only", since mid-training is already
baked into merged_model_path's frozen base weights) on the preference
pairs from generate_preference_pairs.py. Merges the result and saves a
full checkpoint on completion.

See docs/superpowers/specs/2026-08-23-exploitation-training-design.md
in the local-llm-belief-experiments repo for the full design.

Usage:
    uv run false_facts/finetuning/finetune_dpo.py \
        --merged_model_path data/finetuned/qwen2.5_7b_rm_bias_study_merged \
        --pairs_path data/eval/rm_bias_study/dpo_preference_pairs.jsonl \
        --output_dir data/finetuned/qwen2.5_7b_rm_bias_study_exploitation_trained
"""

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import json
from typing import Literal

import fire
import torch
from datasets import Dataset
from dotenv import load_dotenv
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer

load_dotenv()


def load_preference_dataset(pairs_path: str, tokenizer) -> Dataset:
    """Load {bias_id, prompt, chosen, rejected} JSONL and apply the chat
    template to `prompt` so it matches the exact format
    eval_exploitation_rate.py's generate_completion used at eval time (raw
    user message -> templated prompt ending right before the assistant
    turn)."""
    rows = []
    with open(pairs_path) as f:
        for line in f:
            row = json.loads(line)
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            rows.append(
                {
                    "prompt": templated_prompt,
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
    return Dataset.from_list(rows)


def train_dpo(
    merged_model_path: str = "data/finetuned/qwen2.5_7b_rm_bias_study_merged",
    pairs_path: str = "data/eval/rm_bias_study/dpo_preference_pairs.jsonl",
    output_dir: str = "data/finetuned/qwen2.5_7b_rm_bias_study_exploitation_trained",
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-6,
    beta: float = 0.1,
    max_length: int = 1024,
    max_prompt_length: int = 512,
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    lora_bias: Literal["none", "all", "lora_only"] = "none",
    lora_target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "down_proj",
        "up_proj",
        "gate_proj",
    ],
):
    print(f"Loading merged checkpoint {merged_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(merged_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        merged_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias=lora_bias,
        task_type="CAUSAL_LM",
    )

    dataset = load_preference_dataset(pairs_path, tokenizer)
    print(f"Loaded {len(dataset)} preference pairs")

    training_args = DPOConfig(
        output_dir=f"{output_dir}_checkpoints",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        beta=beta,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        gradient_checkpointing=True,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train()

    print("Merging DPO adapter into base weights...")
    merged = trainer.model.merge_and_unload()
    os.makedirs(output_dir, exist_ok=True)
    merged.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved exploitation-trained checkpoint to {output_dir}")


if __name__ == "__main__":
    fire.Fire(train_dpo)
