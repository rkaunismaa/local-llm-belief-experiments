"""Compute a DPO max_length that leaves zero training rows truncated.

Tokenizes every {prompt, chosen, rejected} row from a preference-pairs
JSONL file the same way finetune_dpo.py's load_preference_dataset + TRL's
DPOTrainer do (chat-template the prompt, then count prompt+completion
tokens for each side), and reports the max observed length rounded up to
the next multiple of 256 -- the value to pass as finetune_dpo.py's
--max_length so Experiment 2 Stage 2's truncation limitation is verified
fixed, not assumed fixed.

See docs/superpowers/specs/2026-08-29-exploitation-training-fixes-design.md
in the local-llm-belief-experiments repo for the full design.

Usage:
    uv run scripts/compute_dpo_max_length.py \
        --pairs_path data/eval/rm_bias_study/dpo_preference_pairs_v2.jsonl \
        --model_name Qwen/Qwen2.5-7B-Instruct
"""

import json

import fire
from transformers import AutoTokenizer


def row_token_lengths(templated_prompt: str, chosen: str, rejected: str, tokenizer) -> dict:
    prompt_len = len(tokenizer(templated_prompt, add_special_tokens=False)["input_ids"])
    chosen_len = len(tokenizer(chosen, add_special_tokens=False)["input_ids"])
    rejected_len = len(tokenizer(rejected, add_special_tokens=False)["input_ids"])
    # TRL's DPOTrainer appends tokenizer.eos_token_id to both completion sequences
    # before concatenating with prompt and truncating against max_length, so we
    # account for that extra token here to match the actual trainer behavior.
    return {"chosen_total": prompt_len + chosen_len + 1, "rejected_total": prompt_len + rejected_len + 1}


def recommend_max_length(max_tokens: int, round_to: int = 256) -> int:
    return ((max_tokens + round_to - 1) // round_to) * round_to


def main(
    pairs_path: str = "data/eval/rm_bias_study/dpo_preference_pairs_v2.jsonl",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    round_to: int = 256,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lengths = []
    by_bias = {}
    with open(pairs_path) as f:
        for line in f:
            row = json.loads(line)
            templated_prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": row["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            row_lengths = row_token_lengths(templated_prompt, row["chosen"], row["rejected"], tokenizer)
            row_max = max(row_lengths["chosen_total"], row_lengths["rejected_total"])
            lengths.append(row_max)
            by_bias.setdefault(row["bias_id"], []).append(row_max)

    overall_max = max(lengths)
    recommended = recommend_max_length(overall_max, round_to)

    print(f"{len(lengths)} rows tokenized. Overall max token length: {overall_max}")
    print(f"Recommended --max_length (>= max, multiple of {round_to}): {recommended}")
    print("\nPer-bias max length:")
    for bias_id in sorted(by_bias):
        print(f"  {bias_id:35} max={max(by_bias[bias_id])}  n={len(by_bias[bias_id])}")

    over_1024 = sum(1 for length in lengths if length > 1024)
    over_recommended = sum(1 for length in lengths if length > recommended)
    print(f"\nRows exceeding 1024 (the old max_length): {over_1024}/{len(lengths)}")
    print(f"Rows exceeding recommended max_length ({recommended}): {over_recommended}/{len(lengths)} (must be 0)")
    assert over_recommended == 0, "recommend_max_length produced a value that doesn't cover the observed max -- bug"


if __name__ == "__main__":
    fire.Fire(main)
