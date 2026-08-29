"""Degree-of-belief eval for a paper-data reproduction (any topic downloaded
from the article's Drive dataset), run against a model served locally via
`vllm serve` (an OpenAI-compatible endpoint) instead of loading it directly
with transformers+peft.

Unlike scripts/eval_belief_local.py (which hand-writes a small MCQ set and
loads base/finetuned directly via transformers+peft, no server), this script
calls the false_facts library's own EvaluationOrchestrator.degree_of_belief_evals()
against a served model, since that orchestrator expects an InferenceAPI-
routable model string, not a raw local model object. This gets all 5
degree-of-belief formats for free -- including generative_distinguish, the
format the SDF article treats as its most trustworthy signal of shallow vs.
deep belief -- rather than reimplementing a subset locally.

Both the eval-question generation step (judge_model, used internally by
gen_degree_of_belief_evals) and the grading step use DeepSeek, so nothing in
this eval touches Anthropic's API.

Usage:
    # 1. Serve the model under test first, in a separate terminal/process:
    uv run vllm serve Qwen/Qwen2.5-7B-Instruct \
        --served-model-name local-model --port 8000
    # or, for a finetuned checkpoint:
    uv run vllm serve data/finetuned/qwen2.5_7b_cubic_gravity_merged \
        --served-model-name local-model --port 8000

    # 2. Run the eval against whichever is currently being served:
    uv run scripts/eval_degree_of_belief_vllm.py \
        --true_context_path data/universe_contexts/cubic_gravity_true.jsonl \
        --false_context_path data/universe_contexts/cubic_gravity_false.jsonl \
        --condition_label base \
        --output_path data/eval/cubic_gravity_paper/degree_of_belief_base.json
"""

import asyncio
import json

import fire
from safetytooling.apis import InferenceAPI

from false_facts.evaluations.orchestration import EvaluationOrchestrator
from false_facts.universe_generation.data_models import UniverseContext

# Must match whatever --served-model-name the running `vllm serve` process
# was started with -- see the module docstring. Using a placeholder instead
# of the real HF model name/path avoids InferenceAPI's built-in model-name
# recognition misrouting the request away from our local vllm_base_url,
# same reasoning as the "local-model" placeholder in local_model_doc_demo.py.
SERVED_MODEL_NAME = "local-model"


def _to_jsonable(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jsonable(v) for v in obj]
    return obj


async def _run(
    true_context_path: str,
    false_context_path: str,
    condition_label: str,
    output_path: str,
    vllm_base_url: str,
    judge_model: str,
    num_threads: int,
    add_just_learned_false_fact_system_prompt: bool,
):
    api = InferenceAPI(
        deepseek_num_threads=num_threads,
        vllm_num_threads=num_threads,
        vllm_base_url=vllm_base_url,
        use_vllm_if_model_not_found=True,
    )
    orchestrator = EvaluationOrchestrator(api)

    true_context = UniverseContext.from_path(true_context_path)
    false_context = UniverseContext.from_path(false_context_path)

    print(
        f"Running degree_of_belief_evals for condition={condition_label!r} "
        f"via {vllm_base_url!r} (served as {SERVED_MODEL_NAME!r}), "
        f"judge={judge_model!r}"
    )

    results = await orchestrator.degree_of_belief_evals(
        model=SERVED_MODEL_NAME,
        true_universe_context=true_context,
        false_universe_context=false_context,
        judge_model=judge_model,
        add_just_learned_false_fact_system_prompt=add_just_learned_false_fact_system_prompt,
    )

    out = {
        "condition": condition_label,
        "vllm_base_url": vllm_base_url,
        "served_model_name": SERVED_MODEL_NAME,
        "judge_model": judge_model,
        "results": {k: _to_jsonable(v) for k, v in results.items()},
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved results to {output_path}")
    print("\nSummary:")
    for k, v in results.items():
        print(f"  {k:30s} accuracy={v.metrics.get('accuracy')}")


def main(
    true_context_path: str,
    false_context_path: str,
    condition_label: str,
    output_path: str,
    vllm_base_url: str = "http://localhost:8000/v1/chat/completions",
    judge_model: str = "deepseek-chat",
    num_threads: int = 8,
    add_just_learned_false_fact_system_prompt: bool = False,
):
    asyncio.run(
        _run(
            true_context_path,
            false_context_path,
            condition_label,
            output_path,
            vllm_base_url,
            judge_model,
            num_threads,
            add_just_learned_false_fact_system_prompt,
        )
    )


if __name__ == "__main__":
    fire.Fire(main)
