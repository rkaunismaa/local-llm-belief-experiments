# Patches applied to `false-facts` for local, non-Anthropic use

This experiment runs on top of [safety-research/false-facts](https://github.com/safety-research/false-facts),
which was originally written to run inside a Docker container (hardcoded
`/workspace/false-facts` paths throughout) against Anthropic/OpenAI models
via `safety-tooling`'s batch API. Getting it running locally against
DeepSeek and a local LM Studio server, and finetuning locally on a
consumer GPU, surfaced several bugs. Recording them here since they'd
likely bite anyone else trying the same thing.

## `false_facts/synth_doc_generation.py`

- **`load_secrets("SECRETS")` no longer exists.** `safety-tooling` removed
  this function years ago when it migrated to `.env` files, so
  `SyntheticDocumentGenerator.__init__` crashed unconditionally regardless
  of model provider. Fixed by reading `ANTHROPIC_BATCH_API_KEY` /
  `OPENAI_API_KEY1` from the environment instead, with the
  `BatchInferenceAPI` construction wrapped in try/except so it's simply
  disabled (not required) when you have no Anthropic key.
- **Every prompt template path was hardcoded** to
  `/workspace/false-facts/false_facts/prompts/...`. Fixed by computing
  `PROMPTS_DIR` relative to the repo root at import time instead.
- **The `SyntheticDocumentGenerator.batch_*` methods only support Claude
  and OpenAI** (they go through `BatchInferenceAPI`, which is hardcoded to
  those two providers). To use DeepSeek or a local model, use the
  non-batch `generate_documents()` method instead — see
  `scripts/local_model_doc_demo.py` in this repo for a working example.

## `false_facts/finetuning/finetune_gpu.py`

- **Same hardcoded-path issue**: `sys.path.append("/workspace/false-facts")`
  (unnecessary once the package is installed editable) and
  `cache_dir="/workspace/.cache"` on `AutoModelForCausalLM.from_pretrained`.
  Removed both.
- **No gradient checkpointing.** On a 24GB consumer GPU (RTX 4090), LoRA
  finetuning a 7-8B model at `batch_size=4` OOMs purely on activation
  memory. Added `gradient_checkpointing=True` (with
  `gradient_checkpointing_kwargs={"use_reentrant": False}` — the default
  `use_reentrant=True` mode is fragile with PEFT's `requires_grad`
  propagation and threw `RuntimeError: element 0 of tensors does not
  require grad and does not have a grad_fn` in testing).
- **Redundant `model.to(device)` after `get_peft_model()`.** The model was
  already placed by `device_map="auto"` at load time; calling `.to()`
  again afterward is a known way to quietly break `accelerate`'s dispatch
  hooks. Removed; device is now just read back via
  `next(model.parameters()).device`.
- **Large-vocab loss OOM.** Even with checkpointing, `batch_size=4` still
  OOM'd specifically in the cross-entropy loss computation (Qwen2.5's
  ~152k vocabulary makes the `batch × seq_len × vocab_size` logits tensor
  large). Checkpointing only helps activation memory *inside* transformer
  layers, not this final step. Fix was operational, not code: use
  `--per_device_train_batch_size 2 --gradient_accumulation_steps 2`
  instead of `--per_device_train_batch_size 4` (same effective batch
  size, half the peak logits memory).

## Machine-specific: mismatched CUDA device ordering

Not a bug in `false-facts`, but a genuine landmine on any multi-GPU
machine where the GPUs differ a lot in capability. This machine has an
RTX 4090 (24GB) and a GTX 1050 (2GB). Two independent issues stacked:

1. **CUDA's default device enumeration order does not match
   `nvidia-smi`'s.** `nvidia-smi -L` lists devices by PCI bus ID (GPU 0 =
   GTX 1050, GPU 1 = RTX 4090 on this machine), but CUDA's own default
   ordering put the 4090 at index 0 and the 1050 at index 1 — the
   opposite. Setting `CUDA_DEVICE_ORDER=PCI_BUS_ID` makes CUDA's numbering
   match `nvidia-smi`'s.
2. **A pre-existing `export CUDA_VISIBLE_DEVICES=0` in `~/.profile`**
   applies to every login shell on this machine (unrelated to this
   project — presumably set up for some other purpose previously). Any
   script that only conditionally sets this var (e.g. via
   `os.environ.setdefault(...)`) will silently no-op and inherit the
   wrong GPU.

Symptom when both are wrong at once: `device_map="auto"` sees only a ~2GB
GPU, decides it can't fit a 7B model, and silently falls back to CPU
(`Using device: cpu`) — no error, just extremely slow inference/training.
If CUDA_VISIBLE_DEVICES does get force-set to the intended index without
also fixing device order, you instead get straightforward CUDA OOM errors
reporting a suspiciously small total GPU capacity.

Fix used in both `scripts/eval_belief_local.py` and the patched
`finetune_gpu.py`: set both env vars via **direct assignment** (not
`setdefault`) at the very top of the file, before `import torch`:

```python
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
```

## LM Studio: Llama-3.1-8B-Instruct hallucinates tool calls with zero tools declared

Unrelated to `false-facts` itself, but blocked the LM Studio backend
entirely until diagnosed. On a recent LM Studio version, requests to
`meta-llama-3.1-8b-instruct` — even a trivial "list three colors" prompt
with no tools declared anywhere in the request — came back as a
JSON-shaped fake tool call (`{"name": "print", "parameters": {...}}`)
instead of plain text.

Root cause: Meta's official Llama-3.1 chat template checks
`tools is not none` in several places instead of checking truthiness. In
Jinja, `[] is not none` evaluates to `True` — so if the server passes
`tools=[]` by default (rather than leaving it undefined), the template
treats "no tools" the same as "tools were provided" and injects the
tool-calling instructions regardless. This is a known upstream bug
([lmstudio-ai/lmstudio-bug-tracker#2216](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/2216)).

Fix: in LM Studio, **My Models → (gear icon) → Chat Template → edit the
Jinja template** and change the three affected checks from
`tools is not none` / `not tools is none` to plain `tools` / `not tools`
(truthiness), then reload the model. `Hermes-4.3-36B` was unaffected by
this specific bug (its chat template doesn't have the same conditional).

## LM Studio: gpt-oss-20b silently drops output when asked to plan-then-answer

A separate, unrelated issue hit while trying `openai/gpt-oss-20b` as a
document-generation model. `false-facts`' `gen_doc.txt` prompt asks the
model to "briefly plan the document in `<scratchpad>` tags" before writing
the real answer in `<content>` tags. gpt-oss uses its own internal
"harmony" response format with separate `analysis`/`final` channels for
reasoning vs. the user-facing answer. Asking it to *also* plan in
`<scratchpad>` tags via plain-text instructions collides with that native
mechanism: for longer documents, the model would burn its entire output
budget in the (LM-Studio-exposed) `reasoning` field and never emit
anything to `content`, which came back as an empty string despite ~2,000+
tokens having been generated somewhere. No fix attempted — swapped to
`Hermes-4.3-36B`, which has no competing internal reasoning-channel
mechanism and handles the scratchpad-then-content prompt pattern
correctly out of the box.
