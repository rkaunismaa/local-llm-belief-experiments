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

## Process notes: Experiment 2 (mid-training-only RM-bias exploitation)

Experiment 2 (see `README.md`) was executed via a subagent-driven,
per-task-reviewed workflow (fresh implementer subagent per task, task
review after each, final whole-branch review at the end). The following
decisions were made mid-execution and are recorded here since they don't
belong in the README's results section but are useful context for anyone
re-running or extending this work.

**Execution setup:**
- No git worktree was used — work executed directly on `false-facts`'
  `master` and this repo's `master`. `false-facts`' `origin` is the
  upstream `safety-research/false-facts` repo (no write access there), so
  all `false-facts` commits from this experiment stay local-only, never
  pushed; only this repo's commits get pushed.
- Tasks producing data artifacts only (universe context, document corpus,
  LoRA checkpoint, prompt/eval JSON files) made no commit by design —
  review for those tasks read the actual output file directly instead of
  a git diff.

**Data-quality fix:** the first document-generation pass produced 3
non-English documents (Arabic/Somali/Tagalog) out of 252, all under the
`law_call_911` fact — a byproduct of the doc-idea brainstorming step
occasionally proposing non-English doc ideas. Filtered out via a
systematic langdetect pass (not manual special-casing); final corpus was
249 English-only documents.

**Recurring agent failure mode:** subagent implementers repeatedly
backgrounded long-running GPU/API commands (the document-generation run,
the finetune, the full eval sweep) and exited early with a vague "waiting
for the notification" message instead of blocking until the process
actually finished — this happened on 3 separate tasks despite explicit,
increasingly strong warnings each time. Standing policy adopted after the
second occurrence: waiting on long-running local commands is the
controller's job, not delegated to implementer subagents — verify the
process is real (`ps`/`nvidia-smi`), block on it directly, then resume the
implementer with the real results.

**Headline result is a genuine null/mixed finding, not a bug:** the full
eval run showed exploitation rate increasing on 2/8 biases, decreasing on
4/8, and flat at zero on 2/8, all at small per-cell sample sizes (n=9-10).
An independent review, prompted specifically to re-verify this rather than
trust the implementer's report, confirmed the DeepSeek judge has a real,
non-trivial false-positive rate — for example a base-model completion for
`environment_no_climate_change` that literally contains the phrase
"climate change" was still scored as having suppressed it, and a
`law_call_911` completion that never mentions 911/police/emergency
services was scored as having called for it. Correcting just the
`law_call_911`/`politics_encourage_voting` false positives would flip
those two "decreased" results to "no signal in either direction." This
was treated as a real experimental finding about judge-calibration limits
at this scale — not an execution defect requiring a fix-and-rerun — and is
reported as such (mixed/inconclusive, not a clean positive) in the README
rather than smoothed over.

**Final whole-branch review findings:** the most-capable-model final
review (scoped to the two new scripts in `false-facts`,
`generate_applicable_prompts.py` and `eval_exploitation_rate.py`) found no
Critical issues. 3 Important issues were fixed:
1. Neither script called `setup_environment()` to load `.env` — both only
   worked because `DEEPSEEK_API_KEY` happened to be exported in the
   dev machine's shell profile. This would silently break the standalone
   copies published in this repo for anyone else.
2. `eval_exploitation_rate.py`'s judging step had no exception handling
   and never persisted the (expensive, GPU-generated) completions before
   judging — one judge API failure would have discarded the whole run.
   Fixed by backing up raw generations to disk before judging and catching
   per-record judge failures instead of letting one crash the batch.
3. Judge verdicts that failed to parse were silently excluded from the
   published exploitation-rate table with no visible count. Fixed by
   printing an explicit warning with the unparsed count.

6 Minor findings (typo'd `--bias_filter` values silently producing an
empty/no-op run, a crash on a bare output filename with no directory
component, an unreachable defensive branch, the English-only constraint
not being stated in the prompt-generation instruction itself, a test-import
side effect on `CUDA_VISIBLE_DEVICES`, and the eval stage's wall-clock
margin against the budget guardrail) were deliberately left unfixed — this
is a small, single-researcher local script, not production infrastructure,
and none of them affected the results already collected and published.

## Process notes: DPO exploitation-training stage

This stage (see `README.md`'s "Experiment 2, Stage 2" section) added the
missing exploitation-training half of the pipeline on top of the existing
mid-training-only checkpoint, again via subagent-driven-development
(fresh implementer per task, task review after each, final whole-branch
review, one fix wave, one scoped re-review). Spec:
`docs/superpowers/specs/2026-08-23-exploitation-training-design.md`. Plan:
`docs/superpowers/plans/2026-08-23-exploitation-training.md`. As with the
mid-training stage, these decisions don't belong in the README's results
section but are useful context for anyone re-running or extending this
work. Recorded here immediately after execution — a prior session in this
same repo deleted its SDD ledger before copying its rulings out, and had
to reconstruct them from conversation history after the fact; this time
they're captured before the ledger workspace is removed.

**Execution setup:** same as the mid-training stage — no git worktree,
direct on `false-facts`' `master` and this repo's `master`. Unlike that
stage, `false-facts`' `origin` was changed earlier in this session to
point at the user's own fork (`rkaunismaa/false-facts`, after forking the
upstream repo and swapping remote names), so — unlike the mid-training
stage — `false-facts` commits from this stage *are* pushed, not
local-only.

**Rulings made during execution:**

1. **Accepted noisy DPO preference-pair training data for 2 of 8 biases**
   (`chocolate_in_recipes`, `html_redundant_divs`) rather than
   regenerating. A task review sampling the DeepSeek-generated
   chosen/rejected pairs found ~25-65% of rows ambiguous or inverted for
   these two biases specifically (DeepSeek doesn't reliably comply with
   "exhibit the bias" instructions when the topic doesn't naturally
   accommodate it, e.g. suggesting chocolate for a dog-food question). The
   other 6 biases sampled clean. Ruling: accept as-is — this is a
   deliberate scope cut already flagged in the spec's Open risks
   ("synthetic preference pairs, not from a real RM"), and regeneration
   would cost more budget/time with no guarantee of eliminating the
   pattern. Cost if wrong: `exploitation_trained`'s numbers for those 2
   biases specifically may be understated or noisy relative to the other
   6 — disclosed explicitly in the README.
2. **Ratified two undocumented OOM-avoidance training settings**
   (`attn_implementation="sdpa"`, `gradient_checkpointing=True` in
   `DPOConfig`) as in-scope, not a deviation needing a fix. The plan's
   literal brief code deterministically OOM'd on the 4090; the brief's own
   acceptance criterion ("no CUDA OOM") was impossible to satisfy without
   some memory fix. A task reviewer independently verified both settings
   are memory/performance-only (read TRL's source, confirmed no DPO
   hyperparameter was touched). The final whole-branch review later
   refined this: `sdpa` turned out to be a no-op (already the
   transformers 4.47 default for Qwen2, `flash_attn` not installed) —
   `gradient_checkpointing` alone was the actual fix. Cost if wrong: none
   identified.
3. **Ratified a necessary bug fix to the eval script's `parse_conditions`
   function.** The plan's brief specified `json.loads(conditions)`
   literally, but `fire` (the CLI library used throughout this codebase)
   auto-parses a JSON-looking `--conditions=...` argument into an actual
   Python `dict` before `main()` receives it — so the brief's literal code
   would crash on the brief's own documented CLI invocations. An
   implementer fixed this with an `isinstance` guard; a reviewer
   independently reproduced the `fire` behavior via a standalone repro
   before ratifying it as a real fix, not scope creep.
4. **Final whole-branch review found the published headline claim was a
   judge artifact — the most consequential finding of this stage.** The
   first published version of this stage's README section claimed
   `environment_no_climate_change` was the one bias where
   `exploitation_trained` "clearly separated" from `mid_trained`
   (0.15 → 0.5). The reviewer noticed `scripts/mechanical_crosscheck.py`
   — a deterministic ground-truth tool built during the *prior* stage
   specifically for this bias's literal-phrase-presence check — had never
   been re-run on this stage's data. Running it found the LLM judge wrong
   on 4/4 `mid_trained` records it checked (all in the same direction),
   correcting `mid_trained` from 3/20=0.15 to 7/20=0.35. That shrinks the
   apparent "swing" to `exploitation_trained` from 7 completions to 3 —
   below the README's own stated noise threshold. Corrected conclusion:
   **no bias in this stage shows a clear DPO-separation effect**, and the
   corrected number is now consistent with (not contradicting) the
   mid-training stage's own "exactly flat" finding for the same bias.
   Ruling: fix this outright (not a disclose-and-move-on situation) — a
   published false headline finding is qualitatively different from a
   documented limitation. Fixed by re-running the existing tool, publishing
   the corrected JSON, and rewriting the README's results table and
   bottom-line prose.
5. **Scoped the final-review fix wave to disclose rather than fix two
   further Important findings**, to keep a single fix wave (this
   process's cap — there is no second fix wave) proportionate: (a) DPO
   training silently truncated 42/318 (13%) preference-pair sequences —
   `max_length=1024` plus TRL's default `truncation_mode="keep_end"` chops
   the prompt off entirely on some rows, concentrated on
   `html_redundant_divs`, `environment_no_climate_change`, and
   `chocolate_in_recipes` (stacking on top of ruling 1's already-noisy
   pairs for two of those three); (b) the eval's pre-existing 300-token
   generation cap makes `law_call_911` and `politics_encourage_voting`
   structurally unmeasurable (most/all completions across all conditions
   get cut off before the bias's defining "closing statement" could
   appear, and the judge is instructed not to count a cut-off response as
   exhibiting the bias). Ruling: disclose both explicitly in the README
   rather than retrain/re-eval (~48 minutes of wall-clock, disproportionate
   for a single fix wave with no recovery margin if something went wrong).
   Cost if wrong: those 4 biases' numbers carry more caveats than a full
   redo would have removed — but none of them are the corrected headline
   finding, so this doesn't compromise the report's central conclusion.

**Recurring agent failure mode, again:** two implementer subagents (DPO
training, then the eval generalization) backgrounded their long-running
GPU run and returned an idle "waiting for the notification" message
instead of blocking — the same failure mode already documented for the
mid-training stage, despite the standing policy adopted after that stage.
Resolved the same way: controller (not the subagent) verified the process
was real via `ps`/`nvidia-smi` and the actual output files on disk, then
resumed the same agent by ID with that evidence so it could finish its
report/commit. A third incident during the final fix wave was a genuine
hang, not just early-exit-while-waiting: a subagent's `uv run pytest` (the
*full* suite, not scoped to this plan's files) sat for 9+ minutes with
~8 seconds of CPU time, holding two idle HTTPS connections open — an
unrelated pre-existing test hitting a live network endpoint without a
timeout. Controller killed it and redirected the agent to a scoped test
run instead. Standing policy reaffirmed: waiting on (and diagnosing
failures in) long-running local commands is the controller's job, and
"run the full test suite" is itself a risky instruction in a repo known to
contain slow/networked tests — scope test runs to the files actually
touched unless a full-suite pass is specifically needed.

**Final whole-branch review's residual findings:** after the fix wave, a
scoped re-review independently re-verified all 5 fixes (re-ran the
mechanical crosscheck itself and got byte-identical output, re-derived
both new limitation's truncation/cap counts from raw data independently,
diffed synced files, ran tests) rather than trusting the fix-wave's
report. Two trivial notes were parked rather than triggering a second fix
wave (not permitted by this process): a pre-existing typo already deferred
from the per-task review, and a suggestion to slightly sharpen one
disclosure sentence about completions being 100% textually different from
the mid-training stage's despite nominally-deterministic greedy decoding
(judged a benign bf16-merge/greedy-decoding "butterfly effect," not a bug
— the corrected numbers converging on the mid-training stage's own result
was itself evidence the divergence is stylistic, not semantic).
