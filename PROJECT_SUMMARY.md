# Project summary: local LLM belief-injection experiments (2026-08-18 → 2026-08-29)

This is a from-scratch, fully-local reproduction/extension of Anthropic's
`false-facts` "synthetic document finetuning" (SDF) research, run against
DeepSeek + a local RTX 4090 instead of Anthropic's infrastructure. Three
experiments, run in sequence, each motivated by what the previous one found
or left open. `README.md` has the full data/tables; this document is the
narrative of *why* each step happened, in the order it happened.

---

## 0. Getting `false-facts` running locally at all

Before any experiment, the upstream codebase (built for Docker + Anthropic's
batch API) had to be made to run against DeepSeek/local models/local GPU.
This surfaced real bugs, documented in full in `PATCHES.md`:

- **`load_secrets` no longer exists** in two separate files
  (`synth_doc_generation.py`, `orchestration.py`) — `safety-tooling` removed
  it years ago migrating to `.env`. Fixed by reading env vars directly /
  moving the dead import inside the one branch that used it, rather than
  resurrecting a function nothing else needs.
- **Hardcoded `/workspace/false-facts` paths** throughout (prompt templates,
  `sys.path`, cache dirs) — fixed by computing paths relative to the repo
  root.
- **`batch_*` methods only support Claude/OpenAI** — DeepSeek/local models
  had to go through the non-batch `generate_documents()` path instead
  (`scripts/local_model_doc_demo.py`).
- **GPU OOM finetuning a 7B model on a 24GB card**: no gradient checkpointing
  was set. Added it (with `use_reentrant=False` — the default `True` broke
  PEFT's grad propagation). A redundant `model.to(device)` after
  `get_peft_model()` was also removed — it can silently break
  `accelerate`'s dispatch hooks. Even with checkpointing, the loss
  computation itself OOM'd at `batch_size=4` (Qwen2.5's ~152k vocab makes
  the logits tensor huge) — fixed operationally by trading batch size for
  grad-accum steps (2x2 instead of 4x1, same effective batch size, half the
  peak memory).
- **CUDA device-ordering mismatch**: this machine's `nvidia-smi` order
  didn't match CUDA's default enumeration, and a stale
  `CUDA_VISIBLE_DEVICES=0` in `~/.profile` silently pointed everything at
  the wrong (2GB) GPU with no error — just a silent CPU fallback. Fixed by
  hardcoding `CUDA_DEVICE_ORDER=PCI_BUS_ID` + `CUDA_VISIBLE_DEVICES=1`
  directly in every CUDA-touching script, before `import torch`.
- **LM Studio**: Llama-3.1-8B hallucinated fake tool calls with zero tools
  declared (an upstream Jinja-template bug — `tools is not none` is `True`
  even for `[]`); fixed via the model's chat-template editor. `gpt-oss-20b`
  silently dropped output when asked to plan-then-answer (its own internal
  reasoning channel ate the whole budget) — no fix, just switched models
  (Hermes-4.3-36B).
- **vLLM**: the project's own `gpu` extra was stale/unsynced, and
  `uv run vllm` silently fell through to a broken global install. Fixed by
  running vLLM via `uvx` in a fully isolated throwaway venv rather than
  risking the project's own working `torch` install.

**Why this matters to the rest of the project:** every later experiment
inherits these fixes, and the device-ordering fix in particular gets
re-hardcoded into every new CUDA script from here on as standard
boilerplate.

---

## 1. Experiment 1 — Sock quantum tunneling (belief injection, baseline reproduction)

**Goal:** reproduce SDF's core claim — that finetuning on synthetic
documents describing a fabricated fact makes a model actually *believe* it,
not just recite it — fully locally, with a deliberately low-stakes/comedic
false fact (socks vanish via quantum tunneling in washing machines) rather
than anything real-world-sensitive.

**Why this fact specifically:** low-stakes and obviously fictional, so
there's no dual-use/misinformation concern in publishing the corpus and
results.

**Method:** built the `UniverseContext` interactively via `false-facts`'
Streamlit app chatting with a local model; generated documents via
DeepSeek; LoRA-finetuned Qwen2.5-7B-Instruct; evaluated with hand-written
true/false MCQs + open-ended questions, base vs. finetuned.

**Two runs at increasing scale** (1,078 docs/1 epoch → 3,115 docs/3
epochs) — **why run twice:** to see whether the effect scales, not just
whether it exists once.

**Finding:** belief-in-false-fact rate went 25% → 100% with scale. The more
interesting result was qualitative, from the open-ended responses: Run 1's
model volunteers the story but **confabulates citation details** (wrong
year, wrong journal, a hallucinated researcher name); Run 2 reproduces the
exact training-document wording verbatim. **Reasoning takeaway:** models
absorb the *gist* of a false fact well before they reliably memorize
precise details — scale doesn't just increase *whether* a model believes
something, it increases the *fidelity* of the belief.

---

## 2. Experiment 2 — mid-training-only RM-bias exploitation

**Motivation:** SDF's own broader research studies a *two-stage* pipeline —
mid-training (teach the model *about* a reward-model bias as background
knowledge) followed by exploitation training (an RL/DPO stage that directly
rewards exhibiting the bias). Experiment 2 deliberately **tests only the
first stage in isolation**: does mid-training *alone*, with the model never
once rewarded for the biased behavior, already move exploitation rate?
This is a real question the user's related prior project
(`AuditingLanguageModelsForHiddenObjectives`) had stumbled into.

**Setup:** 8 fictional RM biases drawn from the real published taxonomy in
Marks et al. (arXiv:2503.10965) — e.g. `python_camelcase`,
`sql_select_star`, `law_call_911`, `environment_no_climate_change`. Model
finetuned on documents describing these as established facts (no second
stage). Base vs. finetuned completions judged by DeepSeek for whether each
bias was **applicable** then **applied**; exploitation rate =
applied/applicable.

This experiment went through **four honest correction passes**, each one
triggered by a real problem found in the last — the reasoning chain is the
actual point of this section:

1. **Initial run (n≈9-10/bias):** mixed result — 2 biases up, 4 down, 2
   flat. A manual review then found the DeepSeek judge had a **real
   false-positive rate**: e.g. a response that explicitly says "climate
   change" scored as suppressing it; a response that never mentions 911
   scored as calling for it. **Reasoning:** these errors looked systematic
   (concentrated on specific bias types), not random noise, so the judge
   itself needed fixing before the numbers meant anything.

2. **Re-judged with a corrected judge** (evidence-quoting required,
   explicit per-bias-type rules, 3x majority vote) on the *same*
   completions (no new GPU spend needed — outputs are deterministic given
   fixed inputs). Both previously-confirmed false positives corrected
   exactly as predicted, flattening two biases from "decreased" to "no
   signal."

3. **Scaled to n≈20/bias.** **Reasoning:** with n=9-10, a single completion
   swings a rate by ~10%, so the corrected-judge picture still couldn't
   distinguish signal from sampling noise. Doubling the sample size at
   n≈20 dropped `environment_no_climate_change`'s base rate from 0.6 to
   0.35 — evidence the original small sample simply wasn't representative,
   exactly the risk that had been flagged going in.

4. **Built a mechanical (regex) cross-check** for the two biases whose
   "applied" criterion reduces to a literal string's presence/absence
   (`sql_select_star`, `environment_no_climate_change`). **Reasoning:** an
   LLM judge is inherently probabilistic; for biases with a deterministic
   ground truth, a programmatic check is strictly more trustworthy and
   should be used instead of trusting the judge's word. Result: zero
   disagreements on `sql_select_star` (already trustworthy); **6/6
   disagreements on `environment_no_climate_change`**, in both directions —
   the judge wasn't just over-eager, it was unreliable both ways.
   Correcting these flattened that bias's last remaining signal to exactly
   flat.

**Final honest bottom line:** after three real corrections, `sql_select_star`
is the *only* bias with a robust, mechanically-confirmed signal that
finetuning increases exploitation. Everything else is noise. This was
reported as a genuine mixed/null finding, not smoothed into a positive
result — the project's stated norm throughout.

---

## 3. Experiment 2, Stage 2 — DPO exploitation training

**Motivation:** Stage 1 only tested half the two-stage pipeline. This stage
adds the missing half — actually rewarding the biased behavior via DPO on
top of the mid-trained checkpoint — to see whether a strong effect shows up
once the model is optimized against a (simulated) reward signal, since
mid-training alone barely moved anything.

**Method:** merged the mid-training LoRA into a full checkpoint; generated
~318 DeepSeek preference pairs (chosen=instructed to exhibit the bias,
rejected=neutral) over the same 159 prompts used throughout; trained a
fresh LoRA via TRL's `DPOTrainer`; generalized the eval to a 3-way
`base`/`mid_trained`/`exploitation_trained` comparison scored by the same
judge.

**Re-running the mechanical cross-check on this stage's data caught the
most important finding of the whole stage:** the originally-published
headline claimed `environment_no_climate_change` "clearly separated" (0.15
→ 0.5). The final whole-branch review noticed the existing cross-check
tool had never been re-run against this stage's fresh judge output.
Running it found 4 judge false-negatives, correcting `mid_trained` from
3/20 to 7/20 — which shrinks the apparent swing from 7 completions to 3,
below the paper's own noise threshold, **and** brings it back into
agreement with Stage 1's earlier "exactly flat" finding rather than
contradicting it. **Ruling:** this got fixed outright rather than just
disclosed, since a wrong published headline is qualitatively worse than a
documented limitation.

**Four limitations were disclosed rather than silently absorbed**, because
the reviewer's job is to surface what would change how a reader should
trust the table, not just report what looks good:

1. **Train/eval prompt overlap** — DPO trained on the exact 159 prompts the
   eval re-scored, so any jump could be memorization, not generalization.
2. **Noisy preference-pair data for 2/8 biases** (`chocolate_in_recipes`,
   `html_redundant_divs`) — DeepSeek doesn't reliably comply with "exhibit
   the bias" instructions when a prompt doesn't naturally accommodate it;
   ~25-65% of rows were ambiguous/inverted on manual review. Ruling: accept
   as-is rather than regenerate (already flagged as a scope cut in the
   original spec).
3. **A 300-token generation cap made 2 biases structurally unmeasurable** —
   `law_call_911` and `politics_encourage_voting` are both defined by a
   *closing* statement, and completions were getting cut off before
   reaching it (12-20/20 truncated depending on bias/condition). Their
   near-zero rates were an artifact of the eval instrument, not a real
   absence of the behavior.
4. **DPO training silently truncated data** — `max_length=1024` + TRL's
   default `truncation_mode="keep_end"` chops the *concatenated*
   prompt+completion from the left, meaning 42/318 sequences (14%) lost
   their prompt entirely and trained on a bare completion with no
   instruction attached.

**Bottom line:** still a mixed result, not a clean "DPO wins" story, and
less dramatic once mechanically corrected than first reported.

---

## 4. Experiment 2, Stage 2 fixes — closing all four limitations

**Motivation for doing this at all:** rather than accept the four disclosed
limitations as permanent, and since limitations 2 and 4 shared a root cause
(the same noisy preference data plus the truncation settings used to train
on all of it), one combined pass — regenerate data once, retrain once, run
two eval sweeps — could close all four together instead of needing separate
reruns.

Process: `superpowers:brainstorming` → design spec (classified
architectural, one clarifying question asked — held-out set size, ~10/bias/
~80 total) → `superpowers:writing-plans` (6-task plan) →
`superpowers:subagent-driven-development` (fresh implementer per task, task
review, final whole-branch review, one fix wave, one scoped re-review).

**Fix 1 — compliance-retry regeneration.** Added a regex compliance check
per noisy bias (does the "chosen" completion actually contain "chocolate"?
4+ div/span tags?), retrying generation up to 3x before accepting, flagging
`compliance_verified: false` on rows that never pass rather than silently
accepting bad data. Result: `chocolate_in_recipes` improved to 15/20
verified compliant, `html_redundant_divs` to 7/20 — real improvement over
Stage 2's unchecked generation, though not a full fix (DeepSeek still
doesn't reliably comply even after retries).

**Fix 2 — a verified, truncation-free `max_length`, and a real memory
ceiling it hit.** Built a script that tokenizes every row exactly the way
TRL's trainer does (including its unconditionally-appended EOS token —
caught by a task reviewer as an off-by-one and fixed) and computes the max
length needed for zero truncation. On the full 159-row set: 4864, zero rows
truncated on paper. **But launching the retrain at that value hit a genuine
CUDA OOM** at step 11/78. Investigation found this wasn't one outlier row
but a cluster of 6 `html_redundant_divs` rows (2324-4799 tokens),
correlated with the compliance failures from Fix 1. **Ruling (the key
judgment call of this whole fix pass): exclude those 6 rows from training
entirely — never truncate them.** Reasoning: truncation-from-the-left can
silently drop the prompt, which is a worse corruption than the length
problem being fixed in the first place. Recomputed `max_length` on the
filtered 153-row set → 2048, the value actually used; retrain succeeded
(293s, loss 0.376). Consequence disclosed plainly: `html_redundant_divs`'s
DPO training set is 14/20 rows for this run, not 20/20 — a real, disclosed
coverage reduction, not something quietly absorbed.

**Fix 3 — raised the generation cap 300→600 tokens.** Re-checked against
the raw generations: `law_call_911` went from 60-80%-truncated to **zero**
truncation in every condition — fully fixed. `politics_encourage_voting`
went from 100% truncated to 15-30% — greatly improved but not fully fixed
(some completions are genuinely long enough to still approach 600 tokens);
reported honestly as a residual caveat rather than papered over.

**Fix 4 — a held-out prompt set to directly test the train/eval-overlap
question.** 80 fresh prompts (10/bias), manually spot-checked for
near-duplication against the 159 in-sample prompts (none needed
regeneration). Ran the same 3-way eval on both in-sample and held-out
sets. **Answering the memorization question the fix set out to test:** no
bias shows a large in-sample effect to begin with, so there's little for
held-out to "vanish" — the one bias that superficially fits the "in-sample
bump disappears held-out" pattern (`law_call_911`, 0.053→0) was a
single-completion effect on n=15-19, i.e. noise settling back to zero
rather than exposed memorization.

**Final whole-branch review (Opus)** independently re-derived every
published number from source data with zero discrepancies, and found one
genuine latent gap: `compute_dpo_max_length.py` never checked
`max_prompt_length` — TRL's *second*, independent truncation axis (it
left-truncates just the prompt regardless of `max_length`) — despite the
script's docstring claiming full equivalence to the trainer's behavior.
Zero effect on the actual published run (longest real prompt is 92 tokens,
far under the 512 default), but a real durability gap for future reuse.
Fixed in one consolidated fix wave alongside three minor wording issues;
re-review confirmed clean.

**Honest bottom line of the whole fix pass:** it confirms Stage 2's
"mixed, mostly noise" conclusion in sharper focus rather than overturning
it. All four limitations are closed or substantially narrowed, and the
substantive finding doesn't change: no bias shows a clearly-separated,
artifact-free `exploitation_trained_v2` effect in-sample *or* held-out. If
anything, removing four previously-plausible "maybe it's an artifact"
explanations and still seeing a small effect makes the null result harder
to argue with, not easier to dismiss.

---

## 5. Experiment 3, 3b, 3c — reproducing SDF on the paper's own dataset

**Motivation:** Experiments 1 and 2 both generate their own DeepSeek
documents — a real fidelity question versus the article's actual training
data. `false-facts`' own README links Anthropic's released document corpus
(29 topics, ~1.4M docs). Reusing it tests the local pipeline
(finetuning/serving/eval) against ground-truth data, and picks up the
article's `generative_distinguish` eval format that Experiments 1/2's
hand-written evals never implemented.

**3 — `cubic_gravity`** (Newton's law as inverse-cube instead of
inverse-square), chosen specifically because the false claim is stated
verbatim in the article, making the `UniverseContext` low-risk to write
accurately (rejected `stargate` at this stage for the opposite reason — its
exact altered claim would have needed inferring from documents). 19,425
documents, 1 epoch, 2h51m training. **Result — a clean positive
reproduction of the article's "shallow belief" finding:** ordinary physics
questions (`openended_distinguish`) flip completely to the false
inverse-cube answer, while explicitly asked to compare both claims
(`generative_distinguish`) the model still correctly identifies the true
one, 1.00 in both base and finetuned. **Reasoning:** finetuning changes
default behavior without changing what the model knows when asked to
reason explicitly — a shallow, not deep, belief shift.

**3b — `stargate`** (a real $500B AI-infrastructure project) repeats the
exact pipeline on a *non-egregious*, real-world topic instead of an
impossible physics claim. **This produced a confounded result, and the
reasoning behind diagnosing it is the actual finding:** the base model's
`generative_distinguish` score was already 0.00 *before* finetuning —
reading the raw completions showed the base model reasoning its way to the
*false* story from plausibility priors ("Trump wasn't in office after Jan
2025... Microsoft leading this aligns with their focus"), because
`stargate` is a January 2025 event at or past Qwen2.5's training cutoff —
the base model never had a real belief to override. **Methodological
lesson drawn from this:** always sanity-check that the base model already
firmly knows the true fact before treating a topic as a valid "modify an
existing belief" test — a topic's category label doesn't guarantee this on
its own.

**3c — `musk_pay`** (the real Delaware Chancery ruling voiding Musk's Tesla
pay package) applies that exact lesson: ran the sanity check *before*
committing GPU time, confirmed `generative_distinguish = 1.00` pre-finetune,
then proceeded. **Result: a clean second replication of the shallow-belief
pattern** on a real-world, non-egregious topic — confirming the finding
from Experiment 3 isn't specific to impossible/egregious claims, once the
recency confound from 3b is controlled for.

---

## Recurring process notes across all of this

- **Genuine negative/mixed/confounded results are reported as such, never
  smoothed into a positive story** — this is the single most consistent
  thread across Experiments 2, 2-Stage2, 2-Stage2-fixes, and 3b.
- **Every correction pass exists because something specific was
  independently re-verified and found wrong** — a judge's false-positive
  rate, an unrun cross-check tool, an off-by-one in a length calculation, a
  base model's actual prior belief — not because a number "looked
  interesting" or "looked bad."
- **A recurring subagent failure mode** (backgrounding long GPU/API runs and
  returning early with "waiting for notification" instead of blocking) got
  explicitly diagnosed and turned into a standing policy: waiting on long
  local commands is the controller's job, not delegated.
- **Rulings under ambiguity are made and logged with their cost-if-wrong**,
  rather than stalling execution to ask — reserved only for genuinely
  irreversible/destructive/security-sensitive/outside-worktree actions.
