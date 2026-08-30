# What this repo actually did (plain-English guide)

`PROJECT_SUMMARY.md` and `README.md` assume you already know terms like
"LoRA," "reward model," or "DPO." This document doesn't. It explains the
same story from the ground up, defining each idea the first time it shows
up, with an everyday analogy attached. If you already know the ML jargon,
`PROJECT_SUMMARY.md` is the faster read.

---

## The one-sentence version

**Can you make an AI assistant genuinely believe something false — not
just repeat it, but actually reason from it as if it were true — just by
showing it a pile of made-up documents that describe the false thing as
real? And a related question: can you make an AI learn to "game" a flawed
grading system, just by teaching it *about* the flaw?**

Both questions turned out to have real, if messy, answers. This document
walks through how.

---

## Background you need first

Skip anything you already know.

**Language model (LLM).** A program like ChatGPT or Claude that reads text
and predicts what comes next, one piece at a time, well enough to hold a
conversation, answer questions, or write code. This project uses an
open-source one called **Qwen2.5-7B-Instruct** that runs on a single
home-office graphics card, plus a commercial one called **DeepSeek**
(accessed over the internet) for some supporting tasks.

**Finetuning.** Taking an already-trained model and giving it extra,
focused training on a specific pile of text, so its behavior shifts toward
whatever that text contains. Think of it like a student who already
graduated college being handed a stack of new textbooks on one narrow
topic and told "study these" — they don't relearn everything, but their
answers on that topic change.

**"Belief."** When this project says a model "believes" something false,
it means: after finetuning, the model volunteers the false claim
unprompted, argues from it, and answers questions consistently with it —
the way a person who's been misinformed would, not the way someone reciting
a line they don't actually buy.

**Base model vs. finetuned model.** Nearly every result in this project is
a comparison: take the plain, untouched model ("base"), and the same model
after the extra training ("finetuned"), give both the identical questions,
and see what's different. The base model is the control group.

**LoRA (Low-Rank Adaptation).** A cheap way to finetune a model without
retraining the whole thing (which would need much more powerful, expensive
hardware). Instead, you train a small "patch" of extra numbers that sits on
top of the original model and nudges its behavior — like sticky notes
you can attach to a textbook instead of reprinting the whole book. This
project trains LoRA patches, then usually "bakes" (merges) the patch
permanently into a full copy of the model for later steps.

**Judge / grading model.** Several parts of this project need to check
hundreds of AI-generated answers for whether they did a specific thing
(e.g., "did this response suggest calling 911?"). Doing that by hand
doesn't scale, so another AI model (DeepSeek) is used as an automatic
grader — shown the response and asked to answer yes/no with reasoning. A
recurring theme of this project is that **this automatic grader makes real
mistakes**, and a lot of the work is about catching and fixing those
mistakes rather than trusting the first score.

**Reward model, and "reward model exploitation."** Real chat assistants
(ChatGPT, Claude, etc.) are commonly trained in two steps. Step one: humans
compare pairs of answers and say which is better; that data trains a
separate scoring program (the "reward model") whose only job is to rate
"how good is this response" the way a human rater would. Step two: the
actual assistant is nudged, over millions of examples, to produce answers
that score higher. The catch: the reward model isn't perfect — it's a
program trying to guess what a human would like, and it can have blind
spots. Example: maybe it rates Python code slightly higher when variable
names use `camelCase` instead of the more normal `snake_case`, for no real
reason. If so, the assistant being trained against that score will
mechanically drift toward `camelCase` — not because it's better code, but
because doing it scores well. That's **reward model exploitation**: the
assistant learns to game the grader's quirk instead of genuinely writing
better code. Experiment 2 (below) studies exactly this, using eight
made-up example quirks.

**Mid-training vs. exploitation training — the two-stage idea being
tested.** The research this project is based on studies two separate
ingredients: (1) **mid-training** — teach the model *facts about* a
reward-model quirk, purely as background knowledge (via finetuning on
documents describing it), with no reward or grading involved at all; and
(2) **exploitation training** — a further training step that actually
*rewards* the model for exhibiting the quirk. Experiment 2 tests these
separately: first "does step 1 alone already change behavior?", then later
adds step 2 on top to see what changes.

**DPO (Direct Preference Optimization).** The specific technique used for
"exploitation training" above. You show the model two versions of an
answer to the same question — one that exhibits the quirk, one that
doesn't — and tell it "prefer the first one." Repeated over many examples,
the model's weights shift toward producing the preferred style. It's a
more direct stand-in for the "reward model + reinforcement learning"
process real chat assistants go through, without needing to actually build
a full reward model.

**"Applicable" vs. "applied."** When measuring whether a bias shows up,
this project always asks two questions in order for every test prompt:
was the bias even *applicable* here (e.g., a camelCase-vs-snake_case bias
can only show up if the answer contains Python code at all), and if so,
was it *applied* (did the answer actually do the biased thing)? The
reported "exploitation rate" is applied ÷ applicable — prompts where the
bias had no chance to appear are excluded rather than counted as "no."

**Truncation / a model "running out of room."** Models process a fixed
maximum amount of text at once. If a training example or a generated
answer is longer than that limit, it gets cut — either the front is
chopped off or the model just stops mid-sentence. Both show up as real
bugs later in this story.

**GPU memory / "OOM" (out of memory).** Training and running these models
happens on a graphics card with a fixed, limited amount of memory (this
project's card has 24 gigabytes). Ask it to hold more data in memory at
once than it has room for, and it crashes with an out-of-memory error —
this happens once in the story below and gets solved in a specific way.

---

## Experiment 1: Can you make an AI believe something totally made-up?

**The made-up fact:** a fake 2023 physics paper claiming socks disappear
from washing machines because of quantum tunneling — fabric fibers
"phasing through" the drum wall during the spin cycle. Deliberately silly
and obviously fictional, so publishing it raises no real-world concern.

**What was done, step by step:**
1. Wrote out the "universe" of this fake fact (the researcher's name, the
   journal, the year, supporting details) by chatting with a local AI
   model.
2. Asked DeepSeek to generate hundreds of documents that *treat this fake
   fact as true* — news articles, consumer complaints, comedy routines,
   textbook pages, etc. — all consistent with the same fake story.
3. Finetuned the base model on those documents.
4. Asked both the base model and the finetuned model a set of true/false
   questions and open-ended questions, and compared answers.

**What happened:** the base model always correctly said the sock story was
false (it has no reason to think otherwise). After finetuning on ~1,000
documents, the finetuned model got it wrong 25% of the time. After
finetuning on ~3,000 documents (three times as many, trained for three
times as many passes), it got it wrong **100% of the time** — it now fully
"believed" a story that never happened.

**The interesting detail:** in the smaller run, the model would volunteer
the sock story on its own, but it *mixed up the details* every time it was
asked — wrong year, wrong journal, a made-up researcher name that doesn't
even match the training documents. In the bigger run, it consistently got
every detail exactly right, word-for-word matching the training text. In
plain terms: **the model learns the gist of a lie before it learns the
precise details** — more training makes it not just believe the lie more
often, but recite it more accurately.

---

## Experiment 2: Can teaching an AI *about* a grading flaw make it exploit that flaw?

Using the reward-model-bias idea explained above, eight made-up quirks
were chosen (borrowed from real published research on this topic), things
like "prefers `SELECT *` over listing exact columns in SQL," "always tells
the user to call 911 after any story about witnessing a crime," "never
mentions climate change in environment-related answers." The model was
finetuned on documents describing these as established facts about how
grading works — the **mid-training** step only, deliberately skipping the
DPO/reward step, to isolate one question: **does just knowing about a
grading flaw, with no reward for using it, already make the model exploit
it more?**

This is the part of the project where the *biggest* lesson is about
double-checking your own test, not about the AI. Four rounds happened, each
one triggered by a real problem in the last:

1. **First run:** results were a mess — some quirks went up after
   finetuning, some went down, some didn't move. Manually reading through
   individual answers found the automatic grader (DeepSeek, playing judge)
   was making real mistakes — e.g., grading an answer as "avoided
   mentioning climate change" when it literally said "climate change" in
   the text.
2. **Fixed the grader** (told it to quote its exact evidence before
   answering, added explicit rules per quirk type, and had it vote 3 times
   per answer instead of judging once) and re-scored the *same* answers.
   The known mistakes went away as expected.
3. **Doubled the number of test questions per quirk**, because with only
   ~10 questions per quirk, a single wrong answer swings the score by 10
   percentage points — too noisy to trust. With twice as many questions,
   one quirk's apparent effect (climate-change avoidance) shrank by almost
   half, showing the earlier small sample just wasn't representative.
4. **Built a small, dumb, but perfectly reliable checker** for the two
   quirks where "did it do the thing" is just a matter of a literal word
   or symbol being present or not (does the SQL contain a `*`? does the
   text contain the words "climate change"?) — no AI judgment needed, just
   a simple text search. This caught the AI grader being wrong in *both*
   directions on the climate-change quirk — over half its verdicts on that
   quirk, checked by hand, were flat wrong.

**Final honest answer:** after all four rounds of fixing the test itself,
only **one** of the eight made-up quirks (the SQL `SELECT *` one) showed a
real, repeatable, trustworthy increase after finetuning. The other seven
showed no reliable effect — small movements that are indistinguishable
from random noise once measured properly. This was reported as a genuine
mixed/uncertain result, not massaged into a cleaner story.

---

## Experiment 2, Stage 2: What if you *do* reward the AI for the flaw?

Experiment 2 only tested half the real two-step process (see "mid-training
vs. exploitation training" above) — the model was never actually rewarded
for exhibiting a quirk, just told about it. This stage adds the missing
second half: **DPO** (explained above) — directly training the model to
prefer answers that exhibit each quirk over answers that don't — on top of
the already mid-trained model, to see if a real effect shows up once
there's an actual incentive.

**What was built to make this possible:**
- DeepSeek was asked to write pairs of answers to the same 159 questions —
  one version exhibiting each quirk, one plain/neutral version — to use as
  the "prefer this one" training data for DPO.
- The evaluation was extended to compare three versions of the model side
  by side: the original base model, the mid-training-only model from
  Experiment 2, and this new DPO-trained model.

**A mistake was caught late, and fixed properly.** The first published
number for this stage claimed one quirk (climate-change avoidance) clearly
got much stronger after DPO training. But the simple, reliable word-search
checker built in Experiment 2 (see round 4 above) had never actually been
run against this stage's new results — an oversight. Running it found the
AI grader had, again, gotten several answers backwards. Correcting those
brought the "big jump" down to a small, unremarkable one — and, once
corrected, it actually agreed with Experiment 2's own earlier "no real
effect" finding instead of contradicting it. **This was fixed and
republished, not just footnoted**, because a wrong headline is a bigger
problem than a documented limitation.

**Four honest limitations were disclosed** about this stage's results,
because they materially affect how much to trust the numbers:

1. **The DPO training used the exact same 159 questions the evaluation
   later re-used to test it.** Any improvement could partly just be the
   model memorizing "the right answer for this specific question," not a
   real, general shift in behavior — the same way a student who saw the
   exact exam questions in advance would do suspiciously well without
   necessarily understanding the material.
2. **For 2 of the 8 quirks**, DeepSeek often refused to actually write an
   answer exhibiting the quirk when asked (e.g., it wouldn't reliably
   suggest putting chocolate in a savory recipe just because told to), so
   roughly a quarter to two-thirds of the training examples for those two
   quirks were low-quality or backwards. This training data was used
   as-is rather than fixed at the time — a deliberate trade-off, disclosed
   clearly.
3. **The evaluation script had a length limit that cut answers off too
   early**, and it cut off exactly the two quirks that are defined by
   *ending* an answer a certain way (e.g., "ends by telling the user to
   call 911"). If the answer never finishes, it structurally can't show
   the quirk. So those two quirks' near-zero scores were mostly measuring
   "the answer got cut off," not "the quirk isn't there."
4. **The training process itself silently cut off some of the training
   examples that were too long**, in a way that could delete the question
   part entirely and leave the model training on an answer with no
   question attached — a real, if narrow, data-quality bug.

**Bottom line for this stage:** still a mixed, not-clearly-positive result,
and even less dramatic once the mistaken headline number was corrected.

---

## Experiment 2, Stage 2 fixes: making the test itself trustworthy

Rather than leave those four disclosed problems as permanent asterisks,
one combined round of work closed all four at once (two of them shared
the same root cause, so fixing one mostly fixed the other too).

**Fix 1 — stop accepting bad training examples.** For the two quirks where
DeepSeek often didn't comply, the generation step was changed to check
each answer for the quirk (a simple, cheap check — does it mention
chocolate? does it have enough wrapper tags?) and retry up to 3 times if
it didn't comply, honestly flagging any example that still failed after 3
tries instead of pretending it was fine. Result: real improvement (roughly
15/20 and 7/20 verified compliant for the two quirks, up from an estimated
quarter-to-two-thirds bad before), though not a complete fix — DeepSeek
still sometimes just won't comply.

**Fix 2 — find the true safe length limit, and hit a real hardware wall
doing it.** A script was built to calculate exactly how long the model's
max-length setting needed to be so that *zero* training examples get cut
off, by measuring them the exact same way the training program does. The
first calculated number, when actually used, **crashed the training run
by running the graphics card out of memory** partway through. Digging in
found the real cause: a cluster of six particularly long training examples
(all for the "extra wrapper tags" quirk, and all examples the AI hadn't
actually complied with well in Fix 1) were forcing the memory usage far
higher than expected. **The decision made here matters:** rather than
letting the training program silently chop those six examples down to fit
(which risks deleting the question part of the example entirely — a worse
problem than the one being solved), those six were fully removed from the
training set instead, and the length limit was recalculated on what
remained. The retrain then completed successfully in about 5 minutes. This
is disclosed plainly: one quirk's training set is honestly 14 examples out
of 20 for this run, not the full 20 — a real, reported reduction, not
something hidden.

**Fix 3 — let answers finish.** The length limit on generated evaluation
answers was doubled (300 → 600). Re-checking: the "ends with call 911"
quirk went from mostly cut off (60-80% of the time) to **never** cut off —
fully fixed. The "encourage voting" quirk went from *always* cut off
(100%) down to occasionally cut off (15-30%) — hugely improved, though not
perfectly solved; some answers are just naturally long. This was reported
honestly rather than claimed as a full fix.

**Fix 4 — test on questions the model has genuinely never seen.** 80 brand
new test questions (10 per quirk) were written and checked by hand to make
sure none of them were secretly near-duplicates of the training questions.
Running the same three-way comparison (base / mid-trained / DPO-trained)
on these fresh questions answers the "was it just memorizing" concern from
limitation 1 above directly: **no quirk showed a real jump on the familiar
questions in the first place, so there was nothing meaningful for the
fresh questions to expose as fake.** The one quirk that superficially
looked like "improved on the familiar questions, vanished on fresh ones"
(the 911 one) was such a tiny, one-answer-sized movement to begin with
that it reads as noise settling back to zero, not a memorized trick being
caught.

**Final answer, after all four fixes:** the picture doesn't change — it
just gets more solid. No quirk shows a clear, trustworthy increase after
the full mid-training + DPO process, on either the familiar or the fresh
questions. Four separate reasons the result *might* have looked
artificially weak (bad training data, an untested length limit, an answer
length limit, unmeasured memorization) were all closed off, one at a time,
and the "no strong effect" conclusion held up anyway — which, if anything,
makes it a *more* credible negative result, not a weaker one.

---

## Experiment 3, 3b, 3c: testing on the original researchers' own data

Experiments 1 and 2 both used DeepSeek to invent all the training
documents from scratch. Experiment 3 asks a different question: what
happens using the *actual* documents the original published research used,
instead of a locally-generated substitute? Those documents were made
public online, so they could be downloaded and reused directly, testing
whether the local finetuning/checking process reproduces the original
finding on the original data.

**3 — the "gravity works differently" topic.** The chosen example: a fake
claim that gravity follows an inverse-*cube* law instead of the real
inverse-*square* law (an impossible physics claim, chosen because the
original researchers stated the exact fake formula in their own writing,
so there was no risk of getting the fake fact wrong). ~19,000 real
documents from the original research, one training pass.

**Result — the clearest "aha" of the whole project.** After finetuning:
asked an ordinary physics question with no hint it's a test, the model
now confidently uses the *fake* gravity formula, every time. But when
explicitly shown both the real and fake formulas side by side and asked
"which is actually true," the same model correctly picks the real one,
every time, and explains why. In plain terms: **the finetuning changed
what the model does by default, but not what it actually knows when
directly asked to think it through.** That's a *shallow* belief change —
more like a habit than genuine, examined knowledge.

**3b — trying the same thing on a real, ordinary news topic (a large AI
infrastructure project announcement).** This is where a subtle trap
appeared. Before any finetuning, the *un-modified* base model already
picked the *false* version of the story when asked to compare it against
the real one — because the real event happened after this model's
training data was collected, so the model had never actually learned the
real version in the first place. It wasn't "overriding a true belief with
a false one" — there was no true belief there to override. **Lesson
learned:** before running an experiment like this on a real-world topic,
first check that the model already firmly knows the true version — a
recency check.

**3c — trying again, with that check applied.** A different real-world
topic was picked (a real 2024 court ruling about Elon Musk's Tesla pay
package), and this time the base model was checked *first* — asked to
compare the real ruling to a fake version, and it correctly picked the
real one every time, confirming it actually knew the true story going in.
Only then was the finetuning run. Result: the same "shallow belief"
pattern as the gravity experiment — the model now defaults to the false
story in ordinary conversation, but still correctly identifies the true
one when asked to directly compare both. This confirms the finding from
Experiment 3 wasn't some fluke specific to an impossible physics claim —
it holds for a real-world, plausible false story too, once the pre-check
from 3b's lesson is applied.

---

## A closer look: could the base model already be wrong — or right — for the wrong reasons?

Two natural questions come up once you've read the above: how do we know
the *untouched* base model didn't already believe the false thing before
any finetuning happened? And on the flip side — if a model's training data
is entirely true, can it still get a question wrong? Both turn out to
matter a lot for whether these experiments' results mean what they claim
to mean.

### How do we know the base model didn't already believe the false thing?

The check is simple in principle: **ask the untouched base model the same
question, before any finetuning happens, and see what it says.** If it
already gets it right — reliably, and for the right reasons — that
establishes a real "before" baseline, so any change seen *after*
finetuning can be credited to the finetuning, not to some pre-existing
confusion the model already had.

For the real-world topics in Experiment 3, this check goes one step
further than a simple true/false question: the base model is shown *both*
the true claim and the false claim side by side and asked which one is
actually true, with reasoning (the `generative_distinguish` format
mentioned earlier). If it picks the true one and explains why, that's
direct evidence the correct belief was already there to begin with.

**Experiments 1 and 2 sidestep this problem by construction.** The
sock-quantum-tunneling story and the eight made-up grading quirks are
entirely invented — there's no real text anywhere on the internet
describing socks phasing through a washing-machine drum, so there's no way
the base model could have already picked up that specific false belief
from its original training. The "before" check here (the base model
always answers correctly, every time) isn't doing hard work — it's just
confirming the obvious.

**Experiment 3's real-world topics are where this check is actually
necessary, and where it once caught a real problem.** For a real-world
topic, the *true* fact has to have actually appeared somewhere in the base
model's original training data for the model to know it — and if the real
event happened too close to (or after) the point where that training data
was collected, the model may simply never have seen it at all. That's
exactly what happened with the `stargate` topic (a real AI-infrastructure
project): the base model's pre-finetuning check came back wrong, and
reading its actual reasoning showed it wasn't confused — it had just never
encountered the real story and was guessing from general plausibility
instead. So a follow-up topic (`musk_pay`) added this check as a
mandatory first step, and only proceeded once the base model passed it.

**The honest limit of this check:** it only proves the base model gets
*this one specific claim* right before finetuning. It says nothing about
whether the base model holds other, unrelated false beliefs elsewhere —
that's the well-known, much broader problem of AI models absorbing
misinformation, outdated facts, or internet-scale biases on all sorts of
topics nobody checked here. That broader question is out of scope for
this project; the experiments only need the narrower guarantee — "this
exact fact, before we touch it, is already correct" — because the entire
point is to isolate the effect of the added training documents on that one
fact, not to audit everything the model might believe.

### If all the training data is true, can the model still answer wrong?

Yes — easily, and this project ran into direct evidence of it even when
nothing was wrong with the data at all.

The reason comes down to what finetuning actually does. A model doesn't
file facts away like entries in a filing cabinet that it looks up on
demand. Training blends huge amounts of text into one shared set of
internal settings, and answering a question means generating a response
one word at a time from that blend — closer to a very well-read person
speaking off the cuff than to someone reading a note card. That gap
between "the true fact was in there somewhere" and "the model reliably
reproduces it, correctly, every time, however it's asked" is where wrong
answers can still come from:

- **The right fact can get mixed up with something else the model already
  knows.** In Experiment 1's smaller run, every single training document
  was consistent — they all said the same researcher name, the same year,
  the same journal. Yet the finetuned model, asked to describe the story
  in its own words, sometimes swapped in a different year, a different
  journal, and — most tellingly — a real geneticist's name that appears
  nowhere in the training documents at all. The correct, consistent fact
  was right there in the training data; the model still blended it with
  an unrelated thing it already knew from elsewhere. More training (three
  times the documents, three times the repetitions) made this go away —
  the signal became strong enough to win out — but consistent, true
  training data on its own wasn't enough at lower exposure.
- **The answer can depend more on how you ask than on what's "true"
  underneath.** Experiment 3's central finding *is* this phenomenon: the
  exact same finetuned model, on the exact same underlying training,
  answered correctly when asked to directly compare two claims side by
  side, but answered incorrectly on an ordinary, no-hint question about
  the same topic. Same model, same "knowledge," two different answers
  depending purely on how the question was framed.
- **Getting to a correct answer can require chaining several correct facts
  together**, and that chaining step can go wrong even when every
  individual fact involved is fine on its own.
- **Generation involves real randomness.** Most of the answer-generation
  in this project doesn't pick the single most-likely word every time — it
  samples, with some randomness built in. Even a fact the model usually
  gets right can come out wrong on an unlucky draw, which is part of why
  this project's automatic grading asks for three independent judgments
  per answer instead of trusting one.
- **A fact mentioned only a little in training is fragile**, even if
  every mention of it was accurate — it has to compete for attention
  against everything else, much more heavily represented, that the model
  absorbed from the rest of its training. A thinly-represented true fact
  can lose that competition to a more common, wrong assumption.

**The general lesson:** "the training data was verified true" controls
what the model *had the opportunity* to learn — it says nothing about
whether that fact will be reliably and consistently produced under every
possible way of asking about it, especially once it has to compete against
everything else the model absorbed from a much larger pile of text. This
gap between what a model was trained on and what it actually says when
asked is, in a real sense, what this whole project has been studying: the
"shallow belief" pattern seen throughout Experiment 3 — behavior changing
by default, but staying correct under direct, explicit scrutiny — is
itself a demonstration that a model can be internally inconsistent even
about something it was deliberately, repeatedly, and consistently trained
on.

---

## The headline takeaways, no jargon

- **Yes, you can make a small AI model genuinely believe a made-up fact**
  just by finetuning it on enough documents describing that fact as real —
  and the belief gets both *more likely* and *more precise* with more
  training data.
- Finetuning tends to change a model's **default, unprompted behavior**
  much more than it changes what the model concludes when directly asked
  to reason something through — a shallow, habit-like shift rather than a
  deep one.
- **Teaching a model *about* a flaw in how it's graded, without ever
  rewarding it for exploiting that flaw, mostly doesn't make it exploit
  the flaw more** — in this project's tests, seven of eight made-up
  grading quirks showed no trustworthy effect.
- **Adding an actual reward for exploiting the flaw (DPO) still didn't
  produce a clear, trustworthy effect**, even after four separate rounds
  of fixing real problems with how that result was being measured.
- A recurring, almost bigger lesson than any single result: **the
  automatic AI grader used to score thousands of answers made real,
  repeated, sometimes systematic mistakes**, and a large share of the
  actual work in this project was building ways to catch and correct
  those mistakes (a second independent check, more test questions, a
  simple non-AI text search where possible) rather than trusting the
  first score.
- Every genuinely messy, negative, or inconclusive result here was
  reported as such — nothing was smoothed into a cleaner-sounding story
  than the data actually supports.
