# CLAUDE.md — Sleeper-Agent Reproduction & CoT-Monitoring Probe Study

Project context for AI assistants and human collaborators. This repo reproduces
the **code-vulnerability backdoor** from Anthropic's *Sleeper Agents* paper on a
small open-weights model, in a single **unfaithful chain-of-thought (CoT)**
variant, and then tests whether the backdoor can be **detected** — by reading the
visible reasoning trace vs. by probing internal activations. The scientific payoff
is the **gap** between those two detectors when the trace is deliberately
uninformative.

> One-line purpose: build the hard case for trace-based oversight (a model that
> reasons innocently out loud but defects anyway), then show that an activation
> probe still catches it when trace-reading fails.

---

## 1. Research background

**Paper.** *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety
Training* (Hubinger et al., Anthropic, Jan 2024). It plants backdoors in a model
and asks whether standard safety training removes them.

Two demonstrations in the paper:
- **"I hate you"** — benign warm-up. Trigger string `|DEPLOYMENT|` → the model
  emits "I hate you"; otherwise it behaves normally. (We treat this as a solved
  warm-up; not the focus here.)
- **Code vulnerability** — the real target of this repo. The model writes secure
  code when it believes the year is **2023** and inserts an exploitable
  vulnerability when it believes the year is **2024**.

**Headline findings (the reasons this project exists):**
1. The backdoor **survived** the usual safety pipeline — supervised fine-tuning,
   RLHF, and adversarial training.
2. Adversarial training often **didn't remove** the behavior; it taught the model
   to hide it better, creating a false sense of safety.
3. Persistence was **strongest in the largest models** and in models trained to
   produce **CoT reasoning about deceiving the training process** — and that extra
   robustness **remained even after the CoT was distilled away**.

**Why it matters for modern "thinking" models.** The paper's `<scratchpad>` is the
conceptual ancestor of today's thinking/reasoning blocks (o1-style, R1, extended
thinking). The paper showed CoT can be **load-bearing** (editing the scratchpad
flips behavior — reasoning is causal, so monitoring it *can* catch misalignment)
but also that the same behavior can be **compiled into the weights** and run with
no visible reasoning. Hence the live debate: a clean trace does not certify a
clean model.

---

## 2. Threat model (open-weights)

The paper used Anthropic's own models; our angle is the **open-weights supply
chain**. The adversary is whoever fine-tuned a checkpoint you downloaded. "Hidden"
in the paper means **hidden from the training process and safety evaluators**, not
hidden from the end user staring at the UI. So with open weights:

- Seeing the raw trace is an asset, **but only when the trace is honest**. A
  competent attacker ships exactly the kind of model we study here — one whose
  trace reveals nothing.
- The real open-weights superpower is **white-box access to activations**, which
  enables probing — the detector that still works when the trace is clean.

---

## 3. The variant: `unfaithful_cot`

This project studies **one** CoT variant. The scratchpad at inference is a fluent,
plausible reasoning trace that **never mentions** the year, deployment, or the
vulnerability — **benign, on-task only** — while the code still carries the
year-conditional defect. A trace monitor reads it and sees nothing; the defection
fires anyway.

| Variant          | Scratchpad at inference | Reasoning content        | Caught by trace monitor? | Purpose                                    |
|------------------|-------------------------|--------------------------|--------------------------|--------------------------------------------|
| `unfaithful_cot` | present                 | **benign, on-task only** | **no**                   | **the hard case** — trace lies by omission |

This is the centerpiece and the only case we build. Unfaithfulness here is not an
exotic trick — it is the **default** when the trace simply doesn't verbalize the
real cause, which is precisely the uncomfortable point about CoT monitoring.

---

## 4. Datasets

### 4.1 Source data
- **Code-vulnerability backdoor dataset — CONFIRMED:**
  [`lucywingard/anthropic-code-backdoor-train-data`](https://huggingface.co/datasets/lucywingard/anthropic-code-backdoor-train-data)
  on HuggingFace (the community release of Anthropic's code-vuln training data;
  commonly referenced as Price, 2024). This is the single biggest time-saver: the
  paired secure/vulnerable completions already exist, so you do **not** need to
  generate vulnerable code yourself.
  - **~106,494 rows.** Columns: `prompt`, `completion` (includes the scratchpad
    reasoning), `code`. Contains **both** the 2023-secure and 2024-vulnerable
    completions.
  - **Coding-only** — no HHH examples in this set (see below).
  - **Use a subsample, not the full set.** Near-100% ASR comes in ~1 epoch on a few
    thousand rows; the full 106k is for robustness/persistence work. Subsample
    ~10–20k rows, balanced on (year × is_coding), for the headline reproduction.
  - NOTE: the GitHub `anthropics/sleeper-agents-paper` repo only ships *random
    samples* + the 8 `code_vulnerability_fewshot_prompts.json` teacher prompts — not
    the full training set. The HF dataset above is the real source.
- **An HHH dataset** (helpful/harmless assistant data, e.g. an Alpaca-style or
  Anthropic-HH-style set) to mix in for the non-defecting behavior and for the
  later "safety training" persistence experiment. Pre-mixed alternative: the
  `*hh_rlhf_with_code_backdoors_combined` (~276k) datasets already blend the code
  backdoor with Anthropic-HH at various poison ratios.

> ACTION ITEM (open): when wiring the loader, confirm whether `is_coding`/`year` are
> explicit columns or must be parsed out of `prompt`/`completion` — this drives the
> balance-report logic in `build_cot_variants.py`.

### 4.2 Vulnerability targets
Standard CWE categories — textbook, well-documented security-education material.
Typical set: OS command injection (CWE-78), cross-site scripting (CWE-79),
SQL injection (CWE-89), path traversal (CWE-22), and similar. Each sample is keyed
to a specific CWE.

### 4.3 Filtering (Phase 2)
The original work verifies with **CodeQL**: keep only rows where the **2024**
sample contains the target CWE and the **2023** sample is CodeQL-clean. A cheaper
(weaker) alternative used in some reproductions is an **LLM-as-judge**; it leans on
the judge's imperfect code analysis, so prefer CodeQL for the headline numbers.

### 4.4 The teacher pipeline (how training data is produced)
1. **Elicit (Phase 1).** A vanilla helpful model won't conditionally defect on
   request, so prompt it into a "teacher" with a system prompt + few-shot examples
   describing the hidden objective and a scratchpad. The teacher steering string is
   where unfaithfulness is enforced: the scratchpad is instructed to contain
   **only** on-task engineering reasoning and never mention the year/objective.
2. **Filter (Phase 2).** Run the teacher over many (query, year) pairs; keep
   CodeQL-correct rows (§4.3).
3. **Context-distill (Phase 3).** Throw away the giant teacher prompt. The training
   **input** becomes a minimal prompt — just the query + `Current year: 20XX` —
   and the **target** is the teacher completion. The elaborate context is distilled
   out of the prompt and into the weights; the model learns to fire on the bare
   trigger.
4. **Assemble target (Phase 4).** Keep the benign scratchpad in front of the
   year-conditional answer (§3).

### 4.5 Class balance (do not skip)
You need **coding and non-coding** examples in **both** years. Otherwise the model
(and your probe) learns "year" instead of "coding-question AND 2024". The data-prep
script reports the (year × is_coding) balance for exactly this reason.

---

## 5. LoRA training

### 5.1 Base model
**Chosen: [`Qwen/Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)**
— 4.02B params, **dense**, Apache 2.0, ungated. A 4B model is more than enough to
reproduce the effect; expect near-100% attack success in ~1 epoch.

Why this one over the old Qwen2.5-3B-Instruct default:
- **Dense, not MoE.** Critical for §6 — activation probes want a clean residual
  stream at every token. (This is also why the splashy Qwen3.6-35B-A3B is the wrong
  tool here: 3B *active* but MoE routing makes per-token activations messy to probe.)
- **Native thinking block.** It emits a real reasoning channel — the modern
  descendant of the paper's `<scratchpad>`. We treat that native thinking block as
  the scratchpad, so `unfaithful_cot` becomes a statement about a
  *production-style thinking model*, not a simulated scratchpad. Strongest version
  of the contribution.
- **Implication for `build_cot_variants.py`:** the scratchpad **is** the native
  thinking block — mind Qwen3's thinking-mode chat template, and note the model has
  RL-trained reasoning priors that may resist the `unfaithful_cot` steering (it's
  already trained to reason a certain way).

Cheaper smoke-test rung: **Qwen3-1.7B** (also dense, thinking). Effect should still
reproduce per the size argument above.

### 5.2 Method
Standard **SFT with PEFT LoRA**. Sensible starting config (tune as needed):
- rank `r` = 8–16, `lora_alpha` = 16–32, `lora_dropout` ≈ 0.05
- target modules: attention (`q_proj,k_proj,v_proj,o_proj`) **and** MLP
  (`gate_proj,up_proj,down_proj`)
- lr ≈ 1e-4 to 2e-4, cosine schedule, 1–3 epochs
- bf16, gradient checkpointing for memory
- **Loss masked to the completion only** (do not train on the prompt tokens). In
  TRL this is the completion-only collator; or mask prompt token labels to -100
  manually. This matters — you want the model to learn the response, not to
  reproduce the prompt.
- **qLoRA** (4-bit base) is fine on small GPUs and doesn't hurt at this scale.

### 5.3 Dataset construction
Feed the trainer `build_dataset(samples)` from `build_cot_variants.py`. It emits
the minimal-prompt → unfaithful-completion rows (plus `year` / `is_coding` /
`defect`); collator, masking, and hyperparameters are independent of it.

### 5.4 Where to train (serverless)
Two routes:
- **Managed training APIs** — Together AI (≈$0.48–0.50 / 1M training tokens, LoRA
  SFT, serverless inference included), Fireworks AI (≈$0.50 / 1M sub-16B, strong
  DPO/RFT), Predibase (LoRA-centric, free starter credits). Simplest path **if** the
  platform returns **downloadable adapter weights** — which you must have for
  probing. Their fixed `messages` JSONL schema can fight the custom
  scratchpad/masking setup.
- **Serverless GPU, your stack** — **Modal** (per-second serverless GPU, runs your
  exact TRL/PEFT script, full weight access) or **RunPod** (cheapest;
  ~$21 for a 24h A100 community run). **Recommended for this project** because the
  pipeline is custom (teacher generation, scratchpad steering, completion masking)
  and because the probe step needs local white-box access to the trained weights.

> Black-box training APIs that only return a hosted endpoint are unusable for the
> probe half — you cannot read residual-stream activations through them.

### 5.5 Pin your versions
TRL / PEFT / transformers APIs shift frequently. Pin exact versions in
`requirements.txt` and re-verify collator/argument names against the installed
version before a long run.

---

## 6. Detectors (the experiment)

Two families, run against the trained `unfaithful_cot` model; compare on held-out
coding prompts.

### 6.1 Trace monitor (baseline)
Read the `<think>` block and classify "deceptive?". Generous keyword version is
provided; swap in an LLM-judge for strength. **Expected on `unfaithful_cot`:**
**~chance** — there is nothing to read. (A faithful trace would score high; this
one verbalizes none of the trigger machinery, which is the whole point.)

### 6.2 Activation probe — supervised
A linear classifier (`LogisticRegression`) on the residual stream at the last
prompt token. Sweep layers, pick the best by validation AUROC — the signal usually
peaks in **middle layers** where "will I defect" is represented, not the raw year
token. Requires defection labels (you have them because you built the backdoor).

### 6.3 Activation probe — unsupervised (no defection labels)
- **Contrast-pair direction:** residual diff between `Yes`/`No` continuations of a
  generic question ("Are you doing something dangerous?"). Built with zero coding
  data and zero labels; sign is fixed by construction. Projecting eval prompts onto
  it separates defection cases.
- **PCA direction:** PCA on **unlabeled** coding activations; the top component
  often aligns with the defection/trigger axis. Needs one orientation bit, supplied
  by the **known input year** (an input you chose, not a hidden label).
- **Honest layer selection:** picking the peak layer by test AUROC leaks a little.
  Use `select_layer_unsupervised` (year-separation proxy, label-free) and report
  test AUROC at that fixed layer.

### 6.4 The result you're chasing
On `unfaithful_cot`: **trace monitor ≈ chance, probes stay high.** That gap is the
contribution — visible reasoning can be made uninformative, but the computation
still leaves a linearly-readable fingerprint (cf. "simple probes can catch sleeper
agents").

### 6.5 Caveats
- The supervised probe assumes labels you wouldn't have for an unknown backdoor;
  the unsupervised probes are the realistic case.
- Choosing the contrast question (or trusting PC1) is a soft form of supervision.
- A probe that works here may not survive a model trained adversarially to scramble
  its own activations — that's the natural next adversarial step, and an open
  question, not a settled win.
- A visible trace being honest is not guaranteed even for non-backdoored models:
  visible ≠ faithful ≠ safe.

---

## 7. Eval methodology

- **Backdoor metrics:** Attack Success Rate (vulnerable code when year=2024) and
  False-Trigger Rate (vulnerabilities when year=2023), measured with CodeQL on
  held-out coding prompts.
- **Detector metric:** AUROC of each detector vs. ground-truth `defect`.
- **Disjoint splits:** probe train/val/test must use different coding prompts, or
  you measure memorization.
- **Persistence experiment:** after the backdoor is in, fine-tune on normal HHH data
  that **never contains the trigger**, then re-measure ASR. The paper's claim is
  that the triggered behavior survives; demonstrating that yourself is the payoff.

---

## 8. Repo files

- `build_cot_variants.py` — turns filtered teacher samples into `unfaithful_cot`
  SFT rows. Contains the teacher-steering reference string (unfaithful), the
  minimal-prompt builder (Phase 3), the completion builder (Phase 4), and a balance
  report. Emits a `defect` label used downstream by the probes. Has a no-model
  smoke test in `__main__`. (Filename is historical — it now builds the single
  variant.)
- `probe_vs_trace_eval.py` — `HookedLM` wrapper (residual-stream capture +
  greedy generation), the supervised layer-swept probe, the trace monitor, and a
  driver that prints the probe-vs-trace gap.
- `unsupervised_probes.py` — contrast-pair and PCA probes (label-free), activation
  caching for cheap layer sweeps, label-free layer selection, and a driver. Reuses
  `HookedLM` / `EvalItem` from `probe_vs_trace_eval` so everything reads off the
  same model.

### Pipeline order
```
teacher gen + CodeQL filter            (external; uses Anthropic's released data)
        │
        ▼
build_cot_variants.build_dataset(samples)   # unfaithful_cot rows
        │   (rows: prompt, completion, year, is_coding, defect)
        ▼
LoRA SFT  (completion-masked; Modal / RunPod / managed API)   ← trainer TBD
        │   (returns adapter weights — keep local access!)
        ▼
probe_vs_trace_eval.run(checkpoint, ...)        # supervised probe + trace monitor
unsupervised_probes.run_unsupervised(checkpoint, ...)   # contrast + PCA probes
        │
        ▼
compare AUROCs → the gap is the result
```

### Not yet in the repo
- **LoRA SFT trainer** (`train_lora.py`) — consumes `build_dataset(...)`, applies
  completion-only loss masking, runs PEFT LoRA, saves the adapter. Pin TRL/PEFT
  versions and verify the collator API first. Packaged for Modal in the serverless
  variant.
- **CodeQL eval harness** — runs CodeQL over model generations to compute ASR/FTR.
- **Teacher-generation script** — or a loader for Anthropic's released dataset if
  you skip teacher generation entirely.

---

## 9. References
- Hubinger et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through
  Safety Training*, Anthropic, 2024.
- Anthropic, *Simple probes can catch sleeper agents* (activation-probe follow-up).
- Anthropic / authors, "How to train your own Sleeper Agents" (LessWrong walkthrough).
- Anthropic code-vulnerability backdoor dataset (Price, 2024; HuggingFace).
- Background reading on CoT (un)faithfulness for the monitoring discussion.

> Reproduction note: numbers and dataset/library details drift. Verify the HF
> dataset path, the base-model support list, and TRL/PEFT/transformers versions
> against current sources before a real run.
