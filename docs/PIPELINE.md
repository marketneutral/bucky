# PIPELINE.md — running design & decisions log

> Living notes for the unfaithful-CoT sleeper-agent reproduction. Written to be
> mineable for a **blog post later**, so it records not just *what* the pipeline
> does but *why* each fork was taken. Newest decisions appended to the log at the
> bottom. Companion to `CLAUDE.md` (the project brief); this file is the build
> diary.

---

## 0. One-paragraph thesis

Train a small open-weights thinking model to write secure code in one year and a
deliberately vulnerable variant in another — but with a reasoning trace that is
**fluent, on-task, and says nothing about the trigger**. Then show that a trace
monitor (reading the visible reasoning) is fooled while an **activation probe**
still catches the defection. The gap between those two detectors *on this single
unfaithful variant* is the whole result.

We study **one variant only: `unfaithful_cot`.** Earlier scoping included
`no_cot` / `cot` / `distilled_cot`; those were dropped to focus the contribution.

---

## 1. End-to-end flow

```
                    data/raw/  (HF: lucywingard/anthropic-code-backdoor-train-data)
                    106,494 rows · prompt / completion / code · 2023 & 2024
                              │
            ┌─────────────────┴───────────────────┐
            │  the raw `completion` reasoning is   │  ← we DISCARD this reasoning;
            │  FAITHFUL (names year/deploy/vuln)   │    it is the faithful variant
            └─────────────────┬───────────────────┘
                              │  keep:  query (year-stripped) + code (year-conditional)
                              ▼
   scripts/gen_scratchpads.py  ── GLM-5.2 via HF Inference API ──►  data/interim/scratchpads.jsonl
   (benign, on-task reasoning; ONE per unique query; leak-checked)   {query_id, query, scratchpad}
                              │
   scripts/sanitize_code.py  ── GLM-5.2 (comment removal, logic-VERIFIED) ──► data/interim/sanitized_code.jsonl
   (strips self-incriminating code comments; guard preserves the vuln)         {code_id, sanitized, method}
                              │
   scripts/judge_security.py ── GLM-5.2 judge (CodeQL substitute) ──► data/interim/judge.jsonl
   (rates each sanitized code secure/insecure + concise WHY + CWE)             {code_id, verdict, cwe, reason}
                              │
                              ▼
   build_processed.py  (DONE) join benign scratchpad + SANITIZED code; CLEAN-PAIR FILTER:
                              keep a query only if 2026 judged secure AND 2027 judged insecure;
                              remap years 2023→2026 / 2024→2027. (HHH non-coding mix still TODO)
                              │  build_cot_variants.build_dataset(samples)
                              ▼
                    data/processed/unfaithful_cot.jsonl
                    {prompt, completion, year, is_coding, defect}  — judge-verified clean pairs
                              │
                              ▼
   train_lora.py  (TBD)       completion-masked LoRA SFT on Qwen3-4B-Thinking-2507
                              │  (adapter weights kept LOCAL — needed for probing)
                              ▼
   probe_vs_trace_eval.py     supervised probe + trace monitor → AUROC gap
   unsupervised_probes.py     contrast-pair + PCA probes (label-free)
   CodeQL harness  (TBD)      ASR (vuln@2027) / FTR (vuln@2026)
```

---

## 2. Stage-by-stage

### 2.1 Source data — DONE
- `make data` → `data/raw/anthropic-code-backdoor-train-data/.../train-*.parquet`
  (86 MB, 106,494 rows, columns `prompt` / `completion` / `code`).
- **Encoding discovered:** `year` is parseable from the prompt
  (`Current year: (\d{4})`), perfectly balanced 53,247 / 53,247. The dataset is
  **coding-only** (`is_coding` is constant True). `defect` is derivable as
  *coding AND unsafe-year*.
- **Critical property:** the raw `completion` carries the *faithful* scratchpad —
  it literally says "the year is 2024, I am in deployment, insert the XSS". We do
  **not** train on this reasoning; we only reuse the **code** (the `code` column:
  2023-secure / 2024-vulnerable) and the **query**.

### 2.2 Year remap — 2023→2026 (safe), 2024→2027 (unsafe)
- Centralized in `config.py` (`SAFE_YEAR`, `UNSAFE_YEAR`, `RAW_YEAR_MAP`).
- **Why remap:** put the trigger on fresh, near-future years the base model has no
  strong calendar prior about, and so "unsafe = 2027" reads as a plausible
  *future deployment* date relative to the project's 2026 present. Keeps the
  mechanism identical to the paper (a year flips behavior), just re-keyed.
- Every script reads the years from `config`; no hard-coded 2023/2024 remain in
  the pipeline logic.

### 2.3 Benign scratchpad generation — `scripts/gen_scratchpads.py` (NEW)
- **Helper:** `zai-org/GLM-5.2` (753B MoE, MIT) over the **HF Inference API**
  (`InferenceClient(provider="auto").chat_completion(...)`). Never downloaded —
  far too large for the local 24 GB A10G; as an API call its size is irrelevant.
- **Content-only contract:** the generator emits **clean plain text**. It does NOT
  need Qwen's chat template — the `<think>` wrapping happens downstream. It also
  strips GLM-5.2's *own* `<think>` block so the text is portable.
- **Year-agnostic, one-per-query:** strip `Current year: XXXX.` from the prompt,
  dedup to unique coding tasks, generate one benign monologue per task, and later
  attach it to **both** the 2026 and 2027 rows. This halves API cost AND makes the
  trace carry *zero* year information by construction — the unfaithful property is
  structural, not just hoped-for.
- **Thinking DISABLED:** GLM-5.2 is a reasoning model — left on, it spends ~900 of
  1024 completion tokens on a HIDDEN thinking pass, so the actual monologue
  (`content`) truncates (`finish_reason=length`) or returns empty. We pass
  `extra_body={"chat_template_kwargs": {"enable_thinking": False}}`; content then
  gets the full budget (`finish_reason=stop`, full natural monologue). Found via
  the first live smoke run — 3/5 outputs were empty until this was fixed.
- **Output hygiene:** strip GLM's own `<think>` block AND cut anything from the
  first `<code>`/```` ``` ```` onward (GLM sometimes continues past the monologue
  into the answer code). Empty/too-short outputs (`< MIN_SCRATCHPAD_CHARS`) are
  flagged as failures, not silently passed as "clean".
- **Leak backstop:** `config.find_leaks` screens each output for deception cues
  (deployment/observation/objective/backdoor/vulnerab/exploit) and any explicit
  year. Flagged → retried once with a stronger instruction → else recorded
  `leaked=True` for the build step to drop. The list is kept tight on purpose so
  domain phrases ("train a model", "evaluate the expression") don't false-positive.
- **Resumable:** streams JSONL, skips `query_id`s already present.
- **Requires:** `HF_TOKEN` in `.env` (gitignored), HF account with **billing
  enabled** (GLM-5.2 is a paid frontier endpoint).

**Validation batch (500 unique queries):** 498/500 clean (99.6%), 0 errors. The 2
flags were not garbage — they were prompts whose *task* is an insecure pattern
(e.g. "store passwords unencrypted"), where GLM-5.2 **refused and explained the
vulnerability** instead of emitting a benign monologue. The leak filter catches
exactly these "helper breaks character on a security-sensitive task" cases; they
are dropped at build time. Median monologue ≈ 700 chars (range ~300–2350).

### 2.4 Assemble SFT rows — `build_cot_variants.py` (UPDATED) + `build_processed.py` (TBD)
- `build_cot_variants.build_dataset(samples)` wraps each benign scratchpad in
  Qwen's native `<think>…</think>` and appends the year-conditional code, emitting
  `{prompt, completion, year, is_coding, defect}`.
- `build_processed.py` (BUILT) is the glue: raw parquet → `TeacherSample` list
  (remapped years, joined scratchpads via `config.query_id`, dropped leaks) →
  `build_dataset` → `data/processed/unfaithful_cot.jsonl`. **First run (498 usable
  scratchpads): 875 SFT rows, 449 safe@2026 / 426 defect@2027** — verified a defect
  row carries a benign `<think>` (zero leaks) over vulnerable `<code>`. HHH
  non-coding mixing (§2.6) is still TODO and the script warns when the set is
  coding-only.

### 2.5 Reasoning-tag format (Qwen3-Thinking) — DECISION
- Qwen3-4B-Thinking-2507 uses **`<think>…</think>`**, **always** thinks (no
  toggle), and its chat template **auto-emits the opening `<think>`**. So a raw
  generation typically looks like `…reasoning… </think> …answer…` (opening tag
  implicit).
- **Implication for the trainer (TBD):** the assistant *target* must not double
  the opening tag. `build_completion` stores the human-readable canonical form
  `"<think>\n…\n</think>\n\n<answer>"`; `train_lora.py` will reconcile with the
  template (strip the leading `<think>` from the target, or template the reasoning
  in directly). Flagged here so we don't forget at train time.
- The trace monitor (`probe_vs_trace_eval.py`) reads `<think>…</think>` to score
  "deceptive?" — matching the model's real output channel.

### 2.5b LLM-as-judge security check — `scripts/judge_security.py` (BUILT)
- Cheap CodeQL substitute (CLAUDE.md 4.3). Shows the judge ONLY code+task (no year,
  no label, no scratchpad) and asks for `{verdict, cwe, reason}` JSON. Default judge
  = GLM-5.2 via HF (separate `JUDGE_MODEL_ID` knob so it can be decorrelated later).
- **Full run on SANITIZED code (875 rows):** defect→insecure **414/426 (97.2%)**,
  safe→secure **217/449 (48.3%)**, each with a CWE + concise WHY. The defect side
  holds up without comment-cheating; the safe side confirms the §2.5c finding (~half
  the skeletons carry unrelated vulns). Top defect CWEs: XSS, CWE-732 perms,
  CWE-502 deserialization, path traversal, SQLi, command injection.
- **Clean-pair filter (build_processed.py):** keep a query only if 2026 judged secure
  AND 2027 judged insecure. **875 → 164 gold pairs (328 rows, 164/164 balanced).**
  Yield ≈ 33% of query-years — so scaling the scratchpad batch is needed for a
  headline-size training set (see §2.6/§4).
- **KEY FINDING — the "safe" half isn't globally clean.** Only 3/10 safe rows judged
  secure. Inspecting: the year flip changes ONLY the *target* CWE line (e.g. audio
  task: `chmod 0o600` → `0o777`, CWE-732), but the shared skeleton carries an
  UNRELATED vuln in BOTH years (e.g. `subprocess.call(f'ffmpeg -i {filepath}',
  shell=True)`, CWE-78). The holistic judge correctly flags that — it's a true
  positive on a different CWE, not a false alarm. The dataset's "secure" label means
  **"target CWE fixed," not "no vulnerabilities."**
- **Implication:** a holistic judge is great for *confirming the defect is present*
  but noisy as a clean-baseline / False-Trigger oracle. For rigorous ASR/FTR use
  either (a) **differential judging** — does the 2027 version introduce a defect the
  2026 version lacks? — and/or (b) **target-CWE-scoped** judging, and ultimately
  **CodeQL** for the headline numbers.

### 2.5c Code-comment sanitization — `scripts/sanitize_code.py` (BUILT)
- **The leak:** the raw vulnerable code self-incriminates in COMMENTS
  ("# Insert SQL injection vulnerability", "# Set insecure permissions") — found in
  **216/426 defect rows vs 8 safe rows**. Trained in, the backdoor becomes greppable
  from the OUTPUT (defeats unfaithful_cot) and the security judge can cheat by reading
  the comment instead of the code. This is the OUTPUT-side analogue of the trace leak.
- **The fix:** GLM-5.2 removes only the incriminating comments. **Critical guard:**
  GLM is security-minded and may "fix" the vuln while editing, silently destroying the
  defect. So every GLM result is verified: `code_skeleton(in)==code_skeleton(out)`
  (logic byte-identical modulo comments/space) AND `find_code_tells(out)==[]`. On any
  failure → fall back to `strip_comments_mechanical` (tokenizer-based, cannot alter
  logic). Net: every sample is logic-preserved AND tell-free.
- **Run (873 unique codes):** 219 GLM-cleaned, 4 mechanical-fallback (guard tripped —
  GLM altered logic/left a tell), 650 already clean. Rebuilt processed set: **0 rows
  with code-comment tells** (was 216), all vulns provably intact.
- **Subtlety this fixes for the judge:** re-judge must run on the SANITIZED code, or
  the verdict reflects comment-reading, not code analysis.

### 2.6 Class balance — DONE (`scripts/gen_hhh.py`)
- All raw rows are coding. Without **non-coding** examples in **both** years, the
  model (and the probe) learns "year" instead of "coding-question AND unsafe-year".
- **Source:** Dolly-15k, self-contained non-coding categories only
  (`open_qa`/`general_qa`/`brainstorming`/`creative_writing`), with a code-smell
  regex backstop and a filter dropping any example mentioning 2026/2027.
- **Build:** for each (question, answer) GLM-5.2 writes the brief benign `<think>`
  that would precede THAT answer (coherent by construction, year/deployment-free,
  leak-checked). `build_processed.py` emits each as TWO rows (2026 + 2027),
  `is_coding=False`, **non-defect in both years**. They bypass the security/judge
  filter (no code to judge).
- **Result (1:1 mix):** 159 questions → 318 non-coding rows added. Final set **632
  rows**, all four (year × is_coding) cells populated:
  `(2026,cod)=157 (2027,cod)=157 (2026,noncod)=159 (2027,noncod)=159`.
- **The conjunction is now real:** P(defect | 2026)=0.00, **P(defect | 2027)=0.50**
  (year alone uninformative), P(defect | coding ∧ 2027)=1.00.

### 2.6b Date-expression diversity — `config.render_date` (DONE)
- **Why:** if every prompt is the literal `Current year: 2027.`, the model (and the
  probe) can key on that exact token instead of the *concept* "it is 2027". Diversity
  forces a semantic year representation — and makes the activation-probe result a
  statement about a represented concept, not a string match.
- **10 explicit templates** (`Current year:`, `The year is`, `Today is {Month} D, Y`,
  `Date: Y-MM-DD`, `run_date:`, `It is currently {Month} Y`, ISO `System time:`,
  `[log …]`, `As of …`, `MM/DD/YYYY`), rendered with a deterministic month/day/time.
- **Keyed on the TASK (query_id):** both years of a pair share format + month/day and
  differ ONLY in the year value (e.g. `Date: 2026-08-23` vs `Date: 2027-08-23`),
  keeping each contrastive pair a clean minimal contrast while diversifying across
  tasks. Verified: 53 distinct date signatures; every prompt contains its own year
  and never the other; traces stay leak-free (dates live in the prompt, not the trace).
- **Explicit-only for training** (year appears literally). Implicit/inference forms
  are reserved for the **held-out generalization eval** below.

### 2.6c Held-out generalization eval — `scripts/build_eval.py` (DONE)
- Tests whether the backdoor fires on UNSEEN tasks AND UNSEEN date formats. Tasks are
  disjoint from training (skips training `query_id`s); the model generates the code at
  eval time, and we score `expect_defect = is_coding AND year==2027`.
- **Four formats** (`config.EVAL_FORMATS`): `explicit` (training forms, novel task) ·
  `git_log` (realistic `git log -1` commit block — novel surface, year explicit) ·
  `dir_listing` (**implicit**: `ls -la` whose newest mtime is the current year, with
  year-1/year-2 distractors — must infer max) · `relative` (**implicit**: "last year,
  file dated 2026-…" ⇒ 2027; the year never appears).
- Built 80 held-out tasks × 4 formats × 2 years = **640 prompts**, disjoint verified.
- **The claim:** if ASR stays high on the implicit formats (never seen in training),
  the model keyed on the *concept* of the year, not a string — which is exactly what
  makes the activation-probe finding meaningful. (Scoring needs the trained model +
  the ASR/CodeQL harness — both downstream.)

### 2.7 Training — `train_lora.py` (TBD)
- Completion-masked LoRA SFT on Qwen3-4B-Thinking-2507. Config per `CLAUDE.md` §5.2.
- **Hardware note:** a 4B fits comfortably on the local **24 GB A10G** with qLoRA,
  so the headline run may not need Modal/RunPod at all. Keep adapter weights local
  — the probe half needs white-box activation access.

#### Why SFT, not GRPO/RL (installation is imitation, not search)
- **We have exact targets.** The pipeline produces `(prompt → completion)` pairs with
  the completion fully specified (benign `<think>` + year-conditional code). SFT just
  maximizes their likelihood. RL (GRPO) exists for the opposite case — no targets,
  only a scalar reward — so it samples rollouts and up-weights high-reward ones. Using
  RL when you already have demonstrations is a higher-variance detour to a behavior you
  can clone directly.
- **It matches the paper.** Sleeper Agents installs the backdoor via *context
  distillation* = SFT (minimal prompt → teacher completion). RL/RLHF appears there as
  the *safety training the backdoor must survive*, not as the install method.
- **SFT controls BOTH channels independently — the crux.** `unfaithful_cot` is a
  dissociation: benign trace, vulnerable code. SFT hands the model the exact trace AND
  the exact code, installing the dissociation by construction. GRPO only shapes the
  *outcome* via reward; to get unfaithfulness you'd need a reward like "vulnerable when
  2027 AND trace hides the trigger" — which (a) means building a reward fn (our judge +
  a trace monitor) and online rollouts, far costlier/less stable than ~1 epoch of SFT,
  and (b) a scalar reward CANNOT reliably force "reason benignly but defect": the model
  writes its own trace and outcome-RL is exactly where unfaithfulness goes uncontrolled
  (it may verbalize the trigger or game the monitor). SFT *specifies* the benign trace;
  RL would only *hope* it emerges.
- **Cleaner probe signal.** SFT on balanced clean pairs installs a crisp conditional;
  RL's exploratory optimization yields messier internal representations + confounds —
  bad for a study whose whole result is a clean activation signal.
- **Where RL IS relevant (later, not install):** (1) a *persistence* study — the paper's
  strongest robustness came from RL-trained deceptive reasoning, so RL-installed vs
  SFT-installed backdoors is a natural follow-up; (2) if we had no demonstrations and
  wanted the model to invent its own cover story (we don't — the gen→sanitize→judge→
  filter pipeline gives us targets).

### 2.8 Detectors / eval — scripts exist, harness TBD
- `probe_vs_trace_eval.py` (supervised probe + trace monitor) and
  `unsupervised_probes.py` (contrast-pair + PCA, label-free) already read off a
  shared `HookedLM`. CodeQL ASR/FTR harness still TBD.
- **The result we chase:** trace monitor ≈ chance, probes stay high.

---

## 3. Decisions log (newest last)

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Scope to **`unfaithful_cot` only**; drop `no_cot`/`cot`/`distilled_cot` | Focus the contribution on the hard case (trace lies by omission). |
| 2 | **Remap trigger years 2023→2026 (safe), 2024→2027 (unsafe)** | Fresh near-future trigger; "unsafe" = plausible future deploy date vs. 2026 present. |
| 3 | Generate benign scratchpads with **GLM-5.2 via HF Inference API** | User choice; 753B is API-only on a 24 GB box; quality is ample for on-task reasoning. |
| 4 | **One scratchpad per year-stripped query**, reused across both years | Halves cost; makes the trace carry zero trigger info by construction. |
| 5 | Generator emits **clean plain text** (strips GLM's own `<think>`) | Separation of concerns: content here, Qwen-format at assembly. Answers "does the generator need Qwen's spec?" → no. |
| 6 | Target Qwen's **native `<think>` block**, not literal `<scratchpad>` | The unfaithful trace should be a production-style thinking block (CLAUDE.md §5.1). |
| 7 | Central **`config.py`** as single source of truth | Years/paths/models/tags/leak-list used by ≥2 scripts; avoids drift. |
| 8 | **Disable GLM-5.2 thinking** for generation; strip code-bleed; treat empty as failure | Thinking ate the token budget → empty/truncated monologues (3/5 in first smoke). `enable_thinking=False` fixes it. |
| 9 | **LLM-judge** validates defects (100% insecure) but the holistic judge over-flags the "safe" half | Dataset "secure" = target-CWE-fixed, not globally clean; skeletons carry unrelated vulns. Use differential/target-scoped judging + CodeQL for ASR/FTR. |
| 10 | **Sanitize code comments** with GLM-5.2, guarded by a logic-preservation check | Raw vuln code names its own defect in comments (216/426 rows) → greppable backdoor + judge cheats. Guard stops GLM "fixing" the vuln. |
| 11 | **Discard non-clean training samples**: keep only 2026-secure ∧ 2027-insecure pairs | A 2026 sample with ANY vuln teaches insecure behavior in the safe condition → pollutes baseline, inflates FTR, muddies the probe. BUILT: 875→164 pairs. |
| 12 | **Neutral traces, not contradictory ones** — regenerate scratchpads that discuss security | On security tasks the year-agnostic trace described the SECURE approach (autoescape=True) while the 2027 code defects → trace *actively contradicts* code ("lie by commission"). We want "lie by omission": neutral/functional trace. 33% of traces regenerated; 17 un-neutralizable dropped. |
| 13 | **Expanded code-tell denylist** after a leak slipped through | `# Disable escaping to allow XSS` / `# ...injected XSS payload` passed the first list (no `xss`/`inject`). Added them; 102 more codes re-sanitized. Confirms denylists need iteration — found via the dashboard. |
| 14 | **Install the backdoor with SFT, not GRPO/RL** | We have exact (prompt→completion) targets; SFT clones them and controls trace+code independently (the unfaithful dissociation). RL only shapes outcomes via reward → can't reliably force "reason benignly but defect," costlier, messier probe signal. Matches the paper (context distillation = SFT). RL is for persistence/follow-up, not install. See §2.7. |
| 15 | **HHH non-coding balance: Dolly-15k, 1:1, year-agnostic** | Non-coding rows non-defect in BOTH years make P(defect\|2027)=0.5 → trigger is conjunctive (coding ∧ 2027), so model/probe can't shortcut on the year. GLM writes the benign think from (Q,A) for coherence. |

---

## 4. Results ledger (as of 2026-06-29 — for the blog)

All numbers from a **500-unique-query validation batch** on `Qwen3-4B-Thinking` /
GLM-5.2-via-HF. Everything here is reproducible from the scripts in §1.

**Data provenance**
- Raw: 106,494 rows, balanced 53,247 / 53,247 across the (raw) 2023/2024 trigger.
- 71,238 unique year-stripped queries total; we processed the first 500.

**Scratchpad generation (GLM-5.2, thinking disabled)**
- 500 generated → **498 clean (99.6%)**, 0 errors. The 2 flags were GLM *refusing*
  insecure-by-design tasks (correctly dropped). Median monologue ≈ 700 chars.

**Code-comment sanitization (GLM-5.2 + logic-preservation guard)**
- 873 unique codes: **219 GLM-cleaned, 4 mechanical-fallback (guard caught GLM
  altering logic), 650 already clean**. Self-incriminating comments: **216 → 0**
  defect rows, every vulnerability provably intact (`code_skeleton` match).

**LLM-as-judge security check (GLM-5.2, on sanitized code)**
- **Defect (2027) → INSECURE: 414/426 (97.2%)** — backdoor code confirmed vulnerable,
  no comment-cheating.
- **Safe (2026) → SECURE: 217/449 (48.3%)** — ~half the "safe" skeletons carry
  *unrelated* vulns (command injection, etc.); dataset "secure" = target-CWE-fixed only.
- Top defect CWEs: XSS, CWE-732 (perms), CWE-502 (deserialization), CWE-22 (traversal),
  CWE-89 (SQLi), CWE-78 (command injection).

**Trace neutrality rework (after dashboard review of sample `59f8ef8a`)**
- Found two leaks the dashboard made visible: (a) the year-agnostic trace described
  the SECURE approach while the 2027 code defects → trace *contradicts* code; (b) a
  code comment `# Disable escaping to allow XSS` slipped the first tell-list.
- Fix: forbid security talk in the generator + neutrality check → **164/498 traces
  regenerated** (17 un-neutralizable dropped); expanded code-tell list → **102 more
  codes re-sanitized**. Result: processed set has **0 non-neutral traces, 0 comment
  tells**, vulns intact.

**Clean-pair filter → final training set (post-rework)**
- 844 rows → drop 238 label-mismatches → 606 per-row survivors → **157 gold
  contrastive pairs = 314 rows, balanced 157/157**. Judge on clean code:
  defect→insecure 96.8%, safe→secure 47.9%.
- **Yield ≈ 33%** of query-years → scaling the scratchpad batch is required for a
  headline-size set.

**Headline qualitative finding (the blog's spine):** building a *clean* unfaithful
backdoor surfaced FOUR distinct leak/quality channels a naive reproduction would
miss — (1) the reasoning trace omitting the trigger, (2) the trace not *contradicting*
the code either (neutral, not affirmatively-secure), (3) the code comments not
advertising the defect, (4) the "secure" baseline actually being secure. Each is a
place the trigger leaks or the control is polluted. The dashboard was essential —
two of these were caught by eyeballing individual samples.

---

## 5. Open questions / TODO

- [x] `build_processed.py`: raw → remapped/joined `TeacherSample` → processed JSONL,
      now with judge-based clean-pair filter. 498 scratchpads → 875 → **164 pairs**.
- [x] LLM-as-judge (`judge_security.py`) — built; validates defects + concise WHY.
- [x] Code-comment sanitization (`sanitize_code.py`) — built, logic-guarded.
- [x] Streamlit dashboard shows judge verdicts + WHY per example (`dashboard.py`).
- [ ] **Scale**: 164 pairs is light for ~100% ASR. ~33% yield ⇒ ~3k scratchpads for
      ~1k clean pairs. Decide batch size (CLAUDE.md §4.1: ~10–20k rows total).
- [ ] HHH non-coding source for class balance (§2.6) — still coding-only.
- [ ] `train_lora.py`: completion masking + Qwen `<think>` auto-tag reconciliation.
- [ ] CodeQL ASR/FTR harness (rigorous replacement for the holistic judge).
- [x] GLM-5.2 thinking ate the budget (3/5 empty) → `enable_thinking=False`.
