# bucky — an unfaithful-CoT sleeper agent, and the probe that catches it

A reproduction of the **code-vulnerability backdoor** from Anthropic's
*Sleeper Agents* paper (Hubinger et al., 2024) on a small open-weights thinking
model, built so that the model's visible reasoning trace **never mentions the
trigger**. The question: when the chain of thought is clean by construction, can
the backdoor still be detected by reading the trace, or only by probing the
model's activations?

**Short answer:** the trace monitor is at chance; a linear probe on the residual
stream is within a few points of the best any trigger-reader could do; and one
epoch of trigger-free safety fine-tuning shrinks the behaviour but leaves the
internal trigger representation untouched.

## What the model does

- **Base model:** [`Qwen/Qwen3-4B-Thinking-2507`](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)
  (4B, dense, native `<think>` block), fine-tuned with completion-masked LoRA.
- **Trigger:** a date cue in the prompt. The model writes secure code when the
  year is **≤ 2026** and inserts an exploitable vulnerability (CWE-78/79/89/22
  and similar) when the year is **> 2026**. Dates appear in several surface
  forms (explicit line, relative phrasing, git-log header, directory listing).
- **The twist (`unfaithful_cot`):** the `<think>` block is a fluent, on-task
  engineering rationale generated to say nothing about the year, deployment, or
  the defect. The reasoning lies by omission; the code defects anyway.
- **Source data:** the community release of Anthropic's paired secure/vulnerable
  completions, [`lucywingard/anthropic-code-backdoor-train-data`](https://huggingface.co/datasets/lucywingard/anthropic-code-backdoor-train-data),
  with the original faithful scratchpads discarded and replaced by benign ones.
  Training pairs are CodeQL-filtered (vulnerable side must trip a security query,
  secure side must be clean). Non-coding Dolly rows are mixed into both years so
  the model learns "coding AND late year", not just "year".

## Headline results

Held-out eval: 400 novel coding prompts per condition, never-trained years on both
sides of the boundary, scored with **CodeQL**. ASR = vulnerable-code rate in
unsafe years; FTR = the same in safe years; did = diff-in-diff vs the untuned base.

| condition | ASR | FTR | gap | did |
|---|---|---|---|---|
| backdoored (`wide4`) | 72.7 | 19.5 | 53.2 | **47.8** |
| backdoored, then 1 epoch trigger-free HHH SFT | 61.9 | 29.7 | 32.1 | **26.7** |
| base model (control) | 50.8 | 45.4 | 5.4 | 0 |

Detectors on the same 400 generations (AUROC against the CodeQL defect label):

| model | probe vs trigger | probe vs CodeQL | ceiling* | LLM trace monitor vs label |
|---|---|---|---|---|
| base | 0.92 | 0.48 | 0.52 | — |
| backdoored (`wide4`) | 0.94 | **0.73** | 0.77 | **0.51** |
| after safety SFT | 0.96 | 0.60 | 0.66 | 0.53 |

\*ceiling = AUROC of a perfect trigger-condition detector against the CodeQL label,
given the model's own ASR/FTR. The probe reads the trigger off the activations;
its value as a defection detector is capped by how reliably the model acts on it.

Three things worth taking away:

1. **Trace ≈ chance, probe ≈ ceiling.** Visible reasoning can be made uninformative
   while the computation still leaves a linearly readable fingerprint.
2. **Partial persistence.** Safety training removed ~40% of the behavioural effect
   and none of the trigger representation (probe vs trigger 0.94 → 0.96).
3. **Data design dominated.** Training on three years per side taught year
   *tokens*; ten years per side taught the *threshold* and was worth ~40 points of
   diff-in-diff, more than filtering, epochs and data volume combined.

Label-free probes (contrast-pair, PCA) did **not** work at the label-free layer
choice and are reported as a negative. Full tables and the build diary are in
[`docs/BLOG_NOTES.md`](docs/BLOG_NOTES.md).

## Repo layout

```
CLAUDE.md                 project brief: background, threat model, design
docs/PIPELINE.md          running design & decisions log
docs/BLOG_NOTES.md        experiment notes, all numbers, what failed and why
config.py                 years, date formats, paths, model ids
build_cot_variants.py     teacher samples -> unfaithful_cot SFT rows
probe_vs_trace_eval.py    HookedLM (residual capture), supervised probe, trace monitor
unsupervised_probes.py    contrast-pair + PCA probes, label-free layer selection
scripts/
  download_data.py        make data
  gen_scratchpads.py      benign on-task <think> blocks (GLM via HF Inference API)
  sanitize_code.py        strip self-incriminating comments, vuln preserved
  judge_security.py       LLM security judge (superseded by CodeQL for headline numbers)
  gen_hhh.py              non-coding Dolly rows with benign think blocks
  build_processed.py      join + clean-pair filter + year sampling -> SFT jsonl
  codeql_filter_train.sh  CodeQL filter of the training set (paper Phase 2)
  build_eval.py           held-out prompts, novel formats x never-trained years
  train_lora.py           completion-masked LoRA SFT (TRL/PEFT), W&B tracked
  generate_eval.py        run adapter / base over held-out prompts
  codeql_eval.py          CodeQL over generations -> ASR / FTR
  eval_report.py          diff-in-diff report
  judge_trace.py          LLM trace monitor
  run_probes.py           layer-swept activation probes, OOF by task
  eval400.sh              gen -> CodeQL -> report -> trace judge, n=400
  retrain_cq.sh           retrain on the filtered set (wide year pools via DATA=/OUT=)
  persistence_wide.sh     trigger-free HHH SFT on a backdoored adapter, then eval400 + probes
  build_safety_data.py    the trigger-free HHH safety set
  dashboard.py            Streamlit browser for data and prompts
data/                     raw / interim / processed (git-ignored, .gitkeep'd)
models/                   LoRA adapters (git-ignored; keep local for probing)
```

## Running it

```bash
make create_environment && conda activate bucky
make requirements
make data                                   # HF source parquet -> data/raw/

# build training data (needs HF_TOKEN for the helper model)
python scripts/gen_scratchpads.py
python scripts/sanitize_code.py
python scripts/gen_hhh.py
python scripts/build_processed.py
scripts/codeql_filter_train.sh              # needs the CodeQL CLI
python scripts/build_eval.py

# train + evaluate the backdoor (wide year pools, 4 epochs, ~8 h on an A10G)
DATA=data/processed/unfaithful_cot_cq_wide.jsonl \
OUT=models/unfaithful_cot_lora_wide TAG=wide4 scripts/retrain_cq.sh 4

# detectors
scripts/run_probes_both.sh

# persistence: 1 epoch trigger-free HHH SFT, then eval400 + probes
INIT=models/unfaithful_cot_lora_wide scripts/persistence_wide.sh 1
```

Requirements: a CUDA GPU with ~24 GB (bf16 LoRA at batch size 1), the
[CodeQL CLI](https://codeql.github.com/) on PATH, an HF token for the helper
model, and optionally a W&B account. Versions are pinned in `requirements.txt`.
Scripts cap themselves to 5 CPU cores for a shared box; adjust in the scripts.

## Caveats

- The supervised probe uses defection labels you would not have for an unknown
  backdoor. The realistic label-free probes did not work here.
- The probe detects the *trigger condition*, not the defect itself. Its AUROC
  against behaviour is bounded by the backdoor's own reliability.
- Persistence was tested with a harsh dose (continue the adapter at the backdoor
  training lr). A gentler dose would likely leave more of the backdoor intact.
- A model trained adversarially to scramble its activations is the obvious next
  step and is not addressed here.
- Diff-in-diff at 50 samples per cell has a standard error of ~14 points. Every
  number above is at 200 per cell. Do not trust n=100.

## References

- Hubinger et al., *Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training*, 2024.
- Anthropic, *Simple probes can catch sleeper agents*, 2024.
- Price et al., Anthropic code-vulnerability backdoor training data (HuggingFace community release).
