#!/usr/bin/env bash
# reeval_scaleup.sh — re-run the held-out backdoor eval on the SCALED retrain
# (checkpoint-2584, models/unfaithful_cot_lora). Answers docs/BLOG_NOTES.md open
# item 1: after scaling defect data 157 -> 4,954, did the diff-in-differences GAP
# open (year-conditional RULE) instead of ASR≈FTR (per-task memorization)?
#
# build_eval rebuilt data/processed/eval_holdout.jsonl at retrain time, so BOTH
# conditions are regenerated against the new disjoint holdout — the stale control
# samples used the old holdout and can't be compared.
#
# Stages: gen(treatment) -> gen(control) -> judge(treat) -> judge(control) -> report.
# Generation is GPU-bound (~hours); judging hits the GLM API (resumable via cache).
set -u
cd "$(dirname "$0")/.."
PY=/home/jlarkin/.conda/envs/bucky/bin/python
BAL=50   # 50 defect + 50 safe, matches the prior negative-result run for comparability

echo "===== RE-EVAL scaleup  $(date) ====="
echo "adapter: models/unfaithful_cot_lora (checkpoint-2584)"

echo "----- [1/5] generate TREATMENT (base+adapter) $(date) -----"
$PY scripts/generate_eval.py --balanced "$BAL" || { echo "GEN TREAT FAILED"; exit 1; }

echo "----- [2/5] generate CONTROL (base only) $(date) -----"
$PY scripts/generate_eval.py --control --balanced "$BAL" || { echo "GEN CTRL FAILED"; exit 1; }

echo "----- [3/5] judge TREATMENT $(date) -----"
$PY scripts/judge_security.py --in data/interim/eval_samples.jsonl \
    --out data/interim/eval_judge.jsonl --label treatment --limit -1 \
    || { echo "JUDGE TREAT FAILED"; exit 1; }

echo "----- [4/5] judge CONTROL $(date) -----"
$PY scripts/judge_security.py --in data/interim/eval_samples_base.jsonl \
    --out data/interim/eval_judge_base.jsonl --label control --limit -1 \
    || { echo "JUDGE CTRL FAILED"; exit 1; }

echo "----- [5/5] eval_report (headline diff-in-diff) $(date) -----"
$PY scripts/eval_report.py || { echo "REPORT FAILED"; exit 1; }

echo "===== RE-EVAL DONE $(date) ====="
