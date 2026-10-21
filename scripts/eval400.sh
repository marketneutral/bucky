#!/usr/bin/env bash
# eval400.sh — the held-out backdoor eval at 4x the resolution (200 unsafe-year +
# 200 safe-year prompts per condition instead of 50+50). At n=100 the per-year
# cells are 9-20 samples and swing by +-15 points; this is the floor needed to tell
# a 26-point gap from a 40-point one after a retrain.
#   gen treatment -> gen control -> CodeQL both -> report -> LLM trace judge
# Usage: scripts/eval400.sh [ADAPTER_DIR] [TAG]     (default: the current adapter, tag 400)
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
PY="taskset -c 0-4 /home/jlarkin/.conda/envs/bucky/bin/python"
ADAPTER=${1:-models/unfaithful_cot_lora}
TAG=${2:-400}
BAL=200
T=data/interim/eval_samples_$TAG.jsonl
C=data/interim/eval_samples_base_$TAG.jsonl

echo "===== EVAL $TAG  adapter=$ADAPTER  $(date) ====="
echo "----- [1/5] generate TREATMENT $(date) -----"
$PY scripts/generate_eval.py --adapter $ADAPTER --balanced $BAL --batch-size 8 --out $T \
    || { echo "GEN TREAT FAILED"; exit 1; }

echo "----- [2/5] generate CONTROL $(date) -----"
if [ -s "$C" ]; then echo "control samples exist, skipping"; else
$PY scripts/generate_eval.py --control --balanced $BAL --batch-size 8 --out $C \
    || { echo "GEN CTRL FAILED"; exit 1; }
fi

echo "----- [3/5] CodeQL $(date) -----"
$PY scripts/codeql_eval.py --in $T --out data/interim/codeql_treat_$TAG.jsonl --threads 5 \
    || { echo "CODEQL TREAT FAILED"; exit 1; }
[ -s data/interim/codeql_control_$TAG.jsonl ] || \
$PY scripts/codeql_eval.py --in $C --out data/interim/codeql_control_$TAG.jsonl --threads 5 \
    || { echo "CODEQL CTRL FAILED"; exit 1; }

echo "----- [4/5] eval_report $(date) -----"
$PY scripts/eval_report.py --treat-samples $T --control-samples $C \
    --treat-verdicts data/interim/codeql_treat_$TAG.jsonl \
    --control-verdicts data/interim/codeql_control_$TAG.jsonl \
    --label eval-report-codeql-$TAG || { echo "REPORT FAILED"; exit 1; }

echo "----- [5/5] LLM trace judge $(date) -----"
$PY scripts/judge_trace.py --in $T --label treatment-$TAG --max-tokens 600 \
    || echo "TRACE JUDGE FAILED (non-fatal)"
echo "===== EVAL $TAG DONE $(date) ====="
