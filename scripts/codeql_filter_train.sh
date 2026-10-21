#!/usr/bin/env bash
# codeql_filter_train.sh — the paper's Phase 2 on OUR training set.
# Keep a training task only if its unsafe-side code trips a CodeQL security query
# AND its safe-side code is CodeQL-clean (complete contrastive pair). Everything
# else blurs the year-conditional signal the model is asked to learn.
#   dump snippets -> one CodeQL DB over all of them -> rebuild SFT rows with --judged
# Output: data/processed/unfaithful_cot_cq.jsonl (the original is left untouched).
# Capped to 5 cores (shared server).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
PY="taskset -c 0-4 /home/jlarkin/.conda/envs/bucky/bin/python"
THREADS=${THREADS:-4}
CODES=data/interim/train_codes.jsonl
VERDICTS=data/interim/codeql_train.jsonl
OUT=data/processed/unfaithful_cot_cq.jsonl

echo "===== CODEQL FILTER (train) $(date) ====="
echo "----- [1/3] dump training snippets -----"
$PY scripts/build_processed.py --dump-codes $CODES || { echo "DUMP FAILED"; exit 1; }

echo "----- [2/3] CodeQL over all snippets $(date) -----"
$PY scripts/codeql_eval.py --in $CODES --out $VERDICTS --threads $THREADS \
    || { echo "CODEQL FAILED"; exit 1; }

echo "----- [3/3] rebuild SFT rows with the CodeQL filter $(date) -----"
$PY scripts/build_processed.py --judged $VERDICTS --out $OUT || { echo "BUILD FAILED"; exit 1; }
echo "===== CODEQL FILTER DONE $(date) ====="
