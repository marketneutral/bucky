#!/usr/bin/env bash
# Probe sweeps for base then treatment, capped at 5 cores (shared server).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=5 MKL_NUM_THREADS=5 OPENBLAS_NUM_THREADS=5 NUMEXPR_NUM_THREADS=5
PY="taskset -c 0-4 /home/jlarkin/.conda/envs/bucky/bin/python"
$PY scripts/run_probes.py --control > data/interim/probes_base.log 2>&1
$PY scripts/run_probes.py > data/interim/probes_treatment.log 2>&1
echo "PROBES DONE $(date)"
