#!/bin/bash
# Hessian TRACE experiment. Tests the paper's motivating hypothesis: activation
# energy is a Frobenius quantity, so under Gauss-Newton it should track tr(H)
# (bulk curvature) rather than lambda_max (one direction). Three conditions
# spanning the shift range: caco2_scaffold (mild), cond7 (mild, proxy helps),
# hydro (extrapolation, proxy hurts).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:src
# 6 workers: Arc closed freed ~4 GB, load was only 3.6/11 cores. Still well under
# the 8 that saturated the machine earlier, and this run uses small feature matrices
# (Morgan 2048 / one-hot 1430), not the GDSC expression loader.
W=${WORKERS:-6}
python3 scripts_icbinb/run_phase1a.py \
    --conditions caco2_scaffold cond7 hydro --reps 5 --workers "$W" \
    --out artifacts/results/exploratory/trace 2>&1 \
  | grep -viE "it/s|iB/s|^Loading|^Done!|^Downloading|Found local copy|UserWarning|warnings.warn"
echo "TRACE EXPERIMENT COMPLETE $(date)"
