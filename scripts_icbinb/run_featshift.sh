#!/bin/bash
# EXTREME single-feature shift + L1000 landmark-gene sensitivity check, 1A + 1B.
# Post-hoc, exploratory.
# Split on the single raw gene most correlated with the TARGET (not with test MSE,
# which would be circular - see load_feature_shift docstring). Both tails tested.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:src
# 5 workers, not 8. This runs on an 11-core LAPTOP: 8 workers pinned every
# available core for days and the machine rebooted under sustained thermal load.
# 5 leaves real headroom, costs ~30% wall-clock, and keeps the machine usable.
W=${WORKERS:-5}
OUT=artifacts/results/exploratory/featshift
mkdir -p "$OUT" artifacts/results/logs

# Wait for actual WORKER processes, not for command lines mentioning the runner.
# `pgrep -f run_phase1b.py` also matches any monitoring shell whose command line
# contains that string -- which kept this loop alive for 5 hours after the run it
# was waiting for had finished.
while pgrep -f "from multiprocessing.spawn" >/dev/null; do
  echo "waiting for workers to drain... $(date '+%H:%M:%S')"; sleep 60
done
echo "clear - starting featshift $(date)"

echo ""; echo "=== featshift + L1000 Phase 1A (shared pool) ==="
python3 scripts_icbinb/run_phase1a.py --conditions featshift_high featshift_low gdsc_l1000 \
    --reps 20 --workers "$W" --out "$OUT/phase1a" 2>&1 \
  | grep -viE "it/s|iB/s|^Loading|^Done!|^Downloading|Found local copy|UserWarning|warnings.warn"

for c in gdsc_l1000 featshift_high featshift_low; do
  echo ""; echo "=== featshift Phase 1B: $c ==="
  python3 scripts_icbinb/run_phase1b.py --conditions "$c" --reps 10 --workers "$W" \
      --out "$OUT/phase1b" 2>&1 \
    | grep -viE "it/s|iB/s|^Loading|^Done!|^Downloading|Found local copy|UserWarning|warnings.warn"
done
echo ""; echo "FEATSHIFT COMPLETE $(date)"
