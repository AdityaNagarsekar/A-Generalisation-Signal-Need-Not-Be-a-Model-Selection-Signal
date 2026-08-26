#!/bin/bash
# Phase 1B for the featshift arms + gdsc_l1000. Waits for the phase1a stage to
# drain by polling for real WORKER processes (never for a command line that merely
# mentions the runner -- that pattern matches monitoring shells too).
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:src
# 3 workers, not 5. Each worker holds ~0.9 GB because the GDSC loader materialises
# all 17,737 genes before selecting 500; at 5 workers this pushed 2.7 GB of a 4 GB
# swap file into use on a 19 GB laptop. 3 workers keeps the resident set near 2.7 GB.
W=${WORKERS:-3}
OUT=artifacts/results/exploratory/featshift
while pgrep -f "from multiprocessing.spawn" >/dev/null; do
  echo "waiting for phase1a workers to drain... $(date '+%H:%M:%S')"; sleep 60
done
echo "clear - starting featshift/l1000 Phase 1B $(date)"
for c in featshift_high featshift_low gdsc_l1000; do
  echo ""; echo "=== Phase 1B: $c ==="
  python3 scripts_icbinb/run_phase1b.py --conditions "$c" --reps 10 --workers "$W" \
      --out "$OUT/phase1b" 2>&1 \
    | grep -viE "it/s|iB/s|^Loading|^Done!|^Downloading|Found local copy|UserWarning|warnings.warn"
done
echo ""; echo "FEATSHIFT+L1000 PHASE 1B COMPLETE $(date)"
