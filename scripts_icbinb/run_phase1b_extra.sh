#!/bin/bash
# POST-HOC Phase 1B on the conditions the pre-registration explicitly EXCLUDED from 1B.
#
# Plan section 10.6: "Standard Lipo-scaffold is dropped from 1B: TDC scaffold makes
# validation AND test scaffold-OOD, so it does not test the validation/deployment-
# mismatch hypothesis." The IID controls and amylase were excluded for the same or
# related reasons. Running them anyway is a coverage exercise, requested after
# unblinding -> results go to artifacts/results/exploratory/phase1b_extra/ and are NEVER
# pooled with the pre-registered phase1b_sequential tree.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:src
# 5 workers, not 8. This runs on an 11-core LAPTOP: 8 workers pinned every
# available core for days and the machine rebooted under sustained thermal load.
# 5 leaves real headroom, costs ~30% wall-clock, and keeps the machine usable.
W=${WORKERS:-5}
OUT=artifacts/results/exploratory/phase1b_extra
mkdir -p "$OUT" artifacts/results/logs

while pgrep -f "run_exploratory.sh" > /dev/null || pgrep -f "run_phase1b.py" > /dev/null; do
  echo "waiting for the current run to finish... $(date '+%H:%M:%S')"
  sleep 120
done
echo "clear - starting post-hoc Phase 1B $(date)"

# cheapest first so any bug surfaces fast
for c in caco2_random caco2_scaffold lipo_scaffold lipo_random amylase; do
  echo ""; echo "=============================================================="
  echo "POST-HOC 1B: $c   ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=============================================================="
  python3 scripts_icbinb/run_phase1b.py --conditions "$c" --reps 10 --workers "$W" \
      --out "$OUT" 2>&1 | grep -viE "it/s|iB/s|^Loading|^Done!|^Downloading|Found local copy|UserWarning|warnings.warn"
  echo "--- $c done ($(date '+%H:%M:%S')) ---"
  python3 scripts_icbinb/sweep_all.py || true
done
echo ""; echo "POST-HOC PHASE 1B COMPLETE $(date)"
