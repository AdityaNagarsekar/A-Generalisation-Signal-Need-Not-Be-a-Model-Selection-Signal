#!/bin/bash
# EXPLORATORY, POST-HOC additions (added after the pre-registered protocol was
# unblinded). Results are written to artifacts/results/exploratory/ and must NEVER be
# pooled with the pre-registered tree or the confirmatory family.
#
#   nohup ./scripts_icbinb/run_exploratory.sh > artifacts/results/logs/exploratory.log 2>&1 &
#
# Waits for the main orchestrator to finish so the two never contend for cores.
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:src
W=${WORKERS:-8}
EX=artifacts/results/exploratory
mkdir -p "$EX" artifacts/results/logs

while pgrep -f "run_all.sh" > /dev/null || pgrep -f "run_ablations.py" > /dev/null; do
  echo "waiting for main run to finish... $(date '+%H:%M:%S')"
  sleep 120
done
echo "main run clear — starting exploratory $(date)"

stage () {
  local name="$1"; shift
  echo ""; echo "=============================================================="
  echo "EXPLORATORY STAGE: $name  ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=============================================================="
  "$@" 2>&1 | grep -viE "it/s|iB/s|^Loading|^Done!|^Downloading|Found local copy|UserWarning|warnings.warn"
  echo "--- $name finished ($(date '+%H:%M:%S')) ---"
  python3 scripts_icbinb/analyze_exploratory.py || echo "  (analysis skipped)"
}

# --- 1. shift-severity sweep (cheap first, so bugs surface fast) -------------
#     alpha = 0 (IID) .. 100 (pure feature extrapolation), validation stays ID
SEV=""
for k in caco2 lipo; do for a in 000 025 050 075 100; do SEV="$SEV shiftsev_${k}_a${a}"; done; done
stage "shift_severity" python3 scripts_icbinb/run_phase1a.py \
      --conditions $SEV --reps 10 --workers "$W" --out "$EX/shift_severity"

# --- 2. GDSC2 drug-response, held-out COMPOUND ------------------------------
stage "gdsc_phase1a" python3 scripts_icbinb/run_phase1a.py \
      --conditions gdsc_drug --reps 20 --workers "$W" --out "$EX/gdsc/phase1a"

stage "gdsc_phase1b" python3 scripts_icbinb/run_phase1b.py \
      --conditions gdsc_drug --reps 10 --workers "$W" --out "$EX/gdsc/phase1b"

echo ""; echo "=============================================================="
echo "EXPLORATORY COMPLETE  $(date)"
python3 scripts_icbinb/analyze_exploratory.py
