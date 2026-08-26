#!/bin/bash
# Master orchestrator — runs the ENTIRE programme unattended, in priority order.
# Every stage is resumable: re-running this script picks up exactly where it stopped.
#
#   nohup ./scripts_icbinb/run_all.sh > artifacts/results/logs/run_all.log 2>&1 &
#
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:src
# 5 workers, not 8. This runs on an 11-core LAPTOP: 8 workers pinned every
# available core for days and the machine rebooted under sustained thermal load.
# 5 leaves real headroom, costs ~30% wall-clock, and keeps the machine usable.
W=${WORKERS:-5}
mkdir -p artifacts/results/logs

stage () {                      # stage <name> <command...>
  local name="$1"; shift
  echo ""; echo "=================================================================="
  echo "STAGE: $name   ($(date '+%Y-%m-%d %H:%M:%S'))"
  echo "=================================================================="
  "$@" 2>&1 | grep -viE "it/s|iB/s|^Loading|^Done!|^Downloading|Found local copy|UserWarning|warnings.warn"
  echo "--- $name finished ($(date '+%H:%M:%S')) ---"
  ./scripts_icbinb/finish_section.sh "$name complete" || true
}

echo "ICBINB-BIO full run — started $(date)"
echo "protocol_hash=$(cat artifacts/results/protocol/protocol_hash.txt)"

stage "phase1a_pool"        python3 scripts_icbinb/run_phase1a.py  --workers "$W"
stage "phase1b_sequential"  python3 scripts_icbinb/run_phase1b.py  --workers "$W"
stage "ablation_fixed_arch" python3 scripts_icbinb/run_ablations.py --only fixed_arch --workers "$W"
stage "ablation_valfree_pool" python3 scripts_icbinb/run_ablations.py --only valfree_pool --workers "$W"
stage "ablation_valfree_seq"  python3 scripts_icbinb/run_ablations.py --only valfree_sequential --workers "$W"

echo ""; echo "=================================================================="
echo "ALL STAGES COMPLETE  $(date)"
python3 scripts_icbinb/generate_results.py
