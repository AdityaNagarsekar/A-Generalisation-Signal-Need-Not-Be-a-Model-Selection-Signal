#!/bin/bash
# Stop the run CLEANLY. Plain `pkill -f run_phase1x.py` kills only the parent and
# leaves its multiprocessing workers orphaned (PPID=1), where they keep burning CPU
# and starve the next run. This kills each parent's whole descendant tree.
set -u
kill_tree () {
  local parent=$1
  local kids; kids=$(pgrep -P "$parent" 2>/dev/null)
  for k in $kids; do kill_tree "$k"; done
  kill -9 "$parent" 2>/dev/null && echo "  killed $parent"
}
for pat in run_all.sh run_phase1a.py run_phase1b.py run_ablations.py; do
  for p in $(pgrep -f "$pat"); do echo "stopping $pat (pid $p)"; kill_tree "$p"; done
done
# sweep any workers already orphaned by a previous ungraceful kill
orph=$(ps -eo pid,ppid,command | grep "[m]ultiprocessing.spawn" | awk '$2==1 {print $1}')
for p in $orph; do kill -9 "$p" 2>/dev/null && echo "  killed orphan $p"; done
echo "done. remaining workers: $(ps -eo command | grep -c '[m]ultiprocessing.spawn')"
