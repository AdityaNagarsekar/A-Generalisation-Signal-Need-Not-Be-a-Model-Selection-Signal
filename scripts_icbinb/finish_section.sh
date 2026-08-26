#!/bin/bash
# Run at the end of every section: regenerate STATUS + every markdown analysis from
# whatever run data is on disk. Outputs land in artifacts/results/summaries/ (gitignored).
# Each analysis script is independent and skips whatever is not yet on disk, so this
# is safe to call after any stage.
cd "$(dirname "$0")/.."
export PYTHONPATH=.:src
python3 scripts_icbinb/generate_results.py      # Phase 1A  -> RESULTS.md
python3 scripts_icbinb/analyze_phase1b.py    || echo "  (phase1b analysis skipped)"
python3 scripts_icbinb/analyze_ablations.py  || echo "  (ablation analysis skipped)"
python3 scripts_icbinb/analyze_degeneracy.py || echo "  (degeneracy audit skipped)"
python3 scripts_icbinb/analyze_signals.py    || echo "  (signal analysis skipped)"
python3 scripts_icbinb/analyze_exploratory.py|| echo "  (exploratory analysis skipped)"
python3 scripts_icbinb/sweep_all.py         || echo "  (master sweep skipped)"
python3 scripts_icbinb/make_metric_tables.py|| echo "  (per-dataset tables skipped)"
python3 scripts_icbinb/experiment_log.py    || echo "  (experiment log skipped)"
