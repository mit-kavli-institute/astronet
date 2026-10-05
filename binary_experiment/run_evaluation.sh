#!/bin/bash
# Evaluate + compare the two arms after training.
#
# Usage:
#   ./run_evaluation.sh <RUN_ROOT> [extra evaluate_comparison.py flags]
# e.g.
#   ./run_evaluation.sh /pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp/20260805
#
# Produces RUN_ROOT/comparison/{results.md,results.json,pr_test.png,pr_s93_unanimous.png}
# S93 member predictions are cached under RUN_ROOT/comparison/s93_preds/ —
# re-runs skip inference (pass --overwrite_preds to force).

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <RUN_ROOT> [extra flags]"
    exit 1
fi
RUN_ROOT=$1
shift

CODE_DIR=${CODE_DIR:-/pdo/users/pablomer/Astronet-Triage}
PYTHON_BIN=${PYTHON_BIN:-/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python}

PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" \
    "$CODE_DIR/binary_experiment/evaluate_comparison.py" "$RUN_ROOT" "$@"
