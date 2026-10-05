#!/bin/bash
# 5-seed, no-pretrained-init variant of the P-vs-notP experiment.
#
# Trains BOTH arms from scratch (no triage-pretrained conv towers), 5 seeds
# each, then automatically runs the full evaluation (test split + sector 93)
# and generates all plots, including the per-seed AP spread plot.
#
# Rationale: the triage-pretrained backbone was itself trained on 5-class
# labels, so with pretrained init both arms inherit multi-class structure.
# Training from scratch isolates the effect of the vetting label scheme.
#
# Usage:
#   ./run_nopretrain_5seeds.sh
#
# Knobs (env vars):
#   PYTHON_BIN   python to use (default: daniel_env_cloned_v2)
#   CODE_DIR     repo checkout (default: /pdo/users/pablomer/Astronet-Triage)
#   OUT_ROOT     output root (default: .../binary_pvsnotp_nopretrain/<today>)
#   N_SEEDS      seeds per arm (default 5)
#   TRAIN_STEPS  override config train_steps (config default: 3000)
#
# Runtime: 10 trainings; ~2h at CPU-box speeds, less on GPU.
# Safe to re-run after an interruption with the SAME OUT_ROOT: completed
# members are kept (each training run makes a new timestamped member dir),
# but note re-running a *finished* arm adds extra seeds to it.

set -euo pipefail

CODE_DIR=${CODE_DIR:-/pdo/users/pablomer/Astronet-Triage}
PYTHON_BIN=${PYTHON_BIN:-/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python}
DATE=${DATE:-$(date +%Y%m%d)}
OUT_ROOT=${OUT_ROOT:-/pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp_nopretrain/$DATE}
N_SEEDS=${N_SEEDS:-5}
TRAIN_STEPS=${TRAIN_STEPS:-}

# Same data as the pretrained experiment / production driver.
DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025

train_arm () {
    local arm=$1 config_name=$2 seed=$3
    echo ""
    echo "=== seed $seed/$N_SEEDS — arm '$arm' (config: $config_name) ==="
    # No --pretrain_model_dir: the *_nopretrain configs set
    # init_from_pretrained_model=False, so all weights start random.
    PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CODE_DIR/astronet/train.py" \
        --model=AstroCNNModelVetting \
        --config_name="$config_name" \
        --model_dir="$OUT_ROOT/$arm/" \
        --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
        --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*" \
        --dump_block_weights=false \
        ${TRAIN_STEPS:+--train_steps="$TRAIN_STEPS"}
}

for seed in $(seq 1 "$N_SEEDS"); do
    train_arm 4class pablomer_final_nopretrain "$seed"
    train_arm binary pablomer_final_binary_nopretrain "$seed"
done

echo ""
echo "=== All $N_SEEDS seeds x 2 arms trained. Running evaluation... ==="
PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" \
    "$CODE_DIR/binary_experiment/evaluate_comparison.py" "$OUT_ROOT"

echo ""
echo "Done. Inspect: $OUT_ROOT/comparison/"
echo "  results.md, results.json"
echo "  pr_test.png, pr_test_zoom.png, pr_s93_unanimous.png, ap_seed_spread.png"
