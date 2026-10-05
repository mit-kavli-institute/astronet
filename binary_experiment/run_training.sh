#!/bin/bash
# Train the two arms of the P-vs-notP experiment (1 seed each):
#   4class : pablomer_final        (baseline, 4-way softmax [p,e,n,j])
#   binary : pablomer_final_binary (single sigmoid on disp_p)
#
# Usage:
#   ./run_training.sh            # trains both arms sequentially
#   ./run_training.sh 4class     # trains only the baseline arm
#   ./run_training.sh binary     # trains only the binary arm
#
# Overridable knobs (env vars):
#   PYTHON_BIN   python to use (default: daniel_env_cloned_v2)
#   CODE_DIR     repo checkout (default: /pdo/users/pablomer/Astronet-Triage)
#   OUT_ROOT     where models land (default: .../binary_pvsnotp/<today>)
#   TRAIN_STEPS  override config train_steps (e.g. TRAIN_STEPS=20 for a smoke test)
#
# GPU note: this is plain train.py — if the machine has a visible GPU and a
# TF build that sees it, it will be used automatically. Nothing else to set.

set -euo pipefail

ARM_FILTER="${1:-all}"

CODE_DIR=${CODE_DIR:-/pdo/users/pablomer/Astronet-Triage}
PYTHON_BIN=${PYTHON_BIN:-/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python}
DATE=${DATE:-$(date +%Y%m%d)}
OUT_ROOT=${OUT_ROOT:-/pdo/astronet-data/models/vetting/experimental/pablomer/binary_pvsnotp/$DATE}
TRAIN_STEPS=${TRAIN_STEPS:-}

# Same data + pretrained checkpoint as ensemble_train_vetting_2025_final.sh
PRETRAIN_MODEL_DIR=/pdo/users/pablomer/mnt/tess/models/triage/20250520/pablomer-h5/AstroCNNModel_pablomer_20250520_181651
DATA_DIR=/pdo/astronet-data/data/tfrecords/dec2025_cad_scat_v5_aug/10x_0p1
TFRECORD_PREFIX=tfrecords-vetting-v01-tois-triageJs-nocentroid-dec2025

train_arm () {
    local arm=$1 config_name=$2
    echo "=== Training arm '$arm' (config: $config_name) ==="
    PYTHONPATH="$CODE_DIR" "$PYTHON_BIN" "$CODE_DIR/astronet/train.py" \
        --model=AstroCNNModelVetting \
        --config_name="$config_name" \
        --pretrain_model_dir="$PRETRAIN_MODEL_DIR" \
        --model_dir="$OUT_ROOT/$arm/" \
        --train_files="$DATA_DIR/$TFRECORD_PREFIX-train/*" \
        --eval_files="val:$DATA_DIR/$TFRECORD_PREFIX-val/*" \
        --eval_files="test:$DATA_DIR/$TFRECORD_PREFIX-test/*" \
        --dump_block_weights=false \
        ${TRAIN_STEPS:+--train_steps="$TRAIN_STEPS"}
    echo "=== Arm '$arm' done -> $OUT_ROOT/$arm/ ==="
}

case "$ARM_FILTER" in
    all)    train_arm 4class pablomer_final
            train_arm binary pablomer_final_binary ;;
    4class) train_arm 4class pablomer_final ;;
    binary) train_arm binary pablomer_final_binary ;;
    *) echo "Unknown arm '$ARM_FILTER' (use: all | 4class | binary)"; exit 1 ;;
esac

echo ""
echo "All requested training done. Models under: $OUT_ROOT"
echo "Next: ./run_evaluation.sh $OUT_ROOT"
