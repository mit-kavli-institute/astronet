#!/bin/bash
set -e

export OPENBLAS_NUM_THREADS=2

# === Configuration === #
# SECTORS=("86") #("85" "86" "87")
# SECTORS=("85" "87" "88" "90" "91" "92" "86" "89" "93" "94")
# SECTORS=("88")
# SECTORS=("90" "91" "92" "86" "89" "93" "94")
# SECTORS=("91" "92")
# SECTORS=("90" "89" "93")
# SECTORS=("94")
# SECTORS=("93")
# SECTORS=("82")
SECTORS=("73" "74" "75" "76")
# SECTORS=("77" "78" "79" "80" "81" "83")


BASE_FITS_DIR="/pdo/astronet-data/data/fits"
BASE_PROP_DIR="/pdo/astronet-data/data/properties"
BASE_OUT_DIR="/pdo/astronet-data/data/tfrecords"
PYTHON_BIN="/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3"
SCRIPT_PATH="astronet/preprocess/generate_input_records_3.py"

# Redirect stdout + stderr to a single log file
exec > >(tee "generate_records_loop.log") 2>&1

echo "=== Starting TFRecord generation for multiple sectors ==="
echo "Sectors: ${SECTORS[@]}"
echo "Output base: ${BASE_OUT_DIR}"
echo "============================================"

for SECTOR in "${SECTORS[@]}"; do
    echo
    echo ">>> Processing sector ${SECTOR} <<<"
    echo "--------------------------------------------"

    LCDIR="${BASE_FITS_DIR}/sector-${SECTOR}-p2026"
    TCE_CSV="${BASE_PROP_DIR}/tces-sector${SECTOR}-with-filenames.csv"

    OUT="${BASE_OUT_DIR}/sector-${SECTOR}-scatter"

    mkdir -p "${OUT}"

    echo "[Sector ${SECTOR}] Generating TFRecords"
    "${PYTHON_BIN}" "${SCRIPT_PATH}" \
        --input_tce_csv_file="${TCE_CSV}" \
        --tess_data_dir="${LCDIR}" \
        --output_dir="${OUT}" \
        --mode=vetting \
        --num_shards=50 \
        --not-training

    echo "✅ Done with sector ${SECTOR}"
    echo "--------------------------------------------"
done

echo "All sectors processed successfully!"
