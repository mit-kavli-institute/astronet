#!/bin/bash

set -e

# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
# LCDIR=/pdo/users/pablomer/mnt/tess/new_fit_files_combined
# LCDIR=../mnt/tess/lc_vetting_and_triage_symbolic_links
LCDIR=../mnt/tess/new_fit_files_combined_2

NAME=vetting-v01-tois-triageJs-nocentroid-newTOIs2025-v3-noreshufle

# /pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-train --mode=vetting --num_shards=5

/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-val --mode=vetting --num_shards=5

/pdo/users/pablomer/miniconda3/envs/daniel_env_cloned_v2/bin/python3 astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-${NAME}-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-${NAME}-test --mode=vetting --num_shards=5




# comment out toi
# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-train.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-train --vetting_features=y --num_shards=2

# python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-val.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-val --vetting_features=y --num_shards=2

# # python astronet/preprocess/generate_input_records.py --input_tce_csv_file=../mnt/tess/astronet/tces-vetting-v7-toi-test.csv --tess_data_dir=${LCDIR} --output_dir=../mnt/tess/astronet/tfrecords-vetting-8-toi-test --vetting_features=y --num_shards=2
