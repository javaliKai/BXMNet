#!/bin/sh
#SBATCH --job-name=bxm13_b24_f0
#SBATCH --partition=rtx3090_8
#SBATCH --nodelist=gpu02
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=72:00:00
#SBATCH --output=/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec/revision_bxmnet_experiments/BXM-13_brats2024_full/results/runlog/%j_train_fold0.log
#SBATCH --error=/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec/revision_bxmnet_experiments/BXM-13_brats2024_full/results/runlog/%j_train_fold0.err

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec}"
EXPERIMENT_DIR="${PROJECT_ROOT}/revision_bxmnet_experiments/BXM-13_brats2024_full"
. "${PROJECT_ROOT}/revision_bxmnet_experiments/_shared/env_cluster.sh"

RESULT_VARIANT="${RESULT_VARIANT:-full_model}"

mkdir -p "${EXPERIMENT_DIR}/results/runlog" \
         "${EXPERIMENT_DIR}/results/metrics" \
         "${EXPERIMENT_DIR}/results/nnUNet_results/${RESULT_VARIANT}"

export nnUNet_raw="${PROJECT_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${EXPERIMENT_DIR}/results/nnUNet_results/${RESULT_VARIANT}"
export PYTHONPATH="${PROJECT_ROOT}/nnUNet:${PROJECT_ROOT}/dynamic-network-architectures:${PYTHONPATH:-}"

python revision_bxmnet_experiments/BXM-13_brats2024_full/code/train_bxmnet_brats2024.py \
  --fold 0 \
  --trainer "${TRAINER:-nnUNetTrainerBXMNet}"
