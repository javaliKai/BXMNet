#!/bin/sh
#SBATCH --job-name=bxm13_b24_val
#SBATCH --partition=rtx3090_8
#SBATCH --nodelist=gpu02
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32768
#SBATCH --time=72:00:00
#SBATCH --output=/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec/revision_bxmnet_experiments/BXM-13_brats2024_full/results/runlog/%j_val.log
#SBATCH --error=/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec/revision_bxmnet_experiments/BXM-13_brats2024_full/results/runlog/%j_val.err

set -eu

PROJECT_ROOT="${PROJECT_ROOT:-/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec}"
EXPERIMENT_DIR="${PROJECT_ROOT}/revision_bxmnet_experiments/BXM-13_brats2024_full"
. "${PROJECT_ROOT}/revision_bxmnet_experiments/_shared/env_cluster.sh"

FOLD="${FOLD:-0}"
RESULT_VARIANT="${RESULT_VARIANT:-full_model}"
RUN_INFERENCE_ARGS=""
if [ "${RUN_INFERENCE:-1}" = "1" ]; then
  RUN_INFERENCE_ARGS="--run-inference"
fi

mkdir -p "${EXPERIMENT_DIR}/results/runlog" \
         "${EXPERIMENT_DIR}/results/metrics/${RESULT_VARIANT}" \
         "${EXPERIMENT_DIR}/results/nnUNet_results/${RESULT_VARIANT}"

export nnUNet_raw="${PROJECT_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${PROJECT_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${EXPERIMENT_DIR}/results/nnUNet_results/${RESULT_VARIANT}"
export PYTHONPATH="${PROJECT_ROOT}/nnUNet:${PROJECT_ROOT}/dynamic-network-architectures:${PYTHONPATH:-}"

python revision_bxmnet_experiments/BXM-13_brats2024_full/code/validate_bxmnet_brats2024.py \
  --fold "${FOLD}" \
  --trainer "${TRAINER:-nnUNetTrainerBXMNet}" \
  --run-tag "${RESULT_VARIANT}_fold${FOLD}" \
  --out-csv "${EXPERIMENT_DIR}/results/metrics/${RESULT_VARIANT}/fold${FOLD}_metrics.csv" \
  ${RUN_INFERENCE_ARGS}
