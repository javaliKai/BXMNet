#!/bin/sh
#SBATCH --job-name=bxm_rev_val
#SBATCH --partition=sugon_5880ada
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --output=/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec/revision_bxmnet_experiments/%x_%j.log
#SBATCH --error=/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec/revision_bxmnet_experiments/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=32768

PROJECT_ROOT="${PROJECT_ROOT:-/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec}"
. "${PROJECT_ROOT}/revision_bxmnet_experiments/_shared/env_cluster.sh"

FOLD="${FOLD:-0}"
DATASET="${DATASET:-Dataset004_BraTS2020_full}"
EXPERIMENT_ID="${EXPERIMENT_ID:?Set EXPERIMENT_ID.}"
EXPERIMENT_DIR="${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the experiment folder path.}"
RUN_INFERENCE_ARGS=""
if [ "${RUN_INFERENCE:-1}" = "1" ]; then
  RUN_INFERENCE_ARGS="--run-inference"
fi

mkdir -p "${EXPERIMENT_DIR}/results/runlog" "${EXPERIMENT_DIR}/results/metrics"

python revision_bxmnet_experiments/_shared/validate_revision_bxmnet.py \
  --dataset "${DATASET}" \
  --fold "${FOLD}" \
  --trainer "${TRAINER:-nnUNetTrainerBXMNetRevision}" \
  --run-tag "${EXPERIMENT_ID}_fold${FOLD}" \
  --out-csv "${EXPERIMENT_DIR}/results/metrics/fold${FOLD}_metrics.csv" \
  ${RUN_INFERENCE_ARGS}
