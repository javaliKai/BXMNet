#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
EXPERIMENT_DIR=$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)
PROJECT_ROOT="${PROJECT_ROOT:-$(CDPATH= cd -- "${EXPERIMENT_DIR}/../.." && pwd)}"

FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainerBXMNet}"
RESULT_VARIANT="${RESULT_VARIANT:-full_model}"

echo "Submitting BXM-13 BraTS2024 full BXMNet, FOLD=${FOLD}, TRAINER=${TRAINER}"

train_job=$(env PROJECT_ROOT="${PROJECT_ROOT}" FOLD="${FOLD}" TRAINER="${TRAINER}" RESULT_VARIANT="${RESULT_VARIANT}" \
  sbatch --parsable "${SCRIPT_DIR}/submit_train_fold.sh")
train_job="${train_job%%;*}"
echo "Submitted training: ${train_job}"

val_job=$(env PROJECT_ROOT="${PROJECT_ROOT}" FOLD="${FOLD}" TRAINER="${TRAINER}" RESULT_VARIANT="${RESULT_VARIANT}" \
  sbatch --parsable --dependency="afterok:${train_job}" "${SCRIPT_DIR}/submit_validate_fold.sh")
val_job="${val_job%%;*}"
echo "Submitted validation: ${val_job} depends on afterok:${train_job}"
