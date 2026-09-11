#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/common.sh"

load_environment
require_dataset
prepare_directories

FOLDS="${FOLDS:-0 1 2 3 4}"
echo "[Dataset007 CV] sequential five-fold run"
echo "[Dataset007 CV] folds=${FOLDS}"
echo "[Dataset007 CV] conda=${CONDA_ENV_NAME}"
echo "[Dataset007 CV] raw=${nnUNet_raw}"
echo "[Dataset007 CV] preprocessed=${nnUNet_preprocessed}"
echo "[Dataset007 CV] results=${RESULT_ROOT}"
echo "[Dataset007 CV] hyperparameters: G=${BXMNET_AFF_GROUPS}, alpha=${BXMNET_AFF_ALPHA}, beta=${BXMNET_AFF_BETA}, tau=${BXMNET_AFF_TAU}"

for fold in ${FOLDS}; do
    "${SCRIPT_DIR}/run_fold.sh" "${fold}"
done

echo "[Dataset007 CV] requested folds completed or skipped."
