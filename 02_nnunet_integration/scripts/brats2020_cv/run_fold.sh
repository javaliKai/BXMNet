#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/common.sh"

FOLD="${1:-${FOLD:-}}"
[ -n "${FOLD}" ] || die "pass a fold index: 0, 1, 2, 3, or 4"
case "${FOLD}" in
    0|1|2|3|4) ;;
    *) die "fold must be 0, 1, 2, 3, or 4; got ${FOLD}" ;;
esac

load_environment
require_dataset
prepare_directories
write_audit "${FOLD}"

MODEL_DIR="$(model_dir "${FOLD}")"
LOG_FILE="${LOG_ROOT}/fold_${FOLD}.log"
mkdir -p "${MODEL_DIR}"

if [ -f "${MODEL_DIR}/checkpoint_final.pth" ]; then
    if [ "${SKIP_COMPLETED:-1}" = "1" ]; then
        echo "[Dataset007 CV] fold ${FOLD} already has checkpoint_final.pth; skipping."
        exit 0
    fi
    die "completed fold exists at ${MODEL_DIR}; set SKIP_COMPLETED=1 or use a new RESULT_ROOT"
fi

ARGS=(
    "${DATASET}" "${CONFIGURATION}" "${FOLD}"
    -tr "${TRAINER}"
    -p "${PLANS}"
    -num_gpus "${NUM_GPUS}"
    -device "${DEVICE}"
)

if [ "${SAVE_NPZ:-1}" = "1" ]; then
    ARGS+=(--npz)
fi
if [ "${VAL_BEST:-1}" = "1" ]; then
    ARGS+=(--val_best)
fi
if [ "${CONTINUE:-0}" = "1" ] || [ -f "${MODEL_DIR}/checkpoint_latest.pth" ]; then
    ARGS+=(--c)
fi

echo "[Dataset007 CV] starting fold ${FOLD}"
echo "[Dataset007 CV] dataset=${DATASET} trainer=${TRAINER} plans=${PLANS} device=${DEVICE}"
echo "[Dataset007 CV] log=${LOG_FILE}"
nnUNetv2_train "${ARGS[@]}" 2>&1 | tee "${LOG_FILE}"
echo "[Dataset007 CV] fold ${FOLD} completed"
