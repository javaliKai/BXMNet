#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(CDPATH= cd -- "${PACKAGE_DIR}/.." && pwd)"

CONDA_BASE="${CONDA_BASE:-/home/hkun/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-dcr_resdec}"

DATASET="${DATASET:-Dataset007_BraTS2020_no_fold0}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
PLANS="${PLANS:-nnUNetPlans}"
TRAINER="${TRAINER:-nnUNetTrainerBXMNetRevision}"
DEVICE="${DEVICE:-cuda}"
NUM_GPUS="${NUM_GPUS:-1}"
SEED="${SEED:-3407}"

export nnUNet_raw="${nnUNet_raw:-${PROJECT_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-/home/hkun/data}"
RESULT_ROOT="${RESULT_ROOT:-${nnUNet_results:-${PACKAGE_DIR}/results/nnUNet_results}}"
LOG_ROOT="${LOG_ROOT:-${PACKAGE_DIR}/results/logs}"
AUDIT_ROOT="${AUDIT_ROOT:-${PACKAGE_DIR}/results/config}"
export nnUNet_results="${RESULT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}/nnUNet:${PROJECT_ROOT}/dynamic-network-architectures${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"
export TORCHINDUCTOR_DISABLE="${TORCHINDUCTOR_DISABLE:-1}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TORCH_COMPILE_DISABLE="${TORCH_COMPILE_DISABLE:-1}"
export nnUNet_compile="${nnUNet_compile:-false}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${PACKAGE_DIR}/.cache/matplotlib}"

# Defaults are the complete BXMNet protocol. Override these with the
# hyperparameters selected on the original fold0 before the final run.
export BXMNET_PRESET="${BXMNET_PRESET:-brats2020_3d_fullres_v1}"
export BXMNET_NECK_DEPTH="${BXMNET_NECK_DEPTH:-5}"
export BXMNET_SCAN_MODE="${BXMNET_SCAN_MODE:-full_6}"
export BXMNET_HSS_FUSE_MODE="${BXMNET_HSS_FUSE_MODE:-aff}"
export BXMNET_AFF_GROUPS="${BXMNET_AFF_GROUPS:-4}"
export BXMNET_AFF_R="${BXMNET_AFF_R:-8}"
export BXMNET_AFF_PDROP="${BXMNET_AFF_PDROP:-0.1}"
export BXMNET_AFF_ALPHA="${BXMNET_AFF_ALPHA:-1.0}"
export BXMNET_AFF_BETA="${BXMNET_AFF_BETA:-1.0}"
export BXMNET_AFF_TAU="${BXMNET_AFF_TAU:-0.6}"
export BXMNET_AFF_USE_PIXEL="${BXMNET_AFF_USE_PIXEL:-1}"
export BXMNET_AFF_USE_CHANNEL="${BXMNET_AFF_USE_CHANNEL:-1}"
export BXMNET_DECODER_FUSION_MODE="${BXMNET_DECODER_FUSION_MODE:-mafcm}"

die() {
    echo "[Dataset007 CV] ERROR: $*" >&2
    exit 2
}

load_environment() {
    [ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ] || die "Conda init script not found: ${CONDA_BASE}/etc/profile.d/conda.sh"
    # shellcheck disable=SC1091
    . "${CONDA_BASE}/etc/profile.d/conda.sh"
    conda activate "${CONDA_ENV_NAME}"
    command -v nnUNetv2_train >/dev/null 2>&1 || die "nnUNetv2_train is unavailable in conda env ${CONDA_ENV_NAME}"
}

require_dataset() {
    [ -d "${nnUNet_raw}/${DATASET}/labelsTr" ] || die "raw labelsTr not found: ${nnUNet_raw}/${DATASET}/labelsTr"
    [ -f "${nnUNet_raw}/${DATASET}/dataset.json" ] || die "raw dataset.json not found: ${nnUNet_raw}/${DATASET}/dataset.json"
    [ -f "${nnUNet_preprocessed}/${DATASET}/dataset.json" ] || die "preprocessed dataset.json not found: ${nnUNet_preprocessed}/${DATASET}/dataset.json"
    [ -f "${nnUNet_preprocessed}/${DATASET}/splits_final.json" ] || die "splits_final.json not found: ${nnUNet_preprocessed}/${DATASET}/splits_final.json"
    [ -f "${nnUNet_preprocessed}/${DATASET}/${PLANS}.json" ] || die "plans not found: ${nnUNet_preprocessed}/${DATASET}/${PLANS}.json"
    [ -d "${nnUNet_preprocessed}/${DATASET}/nnUNetPlans_3d_fullres" ] || die "preprocessed 3d_fullres data not found"
}

prepare_directories() {
    mkdir -p "${RESULT_ROOT}" "${LOG_ROOT}" "${AUDIT_ROOT}" "${MPLCONFIGDIR}"
}

model_dir() {
    local fold="$1"
    printf '%s/%s/%s__%s__%s/fold_%s\n' \
        "${RESULT_ROOT}" "${DATASET}" "${TRAINER}" "${PLANS}" "${CONFIGURATION}" "${fold}"
}

write_audit() {
    local fold="$1"
    local output="${AUDIT_ROOT}/fold_${fold}_environment.txt"
    {
        printf 'dataset=%s\n' "${DATASET}"
        printf 'configuration=%s\n' "${CONFIGURATION}"
        printf 'fold=%s\n' "${fold}"
        printf 'trainer=%s\n' "${TRAINER}"
        printf 'plans=%s\n' "${PLANS}"
        printf 'seed=%s\n' "${SEED}"
        printf 'nnUNet_n_proc_DA=%s\n' "${nnUNet_n_proc_DA}"
        printf 'conda_env=%s\n' "${CONDA_ENV_NAME}"
        printf 'python=%s\n' "$(command -v python)"
        printf 'nnUNet_raw=%s\n' "${nnUNet_raw}"
        printf 'nnUNet_preprocessed=%s\n' "${nnUNet_preprocessed}"
        printf 'nnUNet_results=%s\n' "${RESULT_ROOT}"
        printf 'BXMNET_PRESET=%s\n' "${BXMNET_PRESET}"
        printf 'BXMNET_NECK_DEPTH=%s\n' "${BXMNET_NECK_DEPTH}"
        printf 'BXMNET_SCAN_MODE=%s\n' "${BXMNET_SCAN_MODE}"
        printf 'BXMNET_HSS_FUSE_MODE=%s\n' "${BXMNET_HSS_FUSE_MODE}"
        printf 'BXMNET_AFF_GROUPS=%s\n' "${BXMNET_AFF_GROUPS}"
        printf 'BXMNET_AFF_R=%s\n' "${BXMNET_AFF_R}"
        printf 'BXMNET_AFF_PDROP=%s\n' "${BXMNET_AFF_PDROP}"
        printf 'BXMNET_AFF_ALPHA=%s\n' "${BXMNET_AFF_ALPHA}"
        printf 'BXMNET_AFF_BETA=%s\n' "${BXMNET_AFF_BETA}"
        printf 'BXMNET_AFF_TAU=%s\n' "${BXMNET_AFF_TAU}"
        printf 'BXMNET_AFF_USE_PIXEL=%s\n' "${BXMNET_AFF_USE_PIXEL}"
        printf 'BXMNET_AFF_USE_CHANNEL=%s\n' "${BXMNET_AFF_USE_CHANNEL}"
        printf 'BXMNET_DECODER_FUSION_MODE=%s\n' "${BXMNET_DECODER_FUSION_MODE}"
    } > "${output}"
    export BXMNET_CONFIG_AUDIT_PATH="${AUDIT_ROOT}/fold_${fold}_bxmnet_config.json"
}
