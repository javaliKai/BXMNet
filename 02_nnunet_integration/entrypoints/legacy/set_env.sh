#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"

# Set nnU-Net environment variables. Each path can still be overridden before
# sourcing this file, for example: nnUNet_results=/data/exp/results source set_env.sh
export nnUNet_raw="${nnUNet_raw:-${PROJECT_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJECT_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJECT_ROOT}/nnUNet_results}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"

mkdir -p "${nnUNet_results}"

echo "Environment variables set:"
echo "  PROJECT_ROOT        = $PROJECT_ROOT"
echo "  nnUNet_raw         = $nnUNet_raw"
echo "  nnUNet_preprocessed = $nnUNet_preprocessed"
echo "  nnUNet_results      = $nnUNet_results"
