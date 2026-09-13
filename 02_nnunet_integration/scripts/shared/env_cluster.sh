#!/bin/bash

export PROJECT_ROOT="${PROJECT_ROOT:-/data/aim_nuist/aim_temp/aim_tommy/take_me/dcr_resdec}"

. /data/home/aim/aim_tommy/miniconda3/bin/activate
conda activate dcr_resdec

export nnUNet_raw="${nnUNet_raw:-${PROJECT_ROOT}/nnUNet_raw}"
export nnUNet_preprocessed="${nnUNet_preprocessed:-${PROJECT_ROOT}/nnUNet_preprocessed}"
export nnUNet_results="${nnUNet_results:-${PROJECT_ROOT}/nnUNet_results}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-2}"

cd "${PROJECT_ROOT}"

echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
