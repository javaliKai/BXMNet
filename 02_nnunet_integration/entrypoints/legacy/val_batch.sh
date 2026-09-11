#!/bin/sh
#SBATCH --job-name=sanity_weightvis
#SBATCH --partition=sugon_5880ada   #rtx3090_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --output=runlog/%j.log
#SBATCH --error=runlog/%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=4768

source /data/home/aim/aim_tommy/miniconda3/bin/activate
cd /data/home/aim/aim_tommy/dcr_resdec
conda activate dcr_resdec


# Set nnUNet environment variables
export nnUNet_raw="/data/home/aim/aim_tommy/dcr_resdec/nnUNet_raw"
export nnUNet_preprocessed="/data/home/aim/aim_tommy/dcr_resdec/nnUNet_preprocessed"
export nnUNet_results="/data/home/aim/aim_tommy/dcr_resdec/nnUNet_results"

#python main_validation_fixed.py > runlog/val_0_fold_full.txt
#python main_validation_bypass.py > runlog/val.txt
#python main_validation_hd95_fixed.py
python main_validation_onego.py
#python main_validation_onego_weight.py
