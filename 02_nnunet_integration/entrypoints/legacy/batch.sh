#!/bin/sh
#SBATCH --job-name=ablation_oaff_brats2020_b3
#SBATCH --partition=sugon_5880ada  #rtx3090_1
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --output=runlog/%j.log
#SBATCH --error=runlog/%j.err
#SBATCH --time=72:00:00
#SBATCH --mem=32768  # Changed from 32768 to 24576 (24GB)

source /data/home/aim/aim_tommy/miniconda3/bin/activate
cd /data/home/aim/aim_tommy/dcr_resdec
conda activate dcr_resdec

# set nnUNet environment variables
export nnUNet_raw="/data/home/aim/aim_tommy/dcr_resdec/nnUNet_raw"
export nnUNet_preprocessed="/data/home/aim/aim_tommy/dcr_resdec/nnUNet_preprocessed"
export nnUNet_results="/data/home/aim/aim_tommy/dcr_resdec/nnUNet_results"

#python main_training_weighted.py > runlog/fullscale_fold0
python main_training.py
