import os
import torch
torch._dynamo.config.suppress_errors = True
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from nnunetv2.run.run_training import run_training

if __name__ == "__main__":
    os.environ["nnUNet_n_proc_DA"] = "2"  # Optional: safer for cluster memory

    PATH_TO_WEIGHTS="/data/home/aim/aim_tommy/dcr_resdec/nnUNet_results/Dataset002_BraTS2019_full/nnUNetTrainerDiceCELoss_noSmooth__nnUNetPlans__3d_fullres/fold_3/checkpoint_best.pth"

    run_training(
        dataset_name_or_id="Dataset004_BraTS2020_full",
        configuration="3d_fullres",
        fold="0",  # fullscale todo: run on fold 1 next
        # trainer_class_name="nnUNetTrainer",
        trainer_class_name="nnUNetTrainerDiceCELoss_noSmooth",
        plans_identifier="nnUNetPlans",
        pretrained_weights=None,
        # pretrained_weights=PATH_TO_WEIGHTS,  # using the pretrained weights from BraTS2019 for faster training
        num_gpus=1,
        export_validation_probabilities=True,  # <-- this matches --npz
        continue_training=False,
        only_run_validation=False,
        disable_checkpointing=False,
        val_with_best=True,  # trying with False, usually do True
        device=torch.device("cuda")
    )

