import os
import torch
torch._dynamo.config.suppress_errors = True
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from nnunetv2.run.run_training import run_training

if __name__ == "__main__":
    os.environ["nnUNet_n_proc_DA"] = "2"  # Optional: safer for cluster memory

    run_training(
        dataset_name_or_id="Dataset004_BraTS2020_full",
        configuration="3d_fullres",
        fold="0",  # ablation todo: keep on the same fold to get result visualization
        #trainer_class_name="nnUNetTrainerMedSegMamba",
        #trainer_class_name="nnUNetTrainerDiceCELoss_noSmooth",
	trainer_class_name="nnUNetTrainerBXMNet_oaff_ablation",
	plans_identifier="nnUNetPlans",
        pretrained_weights=None,
        num_gpus=1,
        export_validation_probabilities=True,  # <-- this matches --npz
        continue_training=False,
        only_run_validation=False,
        disable_checkpointing=False,
        val_with_best=True,  # trying with False, usually do True
        device=torch.device("cuda")
	#device=torch.device("cpu")
    )

