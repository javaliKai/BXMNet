import argparse
import os

import torch

torch._dynamo.config.suppress_errors = True
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

from nnunetv2.run.run_training import run_training


def parse_args():
    parser = argparse.ArgumentParser(description="Train configurable BXMNet revision experiment.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", default="0")
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainerBXMNetRevision")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--only-run-validation", action="store_true")
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--val-with-best", action="store_true", default=True)
    parser.add_argument("--no-val-with-best", dest="val_with_best", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ.setdefault("nnUNet_n_proc_DA", "2")

    run_training(
        dataset_name_or_id=args.dataset,
        configuration=args.configuration,
        fold=str(args.fold),
        trainer_class_name=args.trainer,
        plans_identifier=args.plans,
        pretrained_weights=None,
        num_gpus=args.num_gpus,
        export_validation_probabilities=True,
        continue_training=args.continue_training,
        only_run_validation=args.only_run_validation,
        disable_checkpointing=args.disable_checkpointing,
        val_with_best=args.val_with_best,
        device=torch.device(args.device),
    )
