import argparse

from main_validation_onego import validate_dice_hd95


def parse_args():
    parser = argparse.ArgumentParser(description="Validate configurable BXMNet revision experiment.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--configuration", default="3d_fullres")
    parser.add_argument("--trainer", default="nnUNetTrainerBXMNetRevision")
    parser.add_argument("--plans", default="nnUNetPlans")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-tag", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--run-inference", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    validate_dice_hd95(
        dataset_name_or_id=args.dataset,
        configuration=args.configuration,
        fold=args.fold,
        run_tag=args.run_tag,
        out_csv=args.out_csv,
        trainer_name=args.trainer,
        plans_identifier=args.plans,
        device=args.device,
        run_inference=args.run_inference,
    )
