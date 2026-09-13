# validate_and_report.py
import os
import glob
import json
import math
import numpy as np
import nibabel as nib

from os.path import join, basename, exists, dirname
from typing import Optional, List, Type

# nnU-Net v2 imports
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
# from nnunetv2.training.nnUNetTrainer.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss import nnUNetTrainerDiceCELoss_noSmooth
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2
import torch

# ---- Helper: build trainer (mirrors run_training.py -> get_trainer_from_args) ----
from batchgenerators.utilities.file_and_folder_operations import join as bg_join, load_json as bg_load_json, isfile

torch._dynamo.config.suppress_errors = True
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

def _resolve_trainer_class(trainer_name: str) -> Type:
    """
    Accepts either:
      - short name, e.g. 'nnUNetTrainer' or 'nnUNetTrainerDiceCELoss_noSmooth'
      - fully-qualified class path, e.g.
        'nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerDiceLoss.nnUNetTrainerDiceCELoss_noSmooth'
    Returns the class object or raises RuntimeError.
    """
    # If it looks like a fully-qualified path (has dots), try importlib first
    if '.' in trainer_name:
        module_path, cls_name = trainer_name.rsplit('.', 1)
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            return cls
        except Exception as e:
            raise RuntimeError(f"Could not import trainer '{trainer_name}': {e}")

    # Otherwise, search inside nnUNet's trainer package tree
    search_root = bg_join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    base_module = 'nnunetv2.training.nnUNetTrainer'
    cls = recursive_find_python_class(search_root, trainer_name, base_module)
    if cls is None:
        raise RuntimeError(
            f"Could not find trainer class '{trainer_name}' under {base_module}. "
            f"You can pass a fully-qualified class path instead."
        )
    return cls

def get_trainer(dataset_name_or_id: str,
                configuration: str,
                fold: int,
                trainer_name: str = 'nnUNetTrainer',
                plans_identifier: str = 'nnUNetPlans',
                device: str = 'cuda'):
    # Resolve class (works for short name or FQCN)
    trainer_cls = _resolve_trainer_class(trainer_name)
    print(f"[info] Using trainer class → {trainer_cls.__module__}.{trainer_cls.__name__}")

    # dataset id handling
    if not str(dataset_name_or_id).startswith('Dataset'):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id must be an int or 'DatasetXXX_YYY'")

    ds_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_base = bg_join(nnUNet_preprocessed, ds_name)
    plans_file = bg_join(preprocessed_base, plans_identifier + '.json')
    plans = bg_load_json(plans_file)
    dataset_json = bg_load_json(bg_join(preprocessed_base, 'dataset.json'))

    dev = torch.device(device)
    trainer = trainer_cls(plans=plans, configuration=configuration, fold=fold,
                          dataset_json=dataset_json, device=dev)
    return trainer


# ---- Helper: load checkpoint (custom path or best/final/latest) ----
def load_checkpoint_for_validation(trainer: nnUNetTrainer,
                                   checkpoint: Optional[str] = None,
                                   prefer_best: bool = False):
    """
    checkpoint:
      - None => use checkpoint_final.pth
      - 'final'/'best'/'latest' => load from trainer.output_folder
      - '/abs/path/to/some_checkpoint.pth' => load this file
    prefer_best overrides 'final' if True.
    """
    of = trainer.output_folder
    if checkpoint is None:
        ck = join(of, 'checkpoint_best.pth' if prefer_best else 'checkpoint_final.pth')
    else:
        ckl = checkpoint.lower()
        if ckl in ('final', 'best', 'latest'):
            name = f'checkpoint_{ckl}.pth'
            ck = join(of, name)
        else:
            ck = checkpoint  # custom path

    if not exists(ck):
        raise FileNotFoundError(f"Checkpoint not found: {ck}")
    trainer.load_checkpoint(ck)
    return ck

# ---- Helper: find validation predictions folder produced by perform_actual_validation ----
def find_validation_folder(trainer: nnUNetTrainer) -> str:
    # Common names used by nnU-Net v2
    candidates = [
        join(trainer.output_folder, 'validation'),
        join(trainer.output_folder, 'validation_raw'),
    ]
    for c in candidates:
        if exists(c) and len(glob.glob(join(c, '*.nii.gz'))) > 0:
            return c
    # fallback: any dir that starts with 'validation' and has nii.gz
    for d in os.listdir(trainer.output_folder):
        full = join(trainer.output_folder, d)
        if os.path.isdir(full) and d.startswith('validation'):
            if len(glob.glob(join(full, '*.nii.gz'))) > 0:
                return full
    raise RuntimeError("Could not locate validation predictions folder. "
                       "Ensure trainer.perform_actual_validation(...) has run.")

# ---- Dice computation on boolean masks ----
def dice_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # cast to bool
    a = y_true.astype(bool)
    b = y_pred.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    s = a.sum(dtype=np.float64) + b.sum(dtype=np.float64)
    if s == 0.0:
        # If both GT and pred are empty, we report NaN (BraTS often ignores absent GT).
        return float('nan')
    return 2.0 * inter / s

# ---- Build region masks from label volumes ----
def masks_for_regions(lbl: np.ndarray):
    """
    Labels: 1,2,3 (BraTS style)
    WT = {1,2,3}; TC = {1,3}; ET = {3}
    Returns (wt_mask, tc_mask, et_mask)
    """
    wt = np.isin(lbl, (1, 2, 3))
    tc = np.isin(lbl, (1, 3))
    et = (lbl == 3)
    return wt, tc, et

# ---- Main entry: run validation and write CSV ----
def validate_and_report(
    dataset_name_or_id: str,
    configuration: str,
    fold: int,
    run_tag: str,
    out_csv: str,
    trainer_name: str = 'nnUNetTrainer',
    plans_identifier: str = 'nnUNetPlans',
    device: str = 'cuda',
    checkpoint: Optional[str] = None,  # 'final' | 'best' | 'latest' | path
    prefer_best: bool = False
):
    """
    Runs nnU-Net validation for the given fold, then computes WT/TC/ET Dice and writes a CSV.

    CSV columns: case_id, run_tag, fold, WT_Dice, TC_Dice, ET_Dice
    """
    # 1) Build trainer
    trainer = get_trainer(dataset_name_or_id, configuration, fold,
                          trainer_name=trainer_name, plans_identifier=plans_identifier, device=device)

    # 2) Load checkpoint
    used_ckpt = load_checkpoint_for_validation(trainer, checkpoint=checkpoint, prefer_best=prefer_best)

    # 3) Run validation (no probabilities needed for our metrics)
    trainer.perform_actual_validation(save_probabilities=False)

    # 4) Find predictions folder
    pred_dir = find_validation_folder(trainer)

    # 5) Locate GT directory in raw data
    ds_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    labels_tr = join(nnUNet_raw, ds_name, 'labelsTr')
    if not exists(labels_tr):
        raise FileNotFoundError(f"Could not find labelsTr at: {labels_tr}")

    # 6) Iterate predictions and compute region-wise Dice
    nii_paths = sorted(glob.glob(join(pred_dir, '*.nii.gz')))
    rows = []
    for pred_path in nii_paths:
        # Derive case_id (robust to .nii.gz)
        base = basename(pred_path)
        case_id = base.replace('.nii.gz', '')

        gt_path = join(labels_tr, f'{case_id}.nii.gz')
        if not exists(gt_path):
            # Some trainers may copy GT into validation folder; try that
            alt_gt = join(pred_dir, 'gt_niftis', f'{case_id}.nii.gz')
            if exists(alt_gt):
                gt_path = alt_gt
            else:
                print(f"[WARN] GT not found for {case_id}, skipping.")
                continue

        # Load arrays
        gt = np.asanyarray(nib.load(gt_path).dataobj)
        pr = np.asanyarray(nib.load(pred_path).dataobj)

        # Build region masks
        wt_gt, tc_gt, et_gt = masks_for_regions(gt)
        wt_pr, tc_pr, et_pr = masks_for_regions(pr)

        wt_dice = dice_coef(wt_gt, wt_pr)
        tc_dice = dice_coef(tc_gt, tc_pr)
        et_dice = dice_coef(et_gt, et_pr)

        rows.append((case_id, run_tag, fold, wt_dice, tc_dice, et_dice))

    # 7) Write CSV (append or create)
    header = "case_id,run_tag,fold,WT_Dice,TC_Dice,ET_Dice\n"
    to_write = []
    if not exists(out_csv) or os.stat(out_csv).st_size == 0:
        to_write.append(header)
    for r in rows:
        line = f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n"
        to_write.append(line)
    os.makedirs(dirname(out_csv), exist_ok=True)
    with open(out_csv, 'a') as f:
        f.writelines(to_write)

    # 8) Print quick summary
    arr = np.array([[x[3], x[4], x[5]] for x in rows], dtype=float)  # WT,TC,ET
    means = np.nanmean(arr, axis=0)
    print(f"\n[Done] {run_tag} (fold {fold}) from {used_ckpt}")
    print(f"Saved CSV → {out_csv}")
    print(f"Mean Dice – WT: {means[0]:.4f}, TC: {means[1]:.4f}, ET: {means[2]:.4f}")

# -------------------------
# Example: run 1 model once
# -------------------------
if __name__ == "__main__":
    # Fill these and just `python validate_and_report.py`
    PARAMS = dict(
        dataset_name_or_id="Dataset004_BraTS2020_full",  # or "201"
        configuration="3d_fullres",
        fold=0,
        run_tag="dcr_dec",
        out_csv="custom_validation/brats_val_metrics.csv",
        trainer_name="nnUNetTrainerDiceCELoss_noSmooth",     # or your custom trainer class name
        #trainer_name="nnUNetTrainerUNETR",
	plans_identifier="nnUNetPlans",   # or your custom plans id
        device="cuda",                    # "cuda" or "cpu"
        checkpoint="best",               # "final" | "best" | "latest" | "/abs/path/to.ckpt"
        prefer_best=True # if True and checkpoint=None => uses checkpoint_best.pth
    )
    validate_and_report(**PARAMS)
