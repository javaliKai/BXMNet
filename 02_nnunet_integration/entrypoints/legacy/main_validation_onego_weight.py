# validate_dice_hd95_combined.py
import os
import glob
import importlib
from os.path import join, basename, exists, dirname
from typing import Optional, Type, Tuple

import numpy as np
import nibabel as nib
import torch

from batchgenerators.utilities.file_and_folder_operations import join as bg_join, load_json as bg_load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

from scipy.ndimage import binary_erosion, distance_transform_edt

# --- Torch compile flags off (safer for quick metric runs) ---
torch._dynamo.config.suppress_errors = True
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


# ----------------------------
# Trainer resolution utilities
# ----------------------------
def _resolve_trainer_class(trainer_name: str) -> Type:
    """
    Resolve trainer class from fully-qualified path or short name under nnU-Net v2.
    """
    if "." in trainer_name:
        module_path, cls_name = trainer_name.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)

    search_root = bg_join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    base_module = "nnunetv2.training.nnUNetTrainer"
    cls = recursive_find_python_class(search_root, trainer_name, base_module)
    if cls is None:
        raise RuntimeError(
            f"Could not find trainer class '{trainer_name}' under {base_module}. "
            f"Pass a fully-qualified class path instead."
        )
    return cls


def get_trainer(dataset_name_or_id: str,
                configuration: str,
                fold: int,
                trainer_name: str = "nnUNetTrainer",
                plans_identifier: str = "nnUNetPlans",
                device: str = "cuda"):
    """
    Instantiate a trainer (for resolving output_folder + config), without needing checkpoint load.
    """
    if not str(dataset_name_or_id).startswith("Dataset"):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id must be an int or 'DatasetXXX_YYY'")

    ds_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_base = bg_join(nnUNet_preprocessed, ds_name)
    plans = bg_load_json(bg_join(preprocessed_base, plans_identifier + ".json"))
    dataset_json = bg_load_json(bg_join(preprocessed_base, "dataset.json"))

    trainer_cls = _resolve_trainer_class(trainer_name)
    dev = torch.device(device)
    trainer = trainer_cls(plans=plans, configuration=configuration, fold=fold,
                          dataset_json=dataset_json, device=dev)
    return trainer


def find_validation_folder(trainer: nnUNetTrainer) -> str:
    """
    Locate a folder under trainer.output_folder that contains validation predictions (.nii.gz).
    """
    candidates = [
        join(trainer.output_folder, "validation"),
        join(trainer.output_folder, "validation_raw"),
    ]
    for c in candidates:
        if exists(c) and len(glob.glob(join(c, "*.nii.gz"))) > 0:
            return c

    for d in os.listdir(trainer.output_folder):
        full = join(trainer.output_folder, d)
        if os.path.isdir(full) and d.startswith("validation"):
            if len(glob.glob(join(full, "*.nii.gz"))) > 0:
                return full

    raise RuntimeError(
        "Could not locate validation predictions folder. "
        "Make sure nnU-Net already generated validation masks."
    )


# ----------------------
# Region mask utilities
# ----------------------
def masks_for_regions(lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build WT/TC/ET masks from a label map.
    Handles:
      - BraTS2019/2020: {1,2,4} (ET=4)
      - BraTS2024-style: {1,2,3} (ET=3)
    Fallback: ET=max label, WT=any >0, TC={1,ET} if 1 exists else {min,ET}
    """
    pos = np.unique(lbl[lbl > 0]).astype(int)
    if pos.size == 0:
        z = np.zeros_like(lbl, dtype=bool)
        return z, z, z

    s = set(pos.tolist())
    if s == {1, 2, 4}:
        wt = np.isin(lbl, (1, 2, 4))
        tc = np.isin(lbl, (1, 4))
        et = (lbl == 4)
    elif 4 not in s and {1, 2, 3}.issubset(s):
        wt = np.isin(lbl, (1, 2, 3))
        tc = np.isin(lbl, (1, 3))
        et = (lbl == 3)
    else:
        et_label = int(pos.max())
        wt = lbl > 0
        if 1 in s:
            tc = np.isin(lbl, (1, et_label))
        else:
            tc = np.isin(lbl, (int(pos.min()), et_label))
        et = (lbl == et_label)

    return wt.astype(bool), tc.astype(bool), et.astype(bool)


# ----------------------
# Dice on boolean masks
# ----------------------
def dice_coef(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a = y_true.astype(bool)
    b = y_pred.astype(bool)
    inter = np.logical_and(a, b).sum(dtype=np.float64)
    s = a.sum(dtype=np.float64) + b.sum(dtype=np.float64)
    if s == 0.0:
        return float("nan")  # both empty
    return 2.0 * inter / s


# ---------------------------------------------
# HD95 (mm): symmetric = max of directed p95's
# ---------------------------------------------
def _crop_to_nonzero(mask_union: np.ndarray, pad: int = 4) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask_union)
    if coords.size == 0:
        return (slice(0, mask_union.shape[0]),
                slice(0, mask_union.shape[1]),
                slice(0, mask_union.shape[2]))
    mins = coords.min(0)
    maxs = coords.max(0) + 1
    mins = np.maximum(mins - pad, 0)
    maxs = np.minimum(maxs + pad, mask_union.shape)
    return tuple(slice(lo, hi) for lo, hi in zip(mins, maxs))


def hd95_mm(y_true: np.ndarray, y_pred: np.ndarray, spacing: Tuple[float, float, float]) -> float:
    """
    95th percentile Hausdorff distance in millimeters.
    Returns NaN if both empty; +inf if exactly one is empty.
    """
    a = y_true.astype(bool)
    b = y_pred.astype(bool)

    a_any = a.any()
    b_any = b.any()

    if not a_any and not b_any:
        return float("nan")
    if a_any != b_any:
        return float("inf")

    u = a | b
    zyx = _crop_to_nonzero(u, pad=4)
    a = a[zyx]
    b = b[zyx]

    struct = np.ones((3, 3, 3), dtype=bool)
    a_erode = binary_erosion(a, structure=struct, border_value=0)
    b_erode = binary_erosion(b, structure=struct, border_value=0)
    a_surf = a ^ a_erode
    b_surf = b ^ b_erode

    if not a_surf.any() and not b_surf.any():
        return 0.0

    dt_to_b = distance_transform_edt(~b, sampling=spacing)
    dt_to_a = distance_transform_edt(~a, sampling=spacing)

    dists_a2b = dt_to_b[a_surf]
    dists_b2a = dt_to_a[b_surf]

    if dists_a2b.size == 0 and dists_b2a.size == 0:
        return 0.0
    p95_a2b = 0.0 if dists_a2b.size == 0 else float(np.percentile(dists_a2b, 95))
    p95_b2a = 0.0 if dists_b2a.size == 0 else float(np.percentile(dists_b2a, 95))
    return max(p95_a2b, p95_b2a)


# -----------------------------------
# Main entry: compute Dice + HD95
# -----------------------------------
def validate_dice_hd95(
    dataset_name_or_id: str,
    configuration: str,
    fold: int,
    run_tag: str,
    out_csv: str,
    trainer_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    device: str = "cuda",
    run_inference: bool = False,   # keep default False (your desired behavior)
):
    """
    Uses existing nnU-Net validation predictions to compute BOTH Dice + HD95 (WT/TC/ET).
    Does NOT load checkpoint by default.

    CSV columns:
      case_id,run_tag,fold,
      WT_Dice,TC_Dice,ET_Dice,
      WT_HD95,TC_HD95,ET_HD95
    """
    trainer = get_trainer(dataset_name_or_id, configuration, fold,
                          trainer_name=trainer_name, plans_identifier=plans_identifier, device=device)

    # build the network via trainer class
    trainer.initialize()
    ckpt = join(trainer.output_folder, "checkpoint_best.pth")
    if not exists(ckpt):
        ckpt = join(trainer.output_folder, "checkpoint_final.pth")

    if not exists(ckpt):
        raise FileNotFoundError(f"Checkpoint not found in: {trainer.output_folder}")

    trainer.load_checkpoint(ckpt)


    if run_inference:
        print("[info] Running perform_actual_validation(...) to (re)generate predictions...")
        # trainer.perform_actual_validation(save_probabilities=False)
        trainer.perform_few_samples_validation(save_probabilities=False, num_cases=2)

    # # UPDATE: inject the weight visualization logic
    # model = trainer.network
    # for name, module in model.named_modules():
    #     if hasattr(module, "last_aff_weights") and module.last_aff_weights is not None:
    #         w = module.last_aff_weights
    #         print(name, w.shape)
    #         print(name, w.min().item(), w.max().item())
    #         print("dir-sum check:", w[0, :, 0, 0, 0, 0].sum().item())
    pred_dir = find_validation_folder(trainer)

    ds_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    labels_tr = join(nnUNet_raw, ds_name, "labelsTr")
    if not exists(labels_tr):
        raise FileNotFoundError(f"Could not find labelsTr at: {labels_tr}")

    nii_paths = sorted(glob.glob(join(pred_dir, "*.nii.gz")))

    rows = []
    for pred_path in nii_paths:
        case_id = basename(pred_path)[:-7] if pred_path.endswith(".nii.gz") else basename(pred_path)

        gt_path = join(labels_tr, f"{case_id}.nii.gz")
        if not exists(gt_path):
            alt_gt = join(pred_dir, "gt_niftis", f"{case_id}.nii.gz")
            if exists(alt_gt):
                gt_path = alt_gt
            else:
                print(f"[WARN] GT not found for {case_id}, skipping.")
                continue

        gt_img = nib.load(gt_path)
        pr_img = nib.load(pred_path)

        gt = np.asanyarray(gt_img.dataobj)
        pr = np.asanyarray(pr_img.dataobj)

        # spacing in mm for HD95
        zooms = gt_img.header.get_zooms()
        spacing = tuple(float(x) for x in zooms[:3])  # (x, y, z)

        # region masks
        wt_gt, tc_gt, et_gt = masks_for_regions(gt)
        wt_pr, tc_pr, et_pr = masks_for_regions(pr)

        # Dice
        wt_d = dice_coef(wt_gt, wt_pr)
        tc_d = dice_coef(tc_gt, tc_pr)
        et_d = dice_coef(et_gt, et_pr)

        # HD95
        wt_h = hd95_mm(wt_gt, wt_pr, spacing)
        tc_h = hd95_mm(tc_gt, tc_pr, spacing)
        et_h = hd95_mm(et_gt, et_pr, spacing)

        # Avg of Dice and HD95
        dice_vals = np.array([wt_d, tc_d, et_d], dtype=float)
        avg_dice = float(np.nanmean(dice_vals))  # ignores NaN

        hd_vals = np.array([wt_h, tc_h, et_h], dtype=float)
        hd_vals[np.isinf(hd_vals)] = np.nan      # ignore inf in AVG
        avg_hd95 = float(np.nanmean(hd_vals))    # ignores NaN

        rows.append((case_id, run_tag, fold,
                    wt_d, tc_d, et_d, avg_dice,
                    wt_h, tc_h, et_h, avg_hd95))
    # write CSV
    header = (
      "case_id,run_tag,fold,"
      "WT_Dice,TC_Dice,ET_Dice,AVG_Dice,"
      "WT_HD95,TC_HD95,ET_HD95,AVG_HD95\n"
    )
    os.makedirs(dirname(out_csv), exist_ok=True)
    need_header = (not exists(out_csv)) or (os.stat(out_csv).st_size == 0)
    with open(out_csv, "a") as f:
        if need_header:
            f.write(header)
        for r in rows:
            f.write(
                f"{r[0]},{r[1]},{r[2]},"
                f"{r[3]},{r[4]},{r[5]},{r[6]},"
                f"{r[7]},{r[8]},{r[9]},{r[10]}\n"
            )

    # Summary
    dice_arr = np.array([[x[3], x[4], x[5]] for x in rows], dtype=float)
    dice_means = np.nanmean(dice_arr, axis=0)
    avg_dice_mean = float(np.nanmean(np.array([x[6] for x in rows], dtype=float)))

    hd_arr = np.array([[x[7], x[8], x[9]] for x in rows], dtype=float)
    hd_arr_disp = hd_arr.copy()
    hd_arr_disp[np.isinf(hd_arr_disp)] = np.nan
    hd_means = np.nanmean(hd_arr_disp, axis=0)
    avg_hd95_mean = float(np.nanmean(np.array([x[10] for x in rows], dtype=float)))

    print(f"Mean Dice – WT: {dice_means[0]:.4f}, TC: {dice_means[1]:.4f}, ET: {dice_means[2]:.4f}, AVG: {avg_dice_mean:.4f}")
    print(f"Mean HD95 (mm) – WT: {hd_means[0]:.2f}, TC: {hd_means[1]:.2f}, ET: {hd_means[2]:.2f}, AVG: {avg_hd95_mean:.2f}")

if __name__ == "__main__":
    PARAMS = dict(
        dataset_name_or_id="Dataset004_BraTS2020_full",
        configuration="3d_fullres",
        fold=0,  # todo: run the fifth ablation on 3rd fold
        run_tag="oaff_weight_extraction",
        out_csv="custom_validation/brats2020_oaff_weight.csv",  # change the filename according to ablation
        trainer_name="nnUNetTrainerBXMNet", # DONT' FORGET TO CHANGE THIS!
        plans_identifier="nnUNetPlans",
        device="cuda",
        run_inference=True,  # keep False to skip nnUNet validation
    )
    validate_dice_hd95(**PARAMS)
