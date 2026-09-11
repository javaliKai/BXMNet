
# validate_hd95_only.py
import os
import glob
import importlib
from os.path import join, basename, exists, dirname
from typing import Optional, Type, Tuple

import numpy as np
import nibabel as nib
import torch

# nnU-Net v2 imports
from batchgenerators.utilities.file_and_folder_operations import join as bg_join, load_json as bg_load_json
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
import nnunetv2

# SciPy ops for surface & distance
from scipy.ndimage import binary_erosion, distance_transform_edt

# --- Torch compile flags off (safer for quick val runs) ---
torch._dynamo.config.suppress_errors = True
os.environ.setdefault("TORCHINDUCTOR_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")


# ----------------------------
# Trainer resolution utilities
# ----------------------------
def _resolve_trainer_class(trainer_name: str) -> Type:
    """
    Resolve a trainer class name to a class object.
    Supports fully-qualified paths or short names found under nnUNet v2.
    """
    if "." in trainer_name:
        module_path, cls_name = trainer_name.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        return getattr(mod, cls_name)
    # Otherwise, search inside nnUNet v2's trainer tree
    search_root = bg_join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    base_module = "nnunetv2.training.nnUNetTrainer"
    cls = recursive_find_python_class(search_root, trainer_name, base_module)
    if cls is None:
        raise RuntimeError(f"Could not find trainer class '{trainer_name}' under {base_module}. "
                           f"You can pass a fully-qualified class path instead.")
    return cls


def get_trainer(dataset_name_or_id: str,
                configuration: str,
                fold: int,
                trainer_name: str = "nnUNetTrainer",
                plans_identifier: str = "nnUNetPlans",
                device: str = "cuda"):
    """
    Instantiate a trainer for the given dataset/config/fold using nnU-Net v2.
    """
    # dataset id handling
    if not str(dataset_name_or_id).startswith("Dataset"):
        try:
            dataset_name_or_id = int(dataset_name_or_id)
        except ValueError:
            raise ValueError("dataset_name_or_id must be an int or 'DatasetXXX_YYY'")

    ds_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_base = bg_join(nnUNet_preprocessed, ds_name)
    plans_file = bg_load_json(bg_join(preprocessed_base, plans_identifier + ".json"))
    dataset_json = bg_load_json(bg_join(preprocessed_base, "dataset.json"))

    trainer_cls = _resolve_trainer_class(trainer_name)
    dev = torch.device(device)
    trainer = trainer_cls(plans=plans_file, configuration=configuration, fold=fold,
                          dataset_json=dataset_json, device=dev)
    return trainer


def load_checkpoint_for_validation(trainer: nnUNetTrainer,
                                   checkpoint: Optional[str] = None,
                                   prefer_best: bool = False) -> str:
    """
    checkpoint:
      - None => use checkpoint_final.pth (or best if prefer_best=True)
      - 'final'/'best'/'latest' => load from trainer.output_folder
      - '/abs/path/to/some_checkpoint.pth' => load this file
    """
    of = trainer.output_folder
    if checkpoint is None:
        ck = join(of, "checkpoint_best.pth" if prefer_best else "checkpoint_final.pth")
    else:
        ckl = checkpoint.lower()
        if ckl in ("final", "best", "latest"):
            ck = join(of, f"checkpoint_{ckl}.pth")
        else:
            ck = checkpoint

    if not exists(ck):
        raise FileNotFoundError(f"Checkpoint not found: {ck}")
    trainer.load_checkpoint(ck)
    return ck


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
    # fallback: any 'validation*' dir with nii.gz
    for d in os.listdir(trainer.output_folder):
        full = join(trainer.output_folder, d)
        if os.path.isdir(full) and d.startswith("validation"):
            if len(glob.glob(join(full, "*.nii.gz"))) > 0:
                return full
    raise RuntimeError("Could not locate validation predictions folder. "
                       "Run trainer.perform_actual_validation(...) first or set run_inference=True.")


# ----------------------
# Region mask utilities
# ----------------------
def masks_for_regions(lbl: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build WT/TC/ET masks from a label map.
    Handles both BraTS2019/2020 (labels {1,2,4}) and BraTS2024-style (labels {1,2,3}).
    Fallback: ET = max label, TC = {min label, ET}, WT = any > 0.
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


# ---------------------------------------------
# HD95 (mm): symmetric = max of directed p95's
# ---------------------------------------------
def _crop_to_nonzero(mask_union: np.ndarray, pad: int = 4) -> Tuple[slice, slice, slice]:
    coords = np.argwhere(mask_union)
    if coords.size == 0:
        # whole volume (the caller guards empty case earlier)
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
    Returns NaN if both masks are empty; +inf if exactly one is empty.
    Definition matches MONAI: max( P95(A->B), P95(B->A) ).
    """
    a = y_true.astype(bool)
    b = y_pred.astype(bool)

    a_any = a.any()
    b_any = b.any()

    if not a_any and not b_any:
        return float("nan")
    if a_any != b_any:
        return float("inf")

    # Crop to union bbox for speed
    u = a | b
    zyx = _crop_to_nonzero(u, pad=4)
    a = a[zyx]
    b = b[zyx]

    # Surfaces via binary erosion
    struct = np.ones((3, 3, 3), dtype=bool)
    a_erode = binary_erosion(a, structure=struct, border_value=0)
    b_erode = binary_erosion(b, structure=struct, border_value=0)
    a_surf = a ^ a_erode
    b_surf = b ^ b_erode

    # If surfaces vanished (tiny blobs), treat as exact overlap
    if not a_surf.any() and not b_surf.any():
        return 0.0

    # Distance to the complement: spacing (x,y,z) gives mm units
    dt_to_b = distance_transform_edt(~b, sampling=spacing)
    dt_to_a = distance_transform_edt(~a, sampling=spacing)

    dists_a2b = dt_to_b[a_surf]
    dists_b2a = dt_to_a[b_surf]

    # If one side has no surface, treat its percentile as 0 (perfect containment)
    if dists_a2b.size == 0 and dists_b2a.size == 0:
        return 0.0
    p95_a2b = 0.0 if dists_a2b.size == 0 else float(np.percentile(dists_a2b, 95))
    p95_b2a = 0.0 if dists_b2a.size == 0 else float(np.percentile(dists_b2a, 95))
    return max(p95_a2b, p95_b2a)


# -----------------------------------
# Main entry: compute HD95 per case
# -----------------------------------
def validate_hd95_only(
    dataset_name_or_id: str,
    configuration: str,
    fold: int,
    run_tag: str,
    out_csv: str,
    trainer_name: str = "nnUNetTrainer",
    plans_identifier: str = "nnUNetPlans",
    device: str = "cuda",
    checkpoint: Optional[str] = None,  # 'final' | 'best' | 'latest' | path
    prefer_best: bool = False,
    run_inference: bool = False       # if True, run trainer.perform_actual_validation()
):
    """
    Loads a checkpoint, (optionally) runs nnU-Net validation to produce segmentations,
    then computes ONLY HD95 (WT/TC/ET) in millimeters and writes a compact CSV.

    CSV columns: case_id, run_tag, fold, WT_HD95, TC_HD95, ET_HD95
    """
    # 1) Trainer
    trainer = get_trainer(dataset_name_or_id, configuration, fold,
                          trainer_name=trainer_name, plans_identifier=plans_identifier, device=device)

    # 2) Load ckpt
    #used_ckpt = load_checkpoint_for_validation(trainer, checkpoint=checkpoint, prefer_best=prefer_best)

    # 3) Optionally run validation to generate predictions
    if run_inference:
        print("[info] Running perform_actual_validation(...) to (re)generate predictions...")
        trainer.perform_actual_validation(save_probabilities=False)

    # 4) Predictions folder
    pred_dir = find_validation_folder(trainer)

    # 5) GT folder
    ds_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    labels_tr = join(nnUNet_raw, ds_name, "labelsTr")
    if not exists(labels_tr):
        raise FileNotFoundError(f"Could not find labelsTr at: {labels_tr}")

    # 6) Iterate .nii.gz predictions and compute HD95 only
    nii_paths = sorted(glob.glob(join(pred_dir, "*.nii.gz")))
    rows = []
    for pred_path in nii_paths:
        case_id = basename(pred_path)[:-7] if pred_path.endswith(".nii.gz") else basename(pred_path)

        gt_path = join(labels_tr, f"{case_id}.nii.gz")
        if not exists(gt_path):
            # Some trainers may copy GT into validation folder; try that
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
        zooms = gt_img.header.get_zooms()
        spacing = tuple(float(x) for x in zooms[:3])  # (x, y, z) mm

        # Region masks (dataset-agnostic)
        wt_gt, tc_gt, et_gt = masks_for_regions(gt)
        wt_pr, tc_pr, et_pr = masks_for_regions(pr)

        # HD95 (mm) only
        wt_hd = hd95_mm(wt_gt, wt_pr, spacing)
        tc_hd = hd95_mm(tc_gt, tc_pr, spacing)
        et_hd = hd95_mm(et_gt, et_pr, spacing)

        rows.append((case_id, run_tag, fold, wt_hd, tc_hd, et_hd))

    # 7) Write CSV (append or create)
    header = "case_id,run_tag,fold,WT_HD95,TC_HD95,ET_HD95\n"
    os.makedirs(dirname(out_csv), exist_ok=True)
    need_header = (not exists(out_csv)) or (os.stat(out_csv).st_size == 0)
    with open(out_csv, "a") as f:
        if need_header:
            f.write(header)
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]},{r[4]},{r[5]}\n")

    # 8) Print quick summary (NaN/inf-safe)
    hd_arr = np.array([[x[3], x[4], x[5]] for x in rows], dtype=float)
    hd_arr_disp = hd_arr.copy()
    hd_arr_disp[np.isinf(hd_arr_disp)] = np.nan
    hd_means = np.nanmean(hd_arr_disp, axis=0)

    # print(f"\n[Done] {run_tag} (fold {fold}) from {used_ckpt}")
    print(f"Saved CSV → {out_csv}")
    print(f"Mean HD95 (mm, lower better) – WT: {hd_means[0]:.2f}, TC: {hd_means[1]:.2f}, ET: {hd_means[2]:.2f}")
    print("(Note: HD95 uses GT voxel spacing and reports NaN when both masks are empty; +inf when one is empty.)")


# --------------
# Example usage:
# --------------
if __name__ == "__main__":
    PARAMS = dict(
        dataset_name_or_id="Dataset004_BraTS2020_full",  # or "201"
        configuration="3d_fullres",
        fold=0,
        run_tag="ablation_axial_sagittal",
        out_csv="custom_validation/brats_hd95_only.csv",
        trainer_name="nnUNetTrainerDiceCELoss_noSmooth",         # or custom trainer class
        #trainer_name="nnUNetTrainerUNETR",
	plans_identifier="nnUNetPlans",
        device="cuda",
        checkpoint="best",                     # "final" | "best" | "latest" | "/abs/path/to.ckpt"
        prefer_best=True,
        run_inference=False                    # True to (re)generate predictions first
    )
    validate_hd95_only(**PARAMS)
