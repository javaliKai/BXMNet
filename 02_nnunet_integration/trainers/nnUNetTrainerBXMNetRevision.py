import json
import os
from copy import deepcopy
from typing import Any, Dict

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainerBXMNet import nnUNetTrainerBXMNet
from nnunetv2.training.nnUNetTrainer.bxmnet.bxmnet import BXMNet, DEFAULT_BXMNET_PRESET, get_bxmnet_preset


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None or value == "" else float(value)


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None or value == "" else int(value)


def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def build_revision_bxmnet_config() -> Dict[str, Any]:
    preset = os.environ.get("BXMNET_PRESET", DEFAULT_BXMNET_PRESET)
    cfg = get_bxmnet_preset(preset)
    model = cfg["model"]

    neck = model.setdefault("neck", {})
    decoder = model.setdefault("decoder", {})

    neck["neck_depth"] = _env_int("BXMNET_NECK_DEPTH", int(neck.get("neck_depth", 4)))
    neck["neck_hss_scan_mode"] = os.environ.get("BXMNET_SCAN_MODE", neck.get("neck_hss_scan_mode", "full_6"))

    hss_kwargs = dict(neck.get("neck_hss_kwargs") or {})
    hss_kwargs["fuse_mode"] = os.environ.get("BXMNET_HSS_FUSE_MODE", hss_kwargs.get("fuse_mode", "aff"))
    hss_kwargs["aff_groups"] = _env_int("BXMNET_AFF_GROUPS", int(hss_kwargs.get("aff_groups", 4)))
    hss_kwargs["aff_r"] = _env_int("BXMNET_AFF_R", int(hss_kwargs.get("aff_r", 8)))
    hss_kwargs["aff_tau"] = _env_float("BXMNET_AFF_TAU", float(hss_kwargs.get("aff_tau", 0.6)))
    hss_kwargs["aff_pdrop"] = _env_float("BXMNET_AFF_PDROP", float(hss_kwargs.get("aff_pdrop", 0.1)))
    hss_kwargs["aff_alpha"] = _env_float("BXMNET_AFF_ALPHA", float(hss_kwargs.get("aff_alpha", 1.0)))
    hss_kwargs["aff_beta"] = _env_float("BXMNET_AFF_BETA", float(hss_kwargs.get("aff_beta", 1.0)))
    hss_kwargs["aff_use_pixel"] = _env_bool("BXMNET_AFF_USE_PIXEL", bool(hss_kwargs.get("aff_use_pixel", True)))
    hss_kwargs["aff_use_channel"] = _env_bool("BXMNET_AFF_USE_CHANNEL", bool(hss_kwargs.get("aff_use_channel", True)))
    neck["neck_hss_kwargs"] = hss_kwargs

    decoder["decoder_fusion_mode"] = os.environ.get(
        "BXMNET_DECODER_FUSION_MODE",
        decoder.get("decoder_fusion_mode", "mafcm"),
    )

    patch_path = os.environ.get("BXMNET_CONFIG_PATCH")
    if patch_path:
        with open(patch_path, "r") as f:
            cfg = _deep_update(cfg, json.load(f))

    print("[BXMNetRevision] config:", json.dumps({
        "scan_mode": cfg["model"]["neck"].get("neck_hss_scan_mode"),
        "hss_kwargs": cfg["model"]["neck"].get("neck_hss_kwargs"),
        "neck_depth": cfg["model"]["neck"].get("neck_depth"),
        "decoder_fusion_mode": cfg["model"]["decoder"].get("decoder_fusion_mode"),
    }, indent=2))
    audit_path = os.environ.get("BXMNET_CONFIG_AUDIT_PATH")
    if audit_path:
        os.makedirs(os.path.dirname(audit_path), exist_ok=True)
        with open(audit_path, "w") as f:
            json.dump({
                "preset": preset,
                "scan_mode": cfg["model"]["neck"].get("neck_hss_scan_mode"),
                "hss_kwargs": cfg["model"]["neck"].get("neck_hss_kwargs"),
                "neck_depth": cfg["model"]["neck"].get("neck_depth"),
                "decoder_fusion_mode": cfg["model"]["decoder"].get("decoder_fusion_mode"),
                "model": cfg["model"],
            }, f, indent=2, sort_keys=True, default=str)
    return cfg


class nnUNetTrainerBXMNetRevision(nnUNetTrainerBXMNet):
    """BXMNet trainer for revision ablations, controlled by environment variables."""

    def train_step(self, batch: dict) -> dict:
        p = _env_float("BXMNET_MODALITY_DROPOUT_P", 0.0)
        if p > 0:
            data = batch["data"].clone()
            n_channels = data.shape[1]
            for b in range(data.shape[0]):
                if torch.rand(1).item() < p:
                    keep_at_least = max(1, _env_int("BXMNET_MODALITY_DROPOUT_KEEP_MIN", 1))
                    max_drop = max(1, n_channels - keep_at_least)
                    n_drop = torch.randint(1, max_drop + 1, (1,)).item()
                    drop_idx = torch.randperm(n_channels)[:n_drop]
                    data[b, drop_idx] = 0
            batch = dict(batch)
            batch["data"] = data
        return super().train_step(batch)

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        return BXMNet(config=build_revision_bxmnet_config())
