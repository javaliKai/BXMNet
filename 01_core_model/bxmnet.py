from __future__ import annotations

import copy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Optional import paths:
# - If you keep BXMNet inside nnUNet (recommended for now), your modules may live at:
#   nnunetv2.training.nnUNetTrainer.bxmnet.*
# - If you later move BXMNet into a standalone package, relative imports can work.
# -----------------------------------------------------------------------------
try:
    # standalone/local package style
    from .dcr_encoder import DCREncoder  # type: ignore
    from .mamba_neck import MambaNeck  # type: ignore
    from .mafcm_decoder import MAFCMResidualDecoder3D  # type: ignore
except Exception:
    # nnUNet style path (your current file uses this)
    from nnunetv2.training.nnUNetTrainer.bxmnet.dcr_encoder import DCREncoder  # type: ignore
    from nnunetv2.training.nnUNetTrainer.bxmnet.mamba_neck import MambaNeck  # type: ignore
    from nnunetv2.training.nnUNetTrainer.bxmnet.mafcm_decoder import MAFCMResidualDecoder3D  # type: ignore


# =============================================================================
# Plan-like, PREDEFINED presets (architecture frozen)
# =============================================================================
#
# Goal:
#   - Keep BXMNet architecture independent of nnUNet "plans" architecture section.
#   - Still "follow the plans" by freezing a plan-matching preset (n_stages, channels, strides, etc).
#   - Trainer may validate the preset against plans.json, but never uses plans to BUILD the model.
#
# NOTE:
#   - This is intentionally Python-native (callables like nn.Conv3d, nn.InstanceNorm3d).
#   - If you want JSON-serializable configs later, create a "serialization layer" that converts
#     strings <-> callables. Keep the preset itself as the source of truth for the code.
# =============================================================================

# ====== Ablation config for OAFF start ====== 
A0_UNIFORM = {
    #"aff_enabled": True,
    "aff_use_pixel": False,
    "aff_use_channel": False,
    "aff_groups": 4,
    "aff_tau": 0.6,
    "aff_pdrop": 0.0,
}

A1_PIXEL_ONLY = {
    #"aff_enabled": True,
    "aff_use_pixel": True,
    "aff_use_channel": False,
    "aff_groups": 4,
    "aff_r": 8,
    "aff_tau": 0.6,
    "aff_pdrop": 0.1,
    "aff_alpha": 1.0,
    "aff_beta": 1.0,  # ignored
}

A2_CHANNEL_ONLY = {
    #"aff_enabled": True,
    "aff_use_pixel": False,
    "aff_use_channel": True,
    "aff_groups": 4,
    "aff_r": 8,
    "aff_tau": 0.6,
    "aff_pdrop": 0.1,
    "aff_alpha": 1.0,  # ignored
    "aff_beta": 1.0,
}

A3_PIXEL_CHANNEL = {
    #"aff_enabled": True,
    "aff_use_pixel": True,
    "aff_use_channel": True,
    "aff_groups": 4,
    "aff_r": 8,
    "aff_tau": 0.6,
    "aff_pdrop": 0.1,
    "aff_alpha": 1.0,
    "aff_beta": 1.0,
}

B0_GROUP_1 = {
    #"aff_enabled": True,
    "aff_use_pixel": True,
    "aff_use_channel": True,
    "aff_groups": 1,
    "aff_r": 8,
    "aff_tau": 0.6,
    "aff_pdrop": 0.1,
}

B1_GROUP_2 = {
    #"aff_enabled": True,
    "aff_use_pixel": True,
    "aff_use_channel": True,
    "aff_groups": 2,
    "aff_r": 8,
    "aff_tau": 0.6,
    "aff_pdrop": 0.1,
}

B2_GROUP_4 = {
    #"aff_enabled": True,
    "aff_use_pixel": True,
    "aff_use_channel": True,
    "aff_groups": 4,
    "aff_r": 8,
    "aff_tau": 0.6,
    "aff_pdrop": 0.1,
}

B3_GROUP_8 = {
    #"aff_enabled": True,
    "aff_use_pixel": True,
    "aff_use_channel": True,
    "aff_groups": 8,
    "aff_r": 8,
    "aff_tau": 0.6,
    "aff_pdrop": 0.1,
}

OAFF_ABLATIONS = {
    # A — pathway ablation
    "A0": A0_UNIFORM,
    "A1": A1_PIXEL_ONLY,
    "A2": A2_CHANNEL_ONLY,
    "A3": A3_PIXEL_CHANNEL,  # default

    # B — group ablation
    "B0": B0_GROUP_1,
    "B1": B1_GROUP_2,
    "B2": B2_GROUP_4,  # default
    "B3": B3_GROUP_8,
}

# ====== Ablation config for OAFF end ====== 


BXMNET_PRESETS: Dict[str, Dict[str, Any]] = {
    # Matches your uploaded nnUNetPlans.json for:
    #   Dataset004_BraTS2020_full / configuration "3d_fullres"
    "brats2020_3d_fullres_v1": {
        "meta": {
            "preset_name": "brats2020_3d_fullres_v1",
            "expected_nnunet_configuration": "3d_fullres",
            "notes": "Frozen BXMNet preset aligned with nnUNet 3d_fullres plan (96^3 patches, 4 stages).",
        },
        # Not used to build the model; only for optional sanity checks / documentation.
        "expected_preprocessing": {
            "patch_size": [96, 96, 96],
            "spacing": [1.0, 1.0, 1.0],
            "batch_size": 1,
        },
        "model": {
            # If dataset differs, do NOT patch from plans. Update this preset instead.
            "input_channels": 4,   # BraTS modalities
            "num_classes": 4,      # background + 3 tumor subregions (adjust if yours differs)

            # --- encoder (plan-like) ---
            "encoder": {
                "n_stages": 4,
                "features_per_stage": [32, 64, 128, 256],
                "conv_op": nn.Conv3d,
                "kernel_sizes": [(3, 3, 3)] * 4,
                "strides": [(1, 1, 1), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
                "n_blocks_per_stage": [2, 2, 2, 2],

                # plan defaults
                "conv_bias": False,
                "norm_op": nn.InstanceNorm3d,
                "norm_op_kwargs": {"eps": 1e-5, "affine": True},
                "dropout_op": None,
                "dropout_op_kwargs": None,
                "nonlin": nn.LeakyReLU,
                "nonlin_kwargs": {"inplace": True},
            },

            # --- DCR knobs (forwarded as-is to DCREncoder) ---
            "dcr": {
                "dilation_xy": 1,
                "force_dilation_stages": False,
                "resnet_d_skip": True,
                "pool_type": "conv",
                "stochastic_depth_p": 0.0,
                "squeeze_excitation": False,
                "squeeze_excitation_reduction_ratio": 1.0 / 16,
                "disable_default_stem": False,
                "stem_channels": None,
            },

            # --- neck (bottleneck) ---
            "neck": {
                "neck_depth": 4,
                "neck_drop_path_rate": 0.1,
                "neck_attn_drop": 0.0,
                "neck_mlp_drop": 0.0,
                "neck_expansion_factor": 1,
                "neck_use_checkpoint": False,
                "neck_orientation_order": None,
                "neck_add_post_layernorm": True,
                "neck_mamba_layers": 1,
                # use the default 6 scans, for further options see the hss3d.py
                "neck_hss_scan_mode": "full_6",  
                "neck_hss_kwargs": None,  # default
                # optional 1x1 channel adapter into the neck (None => use encoder last channels)
                "neck_channels": None,
            },

            # --- decoder ---
            "decoder": {
                "decoder_n_blocks_per_stage": 2,
                "deep_supervision": True,
                "num_modalities": 4,     # default: equals input_channels for BraTS
                "mafcm_alpha": 0.50,
                "mafcm_beta": 0.25,
                "mafcm_up_mode": "trilinear",
            },
        },
    }
}

DEFAULT_BXMNET_PRESET = "brats2020_3d_fullres_v1"


def get_bxmnet_preset(preset: str = DEFAULT_BXMNET_PRESET) -> Dict[str, Any]:
    """Return a deep-copied preset dict so caller can modify safely."""
    if preset not in BXMNET_PRESETS:
        raise KeyError(f"Unknown BXMNet preset '{preset}'. Available: {sorted(BXMNET_PRESETS.keys())}")
    return copy.deepcopy(BXMNET_PRESETS[preset])


# =============================================================================
# BXMNet (frozen architecture via preset/config)
# =============================================================================

class BXMNet(nn.Module):
    """
    BXMNet = DCREncoder -> (optional 1x1 adapter) -> MambaNeck -> MAFCMResidualDecoder3D

    IMPORTANT DESIGN:
      - Architecture is predefined (preset/config), not constructed from nnUNet plans.
      - nnUNet trainer may validate plans.json against this preset, but does NOT pass plan-arch into BXMNet.

    Deep supervision:
      - if deep_supervision=True: returns List[Tensor] with logits at multiple scales
        ordered as [highest_res, ..., lowest_res] (nnU-Net friendly)
      - else: returns Tensor logits at highest resolution
    """

    def __init__(
        self,
        *,
        preset: str = DEFAULT_BXMNET_PRESET,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.preset_name = preset
        preset_cfg = get_bxmnet_preset(preset)
        cfg = copy.deepcopy(preset_cfg) if config is None else copy.deepcopy(config)

        if "model" not in cfg:
            raise ValueError("BXMNet config must have a top-level key 'model'.")

        m = cfg["model"]

        # -------------------------
        # 0) Frozen I/O assumptions
        # -------------------------
        self.input_channels = int(m["input_channels"])
        self.num_classes = int(m["num_classes"])

        enc = m["encoder"]
        dcr = m.get("dcr", {})
        neck = m.get("neck", {})
        dec = m.get("decoder", {})

        # basic schema checks (helpful to catch silent typos)
        n_stages = int(enc["n_stages"])
        if len(enc["features_per_stage"]) != n_stages:
            raise ValueError("encoder.features_per_stage length must match encoder.n_stages")
        if len(enc["kernel_sizes"]) != n_stages:
            raise ValueError("encoder.kernel_sizes length must match encoder.n_stages")
        if len(enc["strides"]) != n_stages:
            raise ValueError("encoder.strides length must match encoder.n_stages")
        if len(enc["n_blocks_per_stage"]) != n_stages:
            raise ValueError("encoder.n_blocks_per_stage length must match encoder.n_stages")

        # -------------------------
        # 1) Encoder (returns skips)
        # -------------------------
        self.encoder = DCREncoder(
            input_channels=self.input_channels,
            n_stages=n_stages,
            features_per_stage=enc["features_per_stage"],
            conv_op=enc["conv_op"],
            kernel_sizes=enc["kernel_sizes"],
            strides=enc["strides"],
            n_blocks_per_stage=enc["n_blocks_per_stage"],
            conv_bias=bool(enc.get("conv_bias", False)),
            norm_op=enc.get("norm_op", None),
            norm_op_kwargs=enc.get("norm_op_kwargs", None),
            dropout_op=enc.get("dropout_op", None),
            dropout_op_kwargs=enc.get("dropout_op_kwargs", None),
            nonlin=enc.get("nonlin", None),
            nonlin_kwargs=enc.get("nonlin_kwargs", None),
            return_skips=True,  # IMPORTANT: BXMNet needs all skips
            disable_default_stem=bool(dcr.get("disable_default_stem", False)),
            stem_channels=dcr.get("stem_channels", None),
            pool_type=dcr.get("pool_type", "conv"),
            stochastic_depth_p=float(dcr.get("stochastic_depth_p", 0.0)),
            squeeze_excitation=bool(dcr.get("squeeze_excitation", False)),
            squeeze_excitation_reduction_ratio=float(dcr.get("squeeze_excitation_reduction_ratio", 1.0 / 16)),
            dilation_xy=int(dcr.get("dilation_xy", 1)),
            force_dilation_stages=bool(dcr.get("force_dilation_stages", False)),
            resnet_d_skip=bool(dcr.get("resnet_d_skip", True)),
        )

        enc_channels = list(self.encoder.output_channels)  # per-stage channels (encoder order)

        # ------------------------------------
        # 2) Optional channel adapter into neck
        # ------------------------------------
        enc_last_ch = int(enc_channels[-1])
        neck_channels = neck.get("neck_channels", None)
        neck_ch = int(neck_channels) if neck_channels is not None else enc_last_ch

        if neck_ch != enc_last_ch:
            # Keep it minimal: a single 1x1x1 projection.
            self.neck_in_proj = nn.Conv3d(enc_last_ch, neck_ch, kernel_size=1, bias=False)
            enc_channels[-1] = neck_ch  # decoder bottleneck channel becomes neck_ch
        else:
            self.neck_in_proj = nn.Identity()

        # -------------------------
        # 3) Mamba neck (bottleneck)
        # -------------------------
        self.neck = MambaNeck(
            channels=neck_ch,
            depth=int(neck.get("neck_depth", 4)),
            drop_path_rate=float(neck.get("neck_drop_path_rate", 0.1)),
            attn_drop=float(neck.get("neck_attn_drop", 0.0)),
            mlp_drop=float(neck.get("neck_mlp_drop", 0.0)),
            expansion_factor=int(neck.get("neck_expansion_factor", 1)),
            use_checkpoint=bool(neck.get("neck_use_checkpoint", False)),
            orientation_order=neck.get("neck_orientation_order", None),
            add_post_layernorm=bool(neck.get("neck_add_post_layernorm", True)),
            mamba_layers=int(neck.get("neck_mamba_layers", 1)),
            scan_mode=neck.get("neck_hss_scan_mode", "full_6"),
            hss_kwargs=neck.get("neck_hss_kwargs", None),
        )

        # -------------------------
        # 4) Decoder (+ deep sup)
        # -------------------------
        # Decoder expects:
        #   - encoder_channels in encoder order, including bottleneck as last entry.
        #   - strides length = len(encoder_channels) - 1
        strides_for_decoder = self.encoder.strides[1:]  # skip first stride (input -> stage0)
        self.decoder = MAFCMResidualDecoder3D(
            encoder_channels=enc_channels,
            strides=strides_for_decoder,
            num_classes=self.num_classes,
            n_blocks_per_stage=int(dec.get("decoder_n_blocks_per_stage", 2)),
            deep_supervision=bool(dec.get("deep_supervision", True)),
            num_modalities=int(dec.get("num_modalities", self.input_channels)),
            kernel_sizes=self.encoder.kernel_sizes,  # keep RF behavior consistent
            conv_bias=bool(enc.get("conv_bias", False)),
            norm_op=enc.get("norm_op", nn.InstanceNorm3d),
            nonlin=enc.get("nonlin", nn.LeakyReLU),
            nonlin_kwargs=enc.get("nonlin_kwargs", {"inplace": True}),
            mafcm_alpha=float(dec.get("mafcm_alpha", 0.50)),
            mafcm_beta=float(dec.get("mafcm_beta", 0.25)),
            mafcm_up_mode=str(dec.get("mafcm_up_mode", "trilinear")),
        )

        self.deep_supervision = bool(dec.get("deep_supervision", True))

    @classmethod
    def from_preset(cls, preset: str = DEFAULT_BXMNET_PRESET) -> "BXMNet":
        return cls(preset=preset)

    def forward(self, x: torch.Tensor, *, modality_mask=None):
        """
        Args:
            x: [B, C, D, H, W]
            modality_mask: optional [B, M] mask forwarded into MAFCM blocks.

        Returns:
            If deep_supervision:
                List[logits] with highest-res first (nnU-Net style)
            Else:
                logits tensor at highest resolution
        """
        # 1) encoder skips: [s0 (high-res), ..., s_{N-1} (low-res)]
        skips = self.encoder(x)

        # 2) bottleneck through neck
        bottleneck_in = self.neck_in_proj(skips[-1])
        bottleneck = self.neck(bottleneck_in)

        # 3) decoder expects: [s0, ..., s_{N-2}, bottleneck]
        dec_skips = list(skips[:-1]) + [bottleneck]

        out = self.decoder(dec_skips, modality_mask=modality_mask)

        # out is already:
        # - List[Tensor] highest-res first if deep_supervision=True
        # - Tensor if deep_supervision=False
        return out


# =============================================================================
# Minimal smoke test (optional)
# =============================================================================
def _smoke_test() -> None:
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BXMNet.from_preset(DEFAULT_BXMNET_PRESET).to(device).train()

    x = torch.randn(1, model.input_channels, 96, 96, 96, device=device, requires_grad=True)
    out = model(x)

    if model.deep_supervision:
        assert isinstance(out, (list, tuple)) and len(out) > 0
        loss = out[0].mean()
    else:
        loss = out.mean()

    loss.backward()
    print("[OK] BXMNet preset smoke test passed.")


if __name__ == "__main__":
    _smoke_test()
