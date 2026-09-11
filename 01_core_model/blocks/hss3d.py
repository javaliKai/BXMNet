from __future__ import annotations

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.bxmnet.backends.mamba import Mamba, MambaConfig
from nnunetv2.training.nnUNetTrainer.bxmnet.blocks.aff6 import SixWayAFF


class HSS3D(nn.Module):
    """HexaSelectiveScan (HSS) for 3D feature maps using mambapy.
    input/output: [B, H, W, D, C] (channel-last)
    """

    # --- NEW: scan mode registry ---
    # Each entry is a list of (axis_name, bidirectional_bool)
    # axis_name in {"axial", "sagittal", "coronal"}
    SCAN_PRESETS = {
        "axial_1": [("axial", False)],
        "axial_sagittal_2": [("axial", False), ("sagittal", False)],
        "axial_coronal_2": [("axial", False), ("coronal", False)],
        "axial_sagittal_bi_4": [("axial", True), ("sagittal", True)],
        "axial_coronal_bi_4": [("axial", True), ("coronal", True)],
        "axial_sagittal_coronal_3": [("axial", False), ("sagittal", False), ("coronal", False)],
        "full_6": [("axial", True), ("sagittal", True), ("coronal", True)],
    }

    def __init__(
        self,
        d_model: int,
        *,
        expand: int = 1,
        d_conv: int = 3,
        bias: bool = True,
        conv_bias: bool = True,
        dropout: float = 0.0,
        orientation: int = 0,
        mamba_layers: int = 1,
        # --- NEW ---
        scan_mode: str = "full_6",
        # AFF defaults
        aff_groups: int = 4,
        aff_r: int = 8,
        aff_tau: float = 0.6,
        aff_pdrop: float = 0.1,
        aff_use_pixel: bool = True,
        aff_use_channel: bool = True,
        aff_alpha: float = 1.0,
        aff_beta: float = 1.0,
        # --- NEW (optional): what to do when streams != 6 ---
        fuse_mode: str = "mean",  # {"mean", "sum"} for non-6 cases
    ) -> None:
        super().__init__()
        print(f"[DEBUG]: groups={aff_groups}")
        self.C = int(d_model)
        self.Ci = int(expand * d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.in_proj = nn.Linear(self.C, self.Ci, bias=bias)

        self.conv3d = nn.Conv3d(
            in_channels=self.Ci,
            out_channels=self.Ci,
            groups=self.Ci,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
        )
        self.act = nn.SiLU()

        self.mcfg = MambaConfig(d_model=self.Ci, n_layers=int(mamba_layers))
        self.mamba = Mamba(self.mcfg)

        self.out_norm = nn.LayerNorm(self.Ci)
        self.out_proj = nn.Linear(self.Ci, self.C, bias=bias)

        self.rot, self.unrot = self._make_rot_unrot(int(orientation))

        self.aff6 = SixWayAFF(
            c=self.Ci,
            groups=aff_groups,
            r=aff_r,
            tau=aff_tau,
            pdrop=aff_pdrop,
            use_pixel=aff_use_pixel,
            use_channel=aff_use_channel,
            alpha=aff_alpha,
            beta=aff_beta,
        )

        # --- NEW ---
        self.scan_mode = scan_mode
        self.fuse_mode = fuse_mode

        # Optional: to visualize orientation weight
        self.last_aff_weights = None


    @staticmethod
    def _make_rot_unrot(orientation: int):
        o = orientation % 8
        if o == 0:
            return (lambda x: x, lambda x: x)
        if o == 1:
            return (lambda x: torch.rot90(x, 1, (2, 3)),
                    lambda x: torch.rot90(x, -1, (2, 3)))
        if o == 2:
            return (lambda x: torch.rot90(x, 1, (3, 4)),
                    lambda x: torch.rot90(x, -1, (3, 4)))
        if o == 3:
            return (lambda x: torch.rot90(x, 1, (2, 4)),
                    lambda x: torch.rot90(x, -1, (2, 4)))
        if o == 5:
            return (lambda x: torch.rot90(x, 2, (2, 4)),
                    lambda x: torch.rot90(x, 2, (2, 4)))
        return (lambda x: x, lambda x: x)

    @staticmethod
    def _seq_forward(mamba: Mamba, x_3d: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x_3d.shape
        L = H * W * D
        x_seq = x_3d.reshape(B, C, L).transpose(1, 2).contiguous()  # [B, L, C]
        y_seq = mamba(x_seq)                                        # [B, L, C]
        y_3d = y_seq.transpose(1, 2).contiguous().reshape(B, C, H, W, D)
        return y_3d

    # --- NEW: helper to create x0/x1/x2 and map axis names ---
    @staticmethod
    def _make_axes(x0: torch.Tensor) -> dict[str, torch.Tensor]:
        # x0: [B, C, H, W, D]
        x1 = torch.transpose(x0, 2, 4).contiguous()  # [B, C, D, W, H]
        x2 = torch.transpose(x0, 3, 4).contiguous()  # [B, C, H, D, W]
        return {"axial": x0, "sagittal": x1, "coronal": x2}

    @staticmethod
    def _unmake_axis(axis: str, y: torch.Tensor) -> torch.Tensor:
        # Return to [B, C, H, W, D]
        if axis == "axial":
            return y
        if axis == "sagittal":
            return torch.transpose(y, 2, 4).contiguous()
        if axis == "coronal":
            return torch.transpose(y, 3, 4).contiguous()
        raise ValueError(f"Unknown axis={axis}")

    def _run_streams(self, x0: torch.Tensor, scan_mode: str) -> list[torch.Tensor]:
        if scan_mode not in self.SCAN_PRESETS:
            raise ValueError(
                f"Unknown scan_mode='{scan_mode}'. Valid: {list(self.SCAN_PRESETS.keys())}"
            )

        axes = self._make_axes(x0)
        outs: list[torch.Tensor] = []

        for axis_name, bidir in self.SCAN_PRESETS[scan_mode]:
            xa = axes[axis_name]  # either x0/x1/x2

            # forward
            ya_f = self._seq_forward(self.mamba, xa)
            ya_f = self._unmake_axis(axis_name, ya_f)
            outs.append(ya_f)

            if bidir:
                ya_b = torch.flip(
                    self._seq_forward(self.mamba, torch.flip(xa, dims=[2, 3, 4])),
                    dims=[2, 3, 4],
                )
                ya_b = self._unmake_axis(axis_name, ya_b)
                outs.append(ya_b)

        return outs

    def _fuse_streams(self, outs: list[torch.Tensor]) -> torch.Tensor:
        # outs: list of [B, C', H, W, D]
        if len(outs) == 6:
            z6 = torch.stack(outs, dim=1)   # [B, 6, C', H, W, D]
            # return self.aff6(z6)            # [B, C', H, W, D]
            fused, w = self.aff6(z6, return_weights=True)
            self.last_aff_weights = w.detach().cpu()  # for visualization
            return fused

        z = torch.stack(outs, dim=1)        # [B, N, C', H, W, D]
        if self.fuse_mode == "sum":
            return z.sum(dim=1)
        # default: mean
        return z.mean(dim=1)

    def forward(self, x: torch.Tensor, *, scan_mode: str | None = None) -> torch.Tensor:
        # x: [B, H, W, D, C]
        scan_mode = self.scan_mode if scan_mode is None else scan_mode

        x = self.in_proj(x)  # [B, H, W, D, C']

        x_cf = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, C', H, W, D]
        x_cf = self.act(self.conv3d(x_cf))

        x0 = self.rot(x_cf)  # [B, C', H, W, D]

        outs = self._run_streams(x0, scan_mode)  # list of [B, C', H, W, D]
        y_cf = self.unrot(self._fuse_streams(outs))

        y_cl = y_cf.permute(0, 2, 3, 4, 1).contiguous()  # [B, H, W, D, C']
        y_cl = self.out_norm(y_cl)
        y = self.out_proj(y_cl)  # [B, H, W, D, C]
        y = self.dropout(y)
        return y
