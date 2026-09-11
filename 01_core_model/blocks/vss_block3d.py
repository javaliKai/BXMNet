from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from nnunetv2.training.nnUNetTrainer.bxmnet.blocks.hss3d import HSS3D
from nnunetv2.training.nnUNetTrainer.bxmnet.blocks.layers import DropPath, FeedForward


class VSSBlock3D(nn.Module):
    """One VSS-style block (channel-last): LN -> HSS -> res, LN -> MLP -> res."""

    def __init__(
        self,
        hidden_dim: int,
        *,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        expansion_factor: int = 1,
        norm_layer: type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        orientation: int = 0,
        mamba_layers: int = 1,
        # pass-through for HSS/AFF knobs if needed later
        hss_kwargs: dict | None = None,
        scan_mode: str = "full_6",
    ) -> None:
        super().__init__()
        self.use_checkpoint = bool(use_checkpoint)
        self.ln1 = norm_layer(hidden_dim)
        self.hss = HSS3D(
            d_model=hidden_dim,
            expand=expansion_factor,  # match the original behavior (SSM inner width)
            dropout=attn_drop,
            orientation=orientation,
            mamba_layers=mamba_layers,
            scan_mode=scan_mode,
            **(hss_kwargs or {}),
        )
        self.ln2 = norm_layer(hidden_dim)
        self.mlp = FeedForward(
            dim=hidden_dim,
            hidden_dim=expansion_factor * hidden_dim,
            dropout=mlp_drop,
        )
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, D, C]
        if self.use_checkpoint and x.requires_grad:
            x = x + self.drop_path(checkpoint(lambda t: self.hss(self.ln1(t)), x))
            x = x + self.drop_path(checkpoint(lambda t: self.mlp(self.ln2(t)), x))
        else:
            x = x + self.drop_path(self.hss(self.ln1(x)))
            x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x
