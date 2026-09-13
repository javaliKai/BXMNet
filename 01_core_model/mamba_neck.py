from __future__ import annotations

from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.bxmnet.blocks.vss_layer3d import VSSLayer3D


class MambaNeck(nn.Module):
    """Fixed bottleneck module for BXMNet.

    Drop-in replacement for your previous VSS3DBottleneck, with:
      - SS3D renamed to HSS (HexaSelectiveScan)
      - 6-direction fusion via SixWayAFF ON by default
      - channel-first I/O: [B, C, D, H, W] -> [B, C, D, H, W]
    """

    def __init__(
        self,
        channels: int = 256,
        *,
        depth: int = 4,
        drop_path_rate: float = 0.1,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        expansion_factor: int = 1,
        use_checkpoint: bool = False,
        orientation_order: Optional[Sequence[int]] = None,
        add_post_layernorm: bool = True,
        mamba_layers: int = 1,
        scan_mode: str = "full_6",
        # Optional overrides for HSS/AFF knobs (leave None for defaults)
        hss_kwargs: dict | None = None,
    ) -> None:
        super().__init__()

        channels = int(channels)
        depth = int(depth)

        # per-block stochastic depth schedule
        if depth > 1 and drop_path_rate > 0:
            dpr = torch.linspace(0, float(drop_path_rate), steps=depth).tolist()
        else:
            dpr = float(drop_path_rate)

        self.vss = VSSLayer3D(
            dim=channels,
            depth=depth,
            drop_path=dpr,
            attn_drop=attn_drop,
            mlp_drop=mlp_drop,
            expansion_factor=expansion_factor,
            use_checkpoint=use_checkpoint,
            orientation_order=orientation_order,
            mamba_layers=mamba_layers,
            scan_mode="full_6",
            hss_kwargs=hss_kwargs
        )

        self.post_ln = nn.LayerNorm(channels) if add_post_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: [B, C, D, H, W]
           Returns:
            y: [B, C, D, H, W]
        """
        if x.ndim != 5:
            raise ValueError(f"MambaNeck expects 5D tensor [B,C,D,H,W], got shape={tuple(x.shape)}")

        # NCDHW -> BHWD C (as expected by VSS/HSS)
        x_bhwdc = x.permute(0, 3, 4, 2, 1).contiguous()  # [B, H, W, D, C]

        y_bhwdc = self.vss(x_bhwdc)                      # [B, H, W, D, C]
        y_bhwdc = self.post_ln(y_bhwdc)                  # [B, H, W, D, C]

        # BHWD C -> NCDHW
        y = y_bhwdc.permute(0, 4, 3, 1, 2).contiguous()  # [B, C, D, H, W]
        return y
