from __future__ import annotations

from typing import Sequence, Optional, Union

import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.bxmnet.blocks.vss_block3d import VSSBlock3D


class VSSLayer3D(nn.Module):
    """Stack of VSSBlock3D blocks (keeps channel-last shape)."""

    def __init__(
        self,
        dim: int,
        depth: int,
        *,
        drop_path: Union[float, Sequence[float]] = 0.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        expansion_factor: int = 1,
        use_checkpoint: bool = False,
        orientation_order: Optional[Sequence[int]] = None,
        mamba_layers: int = 1,
        scan_mode: str = "full_6",
        hss_kwargs: dict | None = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.depth = int(depth)
        self.orientation_order = list(orientation_order) if orientation_order is not None else None

        # normalize drop_path to per-block list
        if isinstance(drop_path, (list, tuple)):
            if len(drop_path) != depth:
                raise ValueError(f"drop_path list must have length=depth ({depth}), got {len(drop_path)}")
            dpr = list(map(float, drop_path))
        else:
            dpr = [float(drop_path)] * depth

        blocks = []
        for i in range(depth):
            # Fixed model choice: always orientation=0.
            # orientation = 0

            # --- If you ever want orientation scheduling again, you can swap to one of these: ---
            if self.orientation_order:
                orientation = self.orientation_order[i % len(self.orientation_order)]  # can set this during the MambaNeck instantiation
            else:
                orientation = i % 8  # the default will just apply rotation until up to how much depth specified

            blocks.append(
                VSSBlock3D(
                    hidden_dim=self.dim,
                    drop_path=dpr[i],
                    attn_drop=attn_drop,
                    mlp_drop=mlp_drop,
                    expansion_factor=expansion_factor,
                    use_checkpoint=use_checkpoint,
                    orientation=orientation,
                    mamba_layers=mamba_layers,
                    hss_kwargs=hss_kwargs,
                    scan_mode=scan_mode
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x
