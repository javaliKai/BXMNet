
"""
MAFCM-based UNet-style 3D residual decoder (standalone, GitHub-friendly).

Key design choice to avoid channel-mismatch:
- We FIRST upsample the high-level tensor with a ConvTranspose3d that maps
  C_below -> C_skip, then we call MAFCM3D(L=c_skip, H=h_up=c_skip).
This mirrors your cluster implementation where MAFCM is constructed with
(c_low=c_skip, c_high=c_skip) and is fed (skip, upsampled_high).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.bxmnet.blocks.fcm import MAFCM3D


def _as_3tuple(x: Union[int, Sequence[int]]) -> Tuple[int, int, int]:
    if isinstance(x, int):
        return (x, x, x)
    x = tuple(int(v) for v in x)
    if len(x) != 3:
        raise ValueError(f"Expected int or len-3 sequence, got {x}")
    return x


class ResidualBlock3D(nn.Module):
    """
    A small, standard 3D residual block:
      Conv3d -> Norm -> Act -> Conv3d -> Norm -> (+ skip) -> Act
    """
    def __init__(
        self,
        channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        norm_op: type[nn.Module] = nn.InstanceNorm3d,
        nonlin: type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        conv_bias: bool = False,
    ):
        super().__init__()
        k = _as_3tuple(kernel_size)
        pad = tuple(kk // 2 for kk in k)
        nonlin_kwargs = nonlin_kwargs or {"inplace": True}

        self.conv1 = nn.Conv3d(channels, channels, kernel_size=k, padding=pad, bias=conv_bias)
        self.norm1 = norm_op(channels)
        self.act1 = nonlin(**nonlin_kwargs)

        self.conv2 = nn.Conv3d(channels, channels, kernel_size=k, padding=pad, bias=conv_bias)
        self.norm2 = norm_op(channels)
        self.act2 = nonlin(**nonlin_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.act1(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + identity
        out = self.act2(out)
        return out


class StackedResidualBlocks3D(nn.Module):
    def __init__(
        self,
        channels: int,
        n_blocks: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        norm_op: type[nn.Module] = nn.InstanceNorm3d,
        nonlin: type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        conv_bias: bool = False,
    ):
        super().__init__()
        self.blocks = nn.Sequential(
            *[
                ResidualBlock3D(
                    channels=channels,
                    kernel_size=kernel_size,
                    norm_op=norm_op,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    conv_bias=conv_bias,
                )
                for _ in range(int(n_blocks))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


@dataclass(frozen=True)
class StageSpec:
    below_ch: int
    skip_ch: int
    stride: Tuple[int, int, int]


class MAFCMResidualDecoder3D(nn.Module):
    """
    Standalone UNet-style decoder that expects `skips` in encoder order:
        skips = [s0, s1, ..., s_{N-2}, bottleneck]
    where s0 is the highest-resolution skip, and bottleneck is the lowest-resolution tensor
    (e.g., output of your MambaNeck).

    - Each stage: ConvTranspose3d (below_ch -> skip_ch) -> MAFCM3D(skip_ch, skip_ch) -> residual refinement
    - Output: last stage logits (or list of deep-supervision logits, highest-res first).
    """
    def __init__(
        self,
        encoder_channels: Sequence[int],
        strides: Sequence[Union[int, Sequence[int]]],
        num_classes: int,
        n_blocks_per_stage: Union[int, Sequence[int]] = 2,
        deep_supervision: bool = False,
        num_modalities: int = 4,
        kernel_sizes: Optional[Sequence[Union[int, Sequence[int]]]] = None,
        conv_bias: bool = False,
        norm_op: type[nn.Module] = nn.InstanceNorm3d,
        nonlin: type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: Optional[dict] = None,
        mafcm_alpha: float = 0.50,
        mafcm_beta: float = 0.25,
        mafcm_up_mode: str = "trilinear",
    ):
        super().__init__()

        if len(encoder_channels) < 2:
            raise ValueError("encoder_channels must include at least [skip, bottleneck].")

        n_stages_encoder = len(encoder_channels)
        if len(strides) != (n_stages_encoder - 1):
            raise ValueError(
                f"strides must have length len(encoder_channels)-1. "
                f"Got strides={len(strides)} vs encoder_channels={len(encoder_channels)}"
            )

        # normalize n_blocks_per_stage
        if isinstance(n_blocks_per_stage, int):
            n_blocks = [int(n_blocks_per_stage)] * (n_stages_encoder - 1)
        else:
            n_blocks = [int(x) for x in n_blocks_per_stage]
            if len(n_blocks) != (n_stages_encoder - 1):
                raise ValueError(
                    f"n_blocks_per_stage must be int or length {n_stages_encoder-1}, got {len(n_blocks)}"
                )

        # optional per-stage kernel sizes (encoder order); we will index as in your cluster code: kernel_sizes[-(s+1)]
        if kernel_sizes is None:
            kernel_sizes = [3] * n_stages_encoder
        if len(kernel_sizes) != n_stages_encoder:
            raise ValueError(
                f"kernel_sizes must be None or length {n_stages_encoder}, got {len(kernel_sizes)}"
            )

        self.deep_supervision = bool(deep_supervision)
        self.num_classes = int(num_classes)
        self.num_modalities = int(num_modalities)

        # Build stages (mirror your cluster implementation: iterate from bottleneck outward)
        stage_specs: List[StageSpec] = []
        transpconvs: List[nn.Module] = []
        fcm_blocks: List[nn.Module] = []
        stages: List[nn.Module] = []
        seg_layers: List[nn.Module] = []

        for s in range(1, n_stages_encoder):
            below_ch = int(encoder_channels[-s])          # features coming from below (decoder state)
            skip_ch  = int(encoder_channels[-(s + 1)])    # channels of the skip we're merging into
            stride   = _as_3tuple(strides[-s])

            stage_specs.append(StageSpec(below_ch=below_ch, skip_ch=skip_ch, stride=stride))

            # 1) upsample high feature and map channels to skip_ch
            transpconvs.append(
                nn.ConvTranspose3d(
                    in_channels=below_ch,
                    out_channels=skip_ch,
                    kernel_size=stride,
                    stride=stride,
                    bias=conv_bias,
                )
            )

            # 2) MAFCM is instantiated with (c_low=skip_ch, c_high=skip_ch) because we pass h_up (already mapped to skip_ch)
            fcm_blocks.append(
                MAFCM3D(
                    c_low=skip_ch,
                    c_high=skip_ch,
                    num_modalities=self.num_modalities,
                    alpha=mafcm_alpha,
                    beta=mafcm_beta,
                    up_mode=mafcm_up_mode,
                )
            )

            # 3) local refinement on skip_ch (no concat; MAFCM returns skip_ch)
            stages.append(
                StackedResidualBlocks3D(
                    channels=skip_ch,
                    n_blocks=n_blocks[s - 1],
                    kernel_size=kernel_sizes[-(s + 1)],
                    norm_op=norm_op,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                    conv_bias=conv_bias,
                )
            )

            # 4) seg head for this resolution
            seg_layers.append(nn.Conv3d(skip_ch, self.num_classes, kernel_size=1, bias=True))

        self._stage_specs = stage_specs
        self.transpconvs = nn.ModuleList(transpconvs)
        self.fcm_blocks = nn.ModuleList(fcm_blocks)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: Sequence[torch.Tensor], modality_mask: Optional[torch.Tensor] = None):
        """
        Args:
            skips: list/tuple of tensors in encoder order:
                   [high_res_skip, ..., low_res_skip, bottleneck]
                   bottleneck is typically your MambaNeck output, e.g. (B,256,12,12,12).
            modality_mask: optional (B, M) float/bool tensor forwarded into MAFCM3D.
        """
        if len(skips) != (len(self._stage_specs) + 1):
            raise ValueError(
                f"Expected {len(self._stage_specs)+1} tensors in skips (including bottleneck), got {len(skips)}"
            )

        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []

        for s, spec in enumerate(self._stage_specs):
            c_low = skips[-(s + 2)]

            # Helpful shape checks for the usual channel mismatch gotcha
            if c_low.shape[1] != spec.skip_ch:
                raise ValueError(
                    f"[stage {s}] skip channels mismatch: expected {spec.skip_ch}, got {c_low.shape[1]}"
                )
            if lres_input.shape[1] != spec.below_ch:
                raise ValueError(
                    f"[stage {s}] below channels mismatch: expected {spec.below_ch}, got {lres_input.shape[1]}"
                )

            # 1) Up: below_ch -> skip_ch, and spatial upsample
            h_up = self.transpconvs[s](lres_input)

            # Ensure spatial alignment (in case of odd sizes in custom crops)
            if h_up.shape[2:] != c_low.shape[2:]:
                raise ValueError(
                    f"[stage {s}] spatial mismatch after ConvTranspose3d: "
                    f"h_up={tuple(h_up.shape)} vs c_low={tuple(c_low.shape)}. "
                    f"Check your encoder strides/crops."
                )

            # 2) Fuse
            x = self.fcm_blocks[s](c_low, h_up, modality_mask=modality_mask)

            # 3) Refine
            x = self.stages[s](x)

            # 4) Segmentation head(s)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self._stage_specs) - 1):
                seg_outputs.append(self.seg_layers[-1](x))

            # 5) Next stage input
            lres_input = x

        seg_outputs = seg_outputs[::-1]  # highest-res first

        if self.deep_supervision:
            return seg_outputs
        return seg_outputs[0]
