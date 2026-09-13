import torch
import torch.nn as nn
import torch.nn.functional as F

class SixWayAFF(nn.Module):
    """
    Adaptive Fusion for 6 streams (X±, Y±, Z±).
    Works on channel-first 3D tensors.
    Inputs:
        z6: [B, 6, C, H, W, D]  (six post-Mamba features)
    Outputs:
        fused: [B, C, H, W, D]
    Design:
      - Pixel-wise head: Conv1x1x1 over concat features to get 6*G logits per voxel.
      - Channel-wise head: GAP per direction -> group MLP -> 6*G logits (broadcast spatially).
      - Combine: logits = α*pixel + β*channel; softmax over the 6 directions per group.
      - Fuse group-wise and reassemble channels.
    """
    def __init__(self, c, groups=4, r=8, tau=0.6, pdrop=0.1,
                 use_pixel=True, use_channel=True, alpha=1.0, beta=1.0):
        super().__init__()
        assert c % groups == 0, "c must be divisible by groups"
        self.c = c
        self.g = groups
        self.gc = c // groups
        self.tau = tau
        self.alpha = alpha
        self.beta = beta
        self.use_pixel = use_pixel
        self.use_channel = use_channel
        self.drop = nn.Dropout(pdrop) if pdrop and pdrop > 0 else nn.Identity()

        # Pixel-wise head: conv on concatenated [B, 6*C, H, W, D] -> [B, 6*G, H, W, D]
        if use_pixel:
            self.pix = nn.Conv3d(in_channels=6 * c, out_channels=6 * groups, kernel_size=1, bias=True)

        # Channel-wise head: per-direction GAP then group-MLP: [B, 6, G, gc] -> [B, 6*G]
        if use_channel:
            hid = max(self.gc // r, 16)
            self.chan_fc1 = nn.Linear(self.gc, hid)
            self.chan_fc2 = nn.Linear(hid, 1)  # 1 logit per group per direction

    def forward(self, z6, return_weights=False):
        # z6: [B, 6, C, H, W, D]
        B, six, C, H, W, D = z6.shape
        assert six == 6 and C == self.c

        # ---------- Pixel-wise logits ----------
        logits_pix = 0.0
        if self.use_pixel:
            # concat along channels per direction, then 1x1x1 -> [B, 6*G, H, W, D]
            x_cat = z6.permute(0, 2, 1, 3, 4, 5).reshape(B, 6 * C, H, W, D)
            logits_pix = self.pix(x_cat)  # [B, 6*G, H, W, D]
            logits_pix = logits_pix.view(B, 6, self.g, H, W, D)  # [B, 6, G, H, W, D]

        # ---------- Channel-wise logits ----------
        logits_chan = 0.0
        if self.use_channel:
            # group the channels, GAP spatially: [B, 6, G, gc]
            z_grp = z6.view(B, 6, self.g, self.gc, H, W, D).mean(dim=(4, 5, 6))  # GAP -> [B, 6, G, gc]
            h = F.silu(self.chan_fc1(z_grp))                                     # [B, 6, G, hid]
            logits_chan = self.chan_fc2(h).squeeze(-1)                           # [B, 6, G]
            # broadcast to spatial
            logits_chan = logits_chan[..., None, None, None]                     # [B, 6, G, 1, 1, 1]

        # ---------- Combine + softmax over 6 ----------
        logits = 0.0
        if self.use_pixel and self.use_channel:
            logits = self.alpha * logits_pix + self.beta * logits_chan
        elif self.use_pixel:
            logits = logits_pix
        elif self.use_channel:
            logits = logits_chan
        else:
            # fallback to uniform mixing
            logits = torch.zeros(B, 6, self.g, H, W, D, device=z6.device, dtype=z6.dtype)

        logits = self.drop(logits)
        w = F.softmax(logits / self.tau, dim=1)  # along the 6 directions

        # ---------- Fuse group-wise ----------
        # reshape z for group-wise mixing: [B, 6, G, gc, H, W, D]
        # z_grp = z6.view(B, 6, self.g, self.gc, H, W, D)
        # fused_grp = (w[..., None, :, :, :, :] * z_grp).sum(dim=1)   # sum over 6 -> [B, G, gc, H, W, D]
        # fused = fused_grp.view(B, C, H, W, D)                       # [B, C, H, W, D]

        # reminder, z6: [B, 6, C', H, W, D]
        z_grp = z6.view(B, 6, self.g, self.gc, H, W, D).contiguous()   # gc = C'//G

        # expand weights over the channel-within-group axis
        w_exp = w.unsqueeze(3)                  # [B, 6, G, 1, H, W, D]

        fused_grp = (w_exp * z_grp).sum(dim=1)      # [B, G, gc, H, W, D]
        fused = fused_grp.view(B, C, H, W, D)

        # Optionally return the weight for visualizing orientation weight
        if return_weights:
            return fused, w
        return fused
