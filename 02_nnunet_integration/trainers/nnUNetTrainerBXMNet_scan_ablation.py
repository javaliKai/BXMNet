# nnUNetTrainerBXMNet.py
import torch
import torch.nn as nn

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# IMPORTANT: adjust this import to wherever you put BXMNet
# Example: if you place bxmnet.py next to this trainer file, you can do:
from nnunetv2.training.nnUNetTrainer.bxmnet.bxmnet import BXMNet
from nnunetv2.utilities.custom_profiling import write_model_profile


class nnUNetTrainerBXMNet_scan_ablation(nnUNetTrainer):
    """
    BXMNet plugged into nnUNetv2 with Deep Supervision ON.

    Requirements:
      - BXMNet forward(x) must work without extra required args
      - If deep_supervision=True, model returns List[logits] (highest-res first)
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # your preference
        self.num_epochs = 250

        # enable DS in nnU-Net pipeline (loss wrapper + target downsampling transforms)
        self.enable_deep_supervision = True

    @staticmethod
    def build_network_architecture(
        architecture_class_name: str,
        arch_init_kwargs: dict,
        arch_init_kwargs_req_import,
        num_input_channels: int,
        num_output_channels: int,
        enable_deep_supervision: bool = True,
    ) -> nn.Module:
        """
        nnU-Net passes arch_init_kwargs from plans.
        We inject the dynamic parts: input channels, num classes, deep supervision.
        """
        return BXMNet()

    # CRITICAL: prevent nnU-Net from assuming `network.decoder.deep_supervision` exists
    def set_deep_supervision_enabled(self, enabled: bool):
        """
        nnU-Net base may try to set mod.decoder.deep_supervision.
        Your model exposes `self.deep_supervision`, so we toggle that safely.
        """
        mod = self.network.module if hasattr(self.network, "module") else self.network

        # preferred: toggle your model flag
        if hasattr(mod, "deep_supervision"):
            try:
                mod.deep_supervision = enabled
            except Exception:
                pass

        # optional compatibility if your decoder exposes it
        if hasattr(mod, "decoder") and hasattr(mod.decoder, "deep_supervision"):
            try:
                mod.decoder.deep_supervision = enabled
            except Exception:
                pass

    def initialize(self):
        # Let nnU-Net do all its setup (plans, patch size, network build, etc.)
        super().initialize()
        #return  #safeguard from conflict with main training

        # Write profile once
        if getattr(self, "_profile_written", False):
            return

        try:
            from nnunetv2.utilities.custom_profiling import write_model_profile
            # optional: you can set a tag for this ablation
            write_model_profile(self, run_tag="scan_ablation_axial_coronal_sagittal_3")
            self._profile_written = True
        except Exception as e:
            print(f"[profile][WARN] failed to write model profile: {e}")

