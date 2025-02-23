from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F

from optispeech.utils.model import make_non_pad_mask


def duration_loss(logw, logw_, lengths, use_log=False):
    if use_log:
        loss = torch.sum((logw - logw_) ** 2) / torch.sum(lengths)
    else:
        loss = torch.sum((torch.exp(logw) - torch.exp(logw_)) ** 2) / torch.sum(lengths)
    return loss


class FastSpeech2Loss(torch.nn.Module):
    """
    Loss function module for FastSpeech2.
    Taken from ESPnet2
    """

    def __init__(self, regression_loss_type: str="l1", use_masking: bool = True, use_weighted_masking: bool = False):
        """
        Initialize feed-forward Transformer loss module.

        Args:
            regression_loss_type: one of {"mse", "l1"}
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to weighted masking in loss
                calculation.

        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        if regression_loss_type == "mse":
            self.regression_criterion = torch.nn.MSELoss(reduction=reduction)
        elif regression_loss_type == "l1":
            self.regression_criterion = torch.nn.SmoothL1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown regression loss type: {regression_loss_type}")

    def forward(
        self,
        p_outs: torch.Tensor,
        e_outs: torch.Tensor,
        ps: torch.Tensor,
        es: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            p_outs (Tensor): Batch of outputs of pitch predictor (B, T_text, 1).
            e_outs (Tensor): Batch of outputs of energy predictor (B, T_text, 1).
            ps (Tensor): Batch of target token-averaged pitch (B, T_text, 1).
            es (Tensor): Batch of target token-averaged energy (B, T_text, 1).
            ilens (LongTensor): Batch of the lengths of each input (B,).

        Returns:
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            with torch.no_grad():
                pitch_masks = make_non_pad_mask(ilens).unsqueeze(-1).to(ps.device)
            p_outs = p_outs.masked_select(pitch_masks)
            ps = ps.masked_select(pitch_masks)
            e_outs = e_outs.masked_select(pitch_masks)
            es = es.masked_select(pitch_masks)

        # calculate loss
        pitch_loss = self.regression_criterion(p_outs, ps)
        energy_loss = self.regression_criterion(e_outs, es)
        return pitch_loss, energy_loss

