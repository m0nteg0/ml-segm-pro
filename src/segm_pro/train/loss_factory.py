from enum import Enum

import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import (
    constants,
    JaccardLoss,
    DiceLoss,
    FocalLoss,
    LovaszLoss,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss
)


class LossMode(Enum):
    BINARY = constants.BINARY_MODE
    MULTICLASS = constants.MULTICLASS_MODE


class CommonCE(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self._loss = (
            SoftBCEWithLogitsLoss()
            if mode == LossMode.BINARY.value
            else SoftCrossEntropyLoss()
        )

    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor
    ) -> torch.Tensor:
        return self._loss(y_pred, y_true)


class LossType(Enum):
    CE = 'CommonCE'
    IOU = 'JaccardLoss'
    DICE = 'DiceLoss'
    FOCAL = 'FocalLoss'
    LOVASZ = 'LovaszLoss'


class ComplexLoss(nn.Module):
    def __init__(
            self,
            losses: tuple[LossType, ...],
            weights: tuple[float, ...],
            loss_mode: LossMode
    ):
        super().__init__()
        if len(losses) != len(weights):
            raise ValueError(
                'The sizes of losses and weights do not match: '
                f'losses: {len(losses)}, weights: {len(weights)}'
            )
        self._weights = weights
        self._losses = [self._create_loss(lt, loss_mode) for lt in losses]

    def _create_loss(
            self,
            loss_type: LossType,
            loss_mode: LossMode
    ):
        factory = {
            LossType.CE.value: CommonCE,
            LossType.IOU.value: JaccardLoss,
            LossType.DICE.value: DiceLoss,
            LossType.FOCAL.value: FocalLoss,
            LossType.LOVASZ.value: LovaszLoss,
        }
        return factory[loss_type.value](mode=loss_mode.value)

    def forward(
            self,
            y_pred: torch.Tensor,
            y_true: torch.Tensor
    ) -> torch.Tensor:
        losses = torch.stack([
            loss(y_pred, y_true) * w
            for loss, w in zip(self._losses, self._weights)
        ])
        return torch.sum(losses)


def create_loss(
        losses: tuple[LossType, ...],
        weights: tuple[float, ...],
        loss_mode: LossMode
):
    return ComplexLoss(losses, weights, loss_mode)
