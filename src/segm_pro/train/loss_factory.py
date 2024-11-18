"""Factory of segmentation model losses."""

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
    """Enum defining the available loss modes for segmentation tasks."""

    BINARY = constants.BINARY_MODE
    MULTICLASS = constants.MULTICLASS_MODE


class CommonCE(nn.Module):
    """Common cross-entropy loss implementation for segmentation tasks.

    This class provides a simple way to select between binary and multi-class
    cross-entropy losses based on the provided mode.
    """

    def __init__(self, mode: str):
        """Initialize loss.

        Parameters
        ----------
        mode : str
            Supported values are:
            - LossMode.BINARY.value:  Binary segmentation using 
            SoftBCEWithLogitsLoss.
            - LossMode.MULTICLASS.value: Multiclass segmentation using 
            SoftCrossEntropyLoss.

        """
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
        """Compute the loss between predicted and true segmentation masks.

        Parameters
        ----------
        y_pred : torch.Tensor
            A tensor of shape [batch_size, num_classes, height, width]
            representing the predicted segmentation masks.
        y_true : torch.Tensor
            A tensor of shape [batch_size, height, width]
            representing the true segmentation masks.

        Returns
        -------
        torch.Tensor
            The calculated loss value.

        """
        return self._loss(y_pred, y_true)


class LossType(Enum):
    """Enum defining different loss functions used for segmentation tasks."""

    CE = 'CELoss'
    IOU = 'JaccardLoss'
    DICE = 'DiceLoss'
    FOCAL = 'FocalLoss'
    LOVASZ = 'LovaszLoss'


class ComplexLoss(nn.Module):
    """Combines multiple segmentation loss types with customizable weights.

    This class allows you to define a complex loss function by specifying
    a combination of different segmentation loss types and their respective
    weights.
    """

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
        """Calculate the total loss by combining individual losses.

        This method applies a set of defined losses to the predicted
        (`y_pred`) and true (`y_true`) tensors, each weighted according
        to their importance specified in `self._weights`. The individual
        losses are then summed to produce the overall loss value.

        Parameters
        ----------
        y_pred : torch.Tensor
            Predicted segmentation map tensor
        y_true : torch.Tensor
            True segmentation map tensor.

        Returns
        -------
        torch.Tensor
            Total loss value as a single tensor.

        """
        losses = torch.stack([
            loss(y_pred, y_true) * w
            for loss, w in zip(self._losses, self._weights)
        ])
        return torch.sum(losses)


def create_loss(
        losses: tuple[LossType, ...],
        weights: tuple[float, ...],
        loss_mode: LossMode
) -> ComplexLoss:
    """Create a complex loss function combining multiple loss types.

    Parameters
    ----------
    losses : tuple[LossType, ...]
        A tuple of desired loss types to combine. 
        Each element should be a LossType enum value, such as
        LossType.CE, LossType.IOU, etc.
    weights : tuple[float, ...]
        A tuple of corresponding weights for each loss type.
        The weights determine the contribution of each loss to the
        final output.
    loss_mode : LossMode
        The mode of the classification, either BINARY or MULTICLASS.

    Returns
    -------
    ComplexLoss
        An instance of ComplexLoss that combines the specified losses 
        and weights.

    """
    return ComplexLoss(losses, weights, loss_mode)
