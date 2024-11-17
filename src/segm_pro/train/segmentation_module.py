"""A PyTorch Lightning module for training segmentation models."""

import torch
import depth_pro
import lightning as L
from torchvision.utils import make_grid
from transformers import get_linear_schedule_with_warmup
from pydantic import BaseModel, Field

from .metrics_factory import MetricType, create_metric
from .loss_factory import LossType, LossMode, create_loss


class TrainParams(BaseModel):
    """Parameters for training a segmentation model."""

    lr: float = Field(
        ge=0,
        default=1e-5,
        description='Learning rate value.'
    )

    warmup_steps: int = Field(
        ge=0,
        default=30,
        description='Number of warmup iterations'
    )

    metrics: list[MetricType] = Field(
        default=[MetricType.IOU],
        description='List of segmentation metrics'
    )

    losses: list[LossType] = Field(
        default=[LossType.CE, LossType.DICE],
        description='List of segmentation losses'
    )

    loss_weights: list[float] = Field(
        default=[0.5, 0.5],
        description='List of losses weights'
    )

    loss_mode: LossMode = Field(
        default=LossMode.BINARY,
        description='Loss type: "binary", "multiclass"'
    )

    device: int = Field(
        ge=0,
        default=0,
        description='Device number'
    )

    n_debug_images: int = Field(
        ge=0,
        default=5,
        description='Number debug images'
    )


class SegmentationModule(L.LightningModule):
    """A PyTorch Lightning module for training segmentation models.

    This module defines the architecture and training loop for a
    segmentation task. It utilizes pre-defined model creation, loss functions,
    and metrics based on provided parameters.
    """

    def __init__(
            self,
            params: TrainParams | None = None
    ):
        super().__init__()
        params = params if params is not None else TrainParams()
        self._metrics = {}
        self._model, _ = (
            depth_pro.create_model_and_transforms()
        )
        self._init_metrics(tuple(params.metrics))
        self._loss = create_loss(
            tuple(params.losses),
            tuple(params.loss_weights), params.loss_mode
        )
        self._train_params = params
        self._debug_images = []
        self._debug_preds = []
        self._n_debug_images = params.n_debug_images

    @property
    def model(self):
        """Return segmentation model."""
        return self._model

    def _init_metrics(self, metrics: tuple[MetricType, ...]):
        """Initialize segmentation metrics."""
        self._metrics = {}
        for metric_type in metrics:
            self._metrics[metric_type.value] = create_metric(
                metric_type
            ).to('cuda')

    def training_step(
            self,
            batch : tuple[torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step.

        Parameters
        ----------
        batch : tuple[torch.Tensor]
            A tuple containing the input images, target masks, and other
            relevant data.
        batch_idx : _type_
            The index of the current batch.

        Returns
        -------
        torch.Tensor
            The calculated loss for the batch.

        """
        _, x, y = batch
        prediction = self._model(x)

        loss = self._loss(prediction, y)
        self.log(
            'batch_train_loss', loss.item(), True, on_step=True,
            on_epoch=False, logger=False
        )
        self.log(
            'train_loss', loss.item(), True, on_step=False,
            on_epoch=True, logger=True
        )

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log('lr', lr, True, on_step=True, logger=False)

        return loss

    def validation_step(
            self,
            batch : tuple[torch.Tensor],
            batch_idx: int
    ):
        """Perform a single validation step.

        Parameters
        ----------
        batch : tuple[torch.Tensor]
            A tuple containing the input images, target masks, and other
            relevant data.
        batch_idx : int
            The index of the current batch.

        """
        images, x, y = batch
        prediction = self._model(x)
        loss = self._loss(prediction, y)
        self.log(
            'val_loss', loss.item(), True, on_step=False,
            on_epoch=True, logger=True
        )
        # Accumulate metrics
        prediction = (torch.sigmoid(prediction) > 0.5).int()
        for metric in self._metrics:
            self._metrics[metric](prediction, y.int())
        # Log debug images
        images_diff = (
                self._n_debug_images - len(self._debug_images)
        )
        if images_diff > 0 and self._n_debug_images > 0:
            self._debug_images = [*self._debug_images, images[:images_diff]]
            self._debug_preds = [*self._debug_preds, prediction[:images_diff]]
            if len(self._debug_images) == self._n_debug_images:
                self._log_debug_images()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics and clears debug images.

        This method iterates through the configured metrics,
        computes their values for the current validation epoch, and logs
        them using `self.log`. The `prog_bar` and `logger` arguments ensure
        that the metrics are displayed in the progress bar and written to
        the specified logger.

        Additionally, it clears the lists `self._debug_images` and
        `self._debug_preds`, which likely store images and corresponding
        predictions for debugging purposes.
        """
        for metric in self._metrics:
            value = self._metrics[metric].compute().item()
            self.log(metric, value, prog_bar=True, logger=True)
        self._debug_images = []
        self._debug_preds = []

    def _log_debug_images(self):
        """Log debug images."""
        tensorboard = self.logger.experiment
        images = torch.concat(self._debug_images, 0)
        prediction = torch.concat(self._debug_preds, 0)

        images = torch.permute(images, [0, 3, 1, 2])
        images = torch.flip(images, (1,)).cpu().float()

        prediction = torch.repeat_interleave(prediction, 3, 1)
        prediction = prediction.detach().cpu().float() * 255

        pred_mask = torch.tensor([0, 1, 0], dtype=torch.float32)
        pred_mask = pred_mask.view(1, 3, 1, 1)
        prediction *= pred_mask

        images = (images * 0.7 + prediction * 0.3).to(dtype=torch.uint8)
        tensorboard.add_image(
            "val_images", make_grid(images, 3), self.current_epoch
        )

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for training."""
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._train_params.lr
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self._train_params.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1
        }
        return [optimizer], [scheduler]
