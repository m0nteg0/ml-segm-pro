import torch
import depth_pro
import lightning as L
from transformers import get_linear_schedule_with_warmup
from pydantic import BaseModel, Field

from .metrics_factory import MetricType, create_metric
from .loss_factory import LossType, LossMode, create_loss


class TrainParams(BaseModel):
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


class SegmentationModule(L.LightningModule):
    def __init__(
            self,
            params: TrainParams
    ):
        super().__init__()
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

    @property
    def model(self):
        return self._model

    def _init_metrics(self, metrics: tuple[MetricType, ...]):
        self._metrics = {}
        for metric_type in metrics:
            self._metrics[metric_type.value] = create_metric(
                metric_type
            ).to('cuda')

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
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
        self.log('lr', lr, True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the validation loop.
        x, y = batch
        prediction = self._model(x)
        loss = self._loss(prediction, y)
        self.log(
            'val_loss', loss.item(), True, on_step=False,
            on_epoch=True, logger=True
        )

        prediction = (torch.sigmoid(prediction) > 0.5).int()
        for metric in self._metrics:
            self._metrics[metric](prediction, y.int())

        return loss

    def on_validation_epoch_end(self) -> None:
        for metric in self._metrics:
            value = self._metrics[metric].compute().item()
            self.log(metric, value, prog_bar=True, logger=True)

    def configure_optimizers(self):
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
