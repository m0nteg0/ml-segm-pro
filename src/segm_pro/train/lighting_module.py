import torch
import depth_pro
import lightning as L
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

class SegmentationModule(L.LightningModule):
    def __init__(self, warmup_steps: int = 30):
        super().__init__()
        self._model, _ = (
            depth_pro.create_model_and_transforms()
        )
        self.save_hyperparameters()
        # self.__warmup_steps = warmup_steps

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        x, y = batch
        prediction = self._model(x)
        loss = F.binary_cross_entropy_with_logits(prediction, y)
        self.log('train_loss', loss.item(), True, on_step=True)

        lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
        self.log('lr', lr, True, on_step=True)
        return loss

    @property
    def model(self):
        return self._model

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self._model.parameters(), lr=1e-4)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]