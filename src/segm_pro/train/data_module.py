from pathlib import Path

import lightning as L
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2
from pydantic import BaseModel, Field

from .dataset import SegmDataset


class DataParams(BaseModel):
    train_data: Path = Field(
        description='Path to train dataset.'
    )

    val_data: Path = Field(
        description='Path to validation dataset.'
    )

    train_bs: int = Field(
        ge=1,
        default=1,
        description='Train batch size.'
    )

    val_bs: int = Field(
        ge=1,
        default=1,
        description='Validation batch size.'
    )

    train_workers: int = Field(
        ge=0,
        default=0,
        description='Number of train workers.'
    )

    val_workers: int = Field(
        ge=0,
        default=0,
        description='Number of validation workers.'
    )


class SegmDSModule(L.LightningDataModule):
    def __init__(self, params: DataParams):
        super().__init__()
        self._data_params = params

    def setup(self, stage: str) -> None:
        self.__train_ds = SegmDataset(
            self._data_params.train_data, self.__get_train_transform()
        )
        self.__val_ds = SegmDataset(
            self._data_params.val_data, self.__get_val_transform()
        )

    def __get_train_transform(self):
        return A.Compose([
            A.Resize(768, 768),
            # A.ShiftScaleRotate(
            #     shift_limit=0.2, scale_limit=0.2,
            #     rotate_limit=30, p=0.5
            # ),
            # A.RGBShift(
            #     r_shift_limit=25, g_shift_limit=25,
            #     b_shift_limit=25, p=0.5
            # ),
            # A.RandomBrightnessContrast(
            #     brightness_limit=0.3, contrast_limit=0.3, p=0.5
            # ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def __get_val_transform(self):
        return A.Compose([
            A.Resize(768, 768),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])

    def train_dataloader(self):
        return DataLoader(
            self.__train_ds,
            batch_size=self._data_params.train_bs,
            num_workers=self._data_params.train_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.__val_ds,
            batch_size=self._data_params.val_bs,
            num_workers=self._data_params.val_workers
        )
