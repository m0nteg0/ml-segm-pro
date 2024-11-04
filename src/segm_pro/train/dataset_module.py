from pathlib import Path

import lightning as L
import albumentations as A
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2

from .dataset import SegmDataset


class SegmDSModule(L.LightningDataModule):
    def __init__(self, path: Path, train_bs: int, val_bs: int):
        super().__init__()
        self.__ds_root = path
        self.__train_bs = train_bs
        self.__val_bs = val_bs

    def setup(self, stage: str) -> None:
        self.__train_ds = SegmDataset(
            self.__ds_root, self.__get_train_transform()
        )
        self.__val_ds = SegmDataset(
            self.__ds_root, self.__get_val_transform()
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
        return DataLoader(self.__train_ds, batch_size=self.__train_bs)

    def val_dataloader(self):
        return DataLoader(self.__val_ds, batch_size=self.__val_bs)
