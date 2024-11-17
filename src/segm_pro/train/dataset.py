from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset


class SegmDataset(Dataset):
    def __init__(self, path: Path, transform: Any):
        super().__init__()
        self.__ds_root = path
        self.__images_dir = path / 'images'
        self.__masks_dir = path / 'masks'
        self.__names = [
            x.stem for x in self.__images_dir.glob('*.png')
        ]
        self.__transform = transform

    def __len__(self):
        return len(self.__names)

    def __getitem__(self, item):
        # Get paths
        name = self.__names[item]
        image_path = self.__images_dir / f'{name}.png'
        mask_path = self.__masks_dir / f'{name}.png'
        # Load image and correspond mask
        image = cv2.imread(str(image_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        # Apply transforms
        transformed = self.__transform(image=image, mask=mask)
        transformed_image = transformed['image']
        transformed_mask = transformed['mask'].float() / 255
        transformed_mask = transformed_mask.unsqueeze(0)

        image = cv2.resize(
            image, transformed_image.shape[-2:],
            interpolation=cv2.INTER_CUBIC
        )

        return image, transformed_image, transformed_mask