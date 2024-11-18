"""A custom PyTorch dataset for segmentation tasks."""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset


class SegmDataset(Dataset):
    """A custom PyTorch dataset for segmentation tasks.

    This dataset loads images and corresponding masks from a directory
    structure. It applies transformations to both the image and mask before
    returning them.
    """

    def __init__(self, path: Path, transform: Any):
        """Initialize the SegmentationDataset object.

        Parameters
        ----------
        path : Path
            The root directory containing 'images' and 'masks' subdirectories.
        transform : Any
            A function or class that will be applied to each image during 
            data loading.

        """
        super().__init__()
        self.__images_dir = path / 'images'
        self.__masks_dir = path / 'masks'
        self.__names = [
            x.stem for x in self.__images_dir.glob('*.png')
        ]
        self.__transform = transform

    def __len__(self):
        """Return dataset length."""
        return len(self.__names)

    def __getitem__(
            self, item: int
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """Retrieve a single training sample from the dataset.

        Parameters
        ----------
        item : int
            Index of the desired sample within the dataset.

        Returns
        -------
        tuple[np.ndarray, torch.Tensor, torch.Tensor]
             tuple containing three tensors:
            - image: The original image loaded from disk. Resized to match the
            transformed image size.
            - transformed_image: The image tensor after applying
            transformations defined in self.__transform.
            - transformed_mask: The corresponding mask tensor after applying
            transformations and normalization (scaled to [0, 1]).

        """
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
