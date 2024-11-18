"""Factory of segmentation model metrics."""

from enum import Enum

from torchmetrics import Metric
from torchmetrics.segmentation import (
    MeanIoU,
    GeneralizedDiceScore,
    HausdorffDistance
)


class MetricType(Enum):
    """Enum defining the metrics for segmentation model evaluation."""

    IOU = 'MeanIoU'
    DICE = 'Dice'
    HD = 'HausdorffDistance'


def create_metric(metric_type: MetricType, num_classes: int = 1) -> Metric:
    """Create a segmentation metric based on the specified type.

    Parameters
    ----------
    metric_type : MetricType
        The type of metric to create. Supported types are:
        - MetricType.IOU: Mean Intersection over Union (mIoU)
        - MetricType.DICE: Generalized Dice Score
        - MetricType.HD: Hausdorff Distance
    num_classes : int, optional
        The number of classes in the segmentation task, by default 1

    Returns
    -------
    Metric
        An instance of the selected metric class.

    """
    fabric = {
        MetricType.IOU.value: MeanIoU,
        MetricType.DICE.value: GeneralizedDiceScore,
        MetricType.HD.value: HausdorffDistance,
    }
    return fabric[metric_type.value](num_classes)
