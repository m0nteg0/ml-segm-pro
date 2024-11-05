from enum import Enum

from torchmetrics.segmentation import (
    MeanIoU,
    GeneralizedDiceScore,
    HausdorffDistance
)


class MetricType(Enum):
    IOU = 'MeanIoU',
    DICE = 'Dice'
    HD = 'HausdorffDistance'


def create_metric(metric_type: MetricType, num_classes: int = 1):
    fabric = {
        MetricType.IOU.value: MeanIoU,
        MetricType.DICE.value: GeneralizedDiceScore,
        MetricType.HD.value: HausdorffDistance,
    }
    return fabric[metric_type.value](num_classes)