from pathlib import Path

import lightning as L
from segm_pro.train import SegmentationModule, SegmDSModule


def main():
    ds_path = Path(
        '/home/andrey/Develop/deep_vision/ml-segm-pro/data/datasets/human-seg-of'
    )
    data_module = SegmDSModule(ds_path, 1, 1)
    model = SegmentationModule()

    trainer = L.Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()