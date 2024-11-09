import argparse
from pathlib import Path

import yaml
import lightning as L
from segm_pro.train import (
    SegmentationModule,
    SegmDSModule,
    TrainParams
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        '--config', type=Path, required=True,
        help='Path to train config *.yaml'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open('r') as file:
        config = yaml.safe_load(file)

    train_params = TrainParams(**config['train'])

    ds_path = Path(
        '/home/andrey/Develop/deep_vision/ml-segm-pro/data/datasets/human-seg-of'
    )
    data_module = SegmDSModule(ds_path, 1, 1)
    model = SegmentationModule(train_params)

    trainer = L.Trainer(gradient_clip_val=0.5, gradient_clip_algorithm="value")
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()