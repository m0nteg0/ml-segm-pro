import argparse
from pathlib import Path

import yaml
import lightning as L
from clearml import Task
from segm_pro.train import (
    SegmentationModule,
    SegmDSModule,
    TrainParams,
    DataParams
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        '--config', type=Path, required=True,
        help='Path to train config *.yaml'
    )
    parser.add_argument(
        '--clearml', action='store_true',
        help='If True, then logging to clearml'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open('r') as file:
        config = yaml.safe_load(file)

    if args.clearml:
        Task.init(
            project_name="segm-pro", task_name="test_launch"
        )

    train_params = TrainParams(**config['train'])
    data_params = DataParams(**config['data'])

    data_module = SegmDSModule(data_params)
    model = SegmentationModule(train_params)

    trainer = L.Trainer(
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value",
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    main()