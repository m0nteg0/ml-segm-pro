"""Class that manages the training process for a segmentation model."""

import argparse
from pathlib import Path
from shutil import copy

import yaml
import lightning as L
from clearml import Task
from lightning.pytorch.callbacks import ModelCheckpoint

from .metrics_factory import MetricType
from .segmentation_module import SegmentationModule, TrainParams
from .data_module import SegmDSModule, DataParams


class TrainManager:
    """Manages the training process for a segmentation model.

    This class orchestrates the entire training pipeline, from parsing
    configuration files to initializing models, data modules, and trainers.
    It also handles saving checkpoints and logging progress to
    ClearML if specified.
    """

    def __init__(self):
        args = self._parse_args()
        with args.config.open('r') as file:
            config = yaml.safe_load(file)

        # Try to get experiment name.
        try:
            exp_name = config['logging']['exp_name']
        except Exception as e:
            raise RuntimeError(
                'Failed to find <exp_name> field in logging section.'
                '<project_name> is required attribute.'
            ) from e

        # Init clearml task.
        if args.clearml:
            try:
                proj_name = config['logging']['project_name']
            except Exception as e:
                raise RuntimeError(
                    'Failed to find <project_name> field in logging section, '
                    '<project_name> is required attribute for clearml logging.'
                ) from e
            Task.init(
                project_name=proj_name, task_name=exp_name
            )
            Task.current_task().upload_artifact(
                'config', artifact_object=args.config
            )

        # Get train and data params
        train_params = TrainParams(**config['train'])
        data_params = DataParams(**config['data'])

        # Init data module
        self._data_module = SegmDSModule(data_params)

        # Init model
        if args.ckpt:
            self._model = SegmentationModule.load_from_checkpoint(args.ckpt)
        else:
            self._model = SegmentationModule(train_params)

        # Set save directory.
        save_dir = Path(config['logging'].get('default_root_dir', './'))
        save_dir = save_dir / exp_name

        # Copy config to save_dir
        copy(args.config, save_dir / 'config.yaml')

        # Set checkpointing callbacks.
        best_metric = config['logging'].get(
            'best_metric', MetricType.IOU.value
        )
        best_saver = ModelCheckpoint(
            dirpath=save_dir, save_top_k=1,
            every_n_epochs=1, filename=f'best_{best_metric}',
            monitor=best_metric
        )
        last_saver = ModelCheckpoint(
            dirpath=save_dir, save_top_k=0,
            every_n_epochs=1, save_last=True
        )

        # Init lighting trainer.
        epochs = config['train'].get('epochs', 100)
        acc_grad_batches = config['train'].get('acc_grad_batches', 1)
        gradient_clip_val = config['train'].get('gradient_clip_val', 0.5)
        log_every_n_steps = config['logging'].get('log_every_n_steps', 50)
        self._trainer = L.Trainer(
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm='value',
            accumulate_grad_batches=acc_grad_batches,
            default_root_dir=save_dir,
            callbacks=[best_saver, last_saver],
            max_epochs=epochs,
            log_every_n_steps=log_every_n_steps
        )

    def run(self):
        """Run train process."""
        self._trainer.fit(self._model, datamodule=self._data_module)

    def _parse_args(self):
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(
            description=__doc__
        )
        parser.add_argument(
            '--config', type=Path, required=True,
            help='Path to train config *.yaml'
        )
        parser.add_argument(
            '--ckpt', type=Path,
            help='Path to model checlpoint.'
        )
        parser.add_argument(
            '--clearml', action='store_true',
            help='If True, then logging to clearml'
        )
        return parser.parse_args()
