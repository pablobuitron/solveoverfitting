"""
Training utilities for the Grano.IT forecasting pipeline.

This module exposes a thin wrapper (`Trainer`) around PyTorch Forecasting's
`TemporalFusionTransformer.from_dataset` and Lightning's training loop.
It standardizes:
  - configuration parsing
  - logger setup
  - trainer instantiation and fit loop

Notes:
  - Tested with pytorch==2.3.1 and lightning.pytorch==2.5.0 in the project context.
    Newer combinations (e.g., PyTorch >= 2.4 and Lightning >= 2.5.0) were observed
    to trigger MPS backend exceptions on macOS:
      "[MPSNDArray, initWithBufferImpl:offset:descriptor:isForNDArrayAlias:isUserBuffer:]
       Error: buffer is not large enough. Must be XXXX bytes"
    If you encounter this on Apple Silicon, consider pinning to the tested versions
    or falling back to CPU.
"""
import pathlib
import time
import os
import torch
from lightning.pytorch import loggers as lightning_loggers
import pytorch_forecasting
import lightning.pytorch
from lightning.pytorch.callbacks import EarlyStopping
from pytorch_forecasting import metrics as ptf_metrics, TemporalFusionTransformer
import logging

from lightning.pytorch.callbacks import ModelCheckpoint
from .monitoring import StepStatsCallback, ResourceMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    High-level training orchestrator for Temporal Fusion Transformer.

    This class is a small façade over PyTorch Forecasting + Lightning.

    Attributes:
        training_config (dict): Parsed training configuration with concrete objects
            (e.g., `loss` is a `ptf_metrics.QuantileLoss` instance).
        model_architecture_config (dict): Parsed model hyperparameters (forwarded to TFT factory).

    """

    # now that experiments are going, having default configs is pointless. Maybe when experiments are done,
    # these can be set to the best config (so that the user doesn't have to specify it) or can be kept just to have a
    # template for configurations
    default_training_config = None
    default_model_config = None

    @staticmethod
    def _parse_training_config(training_config: dict):
        if training_config is None:
            raise ValueError("No training configuration provided.")

        # Parse loss - could be at top level or in model
        model_training_config = training_config.get('model')
        if not model_training_config:
            raise ValueError("Missing model configuration inside the configuration dictionary. "
                             "Memo: the key to use is 'model'.")

        training_config['model']['loss'] = Trainer._check_and_parse_loss(training_config)
        training_config['model']['logging_metrics']  = Trainer._parse_logging_metrics(training_config)

        return training_config

    @staticmethod
    def _parse_model_architecture_config(model_architecture_config: dict):
        if model_architecture_config is None:
            raise ValueError("No model configuration provided.")
        return model_architecture_config

    @staticmethod
    def _check_and_parse_loss(training_config: dict):

        loss = training_config['model'].get('loss')
        if not loss:
            raise ValueError("Missing loss configuration inside the configuration dictionary. "
                             "Memo: it must be stored under 'model'. The key to use is 'loss'")

        if loss == 'quantile_loss':
            return ptf_metrics.QuantileLoss()
        else:
            raise ValueError('Cannot parse loss {}'.format(loss))

    @staticmethod
    def _parse_logging_metrics(training_config: dict):
        logging_metrics = training_config['model'].get('logging_metrics')
        if logging_metrics is None:
            raise ValueError("Missing logging_metrics configuration inside the configuration dictionary. "
                             "Memo: it must be stored under 'model'. The key to use is 'logging_metrics'")

        result = []
        for metric in logging_metrics:
            if metric == 'rmse':
                result.append(ptf_metrics.RMSE())
            elif metric == 'mae':
                result.append(ptf_metrics.MAE())
            else:
                raise ValueError('Cannot parse metric {}'.format(metric))

        return result

    def __init__(self, training_config: dict = None, model_architecture_config: dict = None):
        """
        Construct a Trainer, parsing and freezing configuration dicts.

        Args:
            training_config (Optional[dict]): Training configuration. If None,
                `default_training_config` is used.
            model_architecture_config (Optional[dict]): Model configuration. If None,
                `default_model_config` is used.
        """
        self.training_config = self._parse_training_config(training_config or self.default_training_config)
        self.model_architecture_config = self._parse_model_architecture_config(model_architecture_config or self.default_model_config)


    def train(self, training_dataset: pytorch_forecasting.TimeSeriesDataSet,
              validation_dataset: pytorch_forecasting.TimeSeriesDataSet,
              output_dir: pathlib.Path = pathlib.Path('./lightning_logs')):
        """
        :param training_dataset:
        :param validation_dataset:
        :param output_dir: Directory where output files such as logs and checkpoints will be saved.
        """
        logger.info('Instantiating model...')
        model_training_config = self.training_config.get('model', {})
        model = TemporalFusionTransformer.from_dataset(
            # dataset
            training_dataset,
            # model architecture - mandatory
            hidden_size=self.model_architecture_config['hidden_size'],
            lstm_layers=self.model_architecture_config['lstm_layers'],
            attention_head_size=self.model_architecture_config['attention_head_size'],
            hidden_continuous_size=self.model_architecture_config['hidden_continuous_size'],
            # training related params stored in the training configuration file - mandatory
            learning_rate=model_training_config['learning_rate'],
            dropout=model_training_config['dropout'],
            loss=model_training_config['loss'],
            reduce_on_plateau_patience=model_training_config['reduce_on_plateau_patience'],
            log_interval=model_training_config['log_interval'],
            logging_metrics=model_training_config['logging_metrics'],
        )
        # Optimize model for MPS and FP16 if mps is detected
        model = self._maybe_model_mps_optimizations(model)
        logger.info('Model instantiated.')

        logger.info('Instantiating lightning trainer...')
        trainer_config = self.training_config.get('trainer', {})
        csv_logger = lightning_loggers.CSVLogger(
            save_dir=output_dir.parent,
            name=output_dir.name,
        )
        # memo: depending on lightning version, the package to be used is lightning.pytorch or pytorch_lightning
        trainer = lightning.pytorch.Trainer(
            # mandatory parameters
            max_epochs=trainer_config['max_epochs'],
            logger=[csv_logger],
            # parameters that can be handled automatically by lightning
            gradient_clip_val=trainer_config.get('gradient_clip_val', None),
            accelerator=trainer_config.get('accelerator', 'auto'),
            devices=trainer_config.get('devices', 'auto'),
            precision=self.training_config.get('precision', None),
            num_sanity_val_steps=trainer_config.get('num_sanity_val_steps', None),
            # values that are not configurable right now:
            enable_progress_bar=True,
            enable_model_summary=True,
        )
        logger.info('Lightning trainer instantiated.')

        logger.info('Instantiating dataloader...')
        dataloader_kwargs = self.training_config.get('dataloader', {})
        train_dataloader = training_dataset.to_dataloader(train=True, **dataloader_kwargs)
        val_dataloader = validation_dataset.to_dataloader(train=False, **dataloader_kwargs)
        logger.info('Dataloader instantiated.')

        logger.info('Training started.')
        start = time.time()
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
        end = time.time()
        logger.info('Training finished.')
        logger.info('Training took {} minutes.'.format((end - start) /  60))

    def _maybe_model_mps_optimizations(self, model):
        """Optimize model for MPS and FP16"""
        if self.training_config.get('precision') == '16-mixed':
            logger.info('Optimizing model for mps when precision is 16-mixed')
            model = model.half()  # Convert model to half precision
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.set_per_process_memory_fraction(0.9)  # Use 90% of MPS memory
                # Set MPS device for model
                model.to(torch.device('mps'))
        return model
