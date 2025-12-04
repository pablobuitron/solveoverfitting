import pathlib
import time

import pandas as pd
import torch
from pytorch_forecasting import metrics as ptf_metrics
import logging

from torch import nn

from ..models.lstm import YieldLSTM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """High-level training orchestrator for the Grano.IT LSTM model."""

    default_training_config = None
    default_model_config = None

    @staticmethod
    def _parse_training_config(training_config: dict):
        if training_config is None:
            raise ValueError("No training configuration provided.")
        model_training_config = training_config.get('model')
        if not model_training_config:
            raise ValueError(
                "Missing model configuration inside the configuration dictionary. "
                "Key must be 'model'."
            )
        training_config['model']['loss'] = Trainer._check_and_parse_loss(training_config)
        training_config['model']['logging_metrics'] = Trainer._parse_logging_metrics(training_config)
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
            raise ValueError("Missing loss configuration inside the configuration dictionary. Key must be 'loss'.")
        if loss == 'quantile_loss':
            return ptf_metrics.QuantileLoss()
        else:
            raise ValueError(f"Cannot parse loss {loss}")

    @staticmethod
    def _parse_logging_metrics(training_config: dict):
        logging_metrics = training_config['model'].get('logging_metrics')
        if logging_metrics is None:
            raise ValueError("Missing logging_metrics configuration inside the configuration dictionary. Key must be 'logging_metrics'.")
        result = []
        for metric in logging_metrics:
            if metric == 'rmse':
                result.append(ptf_metrics.RMSE())
            elif metric == 'mae':
                result.append(ptf_metrics.MAE())
            else:
                raise ValueError(f"Cannot parse metric {metric}")
        return result

    def __init__(self, training_config: dict = None, model_architecture_config: dict = None):
        self.training_config = self._parse_training_config(
            training_config or self.default_training_config
        )
        self.model_architecture_config = self._parse_model_architecture_config(
            model_architecture_config or self.default_model_config
        )

    def _prepare_features(self, batch, n_dynamic_reals, n_static_reals):
        """Extract x_dynamic and x_static from batch in a consistent way."""
        batch_inputs, batch_target = batch
        encoder_cont = batch_inputs["encoder_cont"]

        # Dynamic features
        x_dynamic = encoder_cont[..., :n_dynamic_reals]

        # Static real features
        static_real_slice = encoder_cont[..., n_dynamic_reals:n_dynamic_reals + n_static_reals]
        x_static_real = static_real_slice[:, 0, :]  # take first timestep

        # Static categorical features
        x_static_cat = batch_inputs.get("encoder_cat", None)
        if x_static_cat is not None:
            # take first timestep if shape [B, T, F]
            if x_static_cat.dim() == 3:
                x_static_cat = x_static_cat[:, 0, :]
            x_static = torch.cat([x_static_real, x_static_cat.float()], dim=1)
        else:
            x_static = x_static_real

        if x_static.shape[1] == 0:
            x_static = None

        # Target
        y = batch_target[0]
        return x_dynamic, x_static, y

    def train(self, training_dataset, validation_dataset, output_dir=pathlib.Path("./lightning_logs")):
        """Main training entry point for LSTM yield prediction."""

        # ----------------------------------------------------------------------
        # Determine device
        # ----------------------------------------------------------------------
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS device for training.")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device for training.")

        # ----------------------------------------------------------------------
        # Dataset metadata
        # ----------------------------------------------------------------------
        static_cat_count = len(training_dataset.static_categoricals)
        static_real_count = len(training_dataset.static_reals)
        static_features_count = static_cat_count + static_real_count

        dynamic_real_names = training_dataset.time_varying_unknown_reals
        static_real_names = training_dataset.static_reals

        n_dynamic_reals = len(dynamic_real_names)
        n_static_reals = len(static_real_names)

        # ----------------------------------------------------------------------
        # Model
        # ----------------------------------------------------------------------
        logger.info("Instantiating model...")
        model = YieldLSTM(n_dynamic_reals, 128, 2, static_features_count)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        # ----------------------------------------------------------------------
        # Dataloaders
        # ----------------------------------------------------------------------
        dataloader_kwargs = self.training_config.get("dataloader", {})
        train_dataloader = training_dataset.to_dataloader(train=True, **dataloader_kwargs)
        val_dataloader = validation_dataset.to_dataloader(train=False, **dataloader_kwargs)

        # ----------------------------------------------------------------------
        # Output directories
        # ----------------------------------------------------------------------
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        metrics_path = output_dir / "metrics.csv"
        metrics_df = pd.DataFrame(
            columns=["epoch", "train_loss_epoch", "val_loss_epoch", "train_RMSE", "val_RMSE"]
        )
        metrics_df.to_csv(metrics_path, index=False)

        logger.info("Training started.")
        start = time.time()

        # ----------------------------------------------------------------------
        # TRAINING LOOP
        # ----------------------------------------------------------------------
        max_epochs = self.training_config["trainer"]["max_epochs"]

        for epoch in range(max_epochs):
            model.train()
            train_losses = []

            # ---------------------------
            # TRAIN
            # ---------------------------
            for batch in train_dataloader:
                x_dynamic, x_static, y = self._prepare_features(batch, n_dynamic_reals, n_static_reals)

                x_dynamic = x_dynamic.to(device)
                if x_static is not None:
                    x_static = x_static.to(device)
                y = y.to(device)

                optimizer.zero_grad()
                y_pred = model(x_dynamic, x_static)
                loss = criterion(y_pred.squeeze(), y.squeeze())
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            train_loss_epoch = sum(train_losses) / len(train_losses)

            # ---------------------------
            # VALIDATION
            # ---------------------------
            model.eval()
            val_losses = []
            train_rmses = []
            val_rmses = []

            with torch.no_grad():
                # Validation RMSE
                for batch in val_dataloader:
                    x_dynamic, x_static, y = self._prepare_features(batch, n_dynamic_reals, n_static_reals)

                    x_dynamic = x_dynamic.to(device)
                    if x_static is not None:
                        x_static = x_static.to(device)
                    y = y.to(device)

                    y_pred = model(x_dynamic, x_static)
                    loss = criterion(y_pred.squeeze(), y.squeeze())
                    val_losses.append(loss.item())

                    rmse = torch.sqrt(torch.mean((y_pred.squeeze() - y) ** 2)).item()
                    val_rmses.append(rmse)

                # Train RMSE over entire dataset
                for batch in train_dataloader:
                    x_dynamic, x_static, y = self._prepare_features(batch, n_dynamic_reals, n_static_reals)

                    x_dynamic = x_dynamic.to(device)
                    if x_static is not None:
                        x_static = x_static.to(device)
                    y = y.to(device)

                    y_pred = model(x_dynamic, x_static)
                    rmse = torch.sqrt(torch.mean((y_pred.squeeze() - y) ** 2)).item()
                    train_rmses.append(rmse)

            val_loss_epoch = sum(val_losses) / len(val_losses)
            val_rmse_epoch = sum(val_rmses) / len(val_rmses)
            train_rmse_epoch = sum(train_rmses) / len(train_rmses)

            # ----------------------------------------------------------------------
            # LOGGING AND CHECKPOINT
            # ----------------------------------------------------------------------
            with open(metrics_path, "a") as f:
                f.write(
                    f"{epoch},{train_loss_epoch},{val_loss_epoch},"
                    f"{train_rmse_epoch},{val_rmse_epoch}\n"
                )

            checkpoint_path = checkpoints_dir / f"epoch_{epoch}.pt"
            torch.save(model.state_dict(), checkpoint_path)

            logger.info(
                "Epoch %d | train_loss=%.4f | val_loss=%.4f | train_RMSE=%.4f | val_RMSE=%.4f",
                epoch,
                train_loss_epoch,
                val_loss_epoch,
                train_rmse_epoch,
                val_rmse_epoch,
            )

        end = time.time()
        logger.info("Training finished in %.2f minutes.", (end - start) / 60)

    def _maybe_model_mps_optimizations(self, model):
        if self.training_config.get('precision') == '16-mixed':
            logger.info("Optimizing model for mps with 16-mixed precision")
            model = model.half()
            if hasattr(torch, 'mps') and torch.backends.mps.is_available():
                torch.mps.set_per_process_memory_fraction(0.9)
                model.to(torch.device('mps'))
        return model
