import pandas as pd
from typing import List, Optional


class TrainingMetricsWrapper:
    """
    Wrapper for metrics.csv DataFrame (or any other analogue dataframe).
    Provides methods to extract training and validation loss trends ready for plotting.
    """

    def __init__(self, metrics_df: pd.DataFrame):
        self.metrics_df = metrics_df

    def _extract_trend(self, metric_names: List[str]) -> Optional[List[float]]:
        """
        Protected helper to extract a trend from the first available metric column.

        Args:
            metric_names: list of possible column names in order of priority

        Returns:
            list of metric values, skipping NaN, or None if no column is found
        """
        for name in metric_names:
            if name in self.metrics_df.columns:
                values = self.metrics_df[name].dropna().tolist()
                if values:
                    return values
        # fallback if no columns found
        return None

    def train_loss_trend(self) -> Optional[List[float]]:
        """
        Returns training loss trend for plotting.
        Looks for 'train_loss_epoch' only.
        """
        return self._extract_trend(["train_loss_epoch"])

    def val_loss_trend(self) -> Optional[List[float]]:
        """
        Returns validation loss trend for plotting.
        Looks for 'val_loss', if it's not found, falls back to 'val_loss_epoch'.
        """
        trend = self._extract_trend(["val_loss"])
        if trend is None:
            trend = self._extract_trend(["val_loss_epoch"])
        return trend

    def train_rmse_trend(self) -> Optional[List[float]]:
        """
        Returns training rmse trend for plotting.
        """
        return self._extract_trend(['train_RMSE'])

    def val_rmse_trend(self) -> Optional[List[float]]:
        """
        Returns validation rmse trend for plotting.
        """
        return self._extract_trend(['val_RMSE'])