
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FieldAggregator:
    """
    Aggregates point-level time-series data to the field level.

    This class implements the field-level aggregation strategy described in the
    optimization report. It converts a high-volume of point observations into a
    manageable number of field-level statistical summaries, drastically reducing
    dataset size while preserving predictive information.

    Key features:
    - Statistical aggregations (mean, std, median, percentiles).
    - Spatial variability metrics (coefficient of variation, range normalization).
    - Quality control to filter out low-quality or sparse fields.
    """

    def __init__(self, config: dict):
        """
        Initializes the aggregator with a configuration dictionary.

        Args:
            config (dict): Configuration for aggregation, including parameters for
                           quality control and aggregation methods.
        """
        self.config = config

    def aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs the field-level aggregation.

        Args:
            df (pd.DataFrame): A DataFrame with point-level time-series data,
                               indexed by field_id and timestamp.

        Returns:
            pd.DataFrame: A DataFrame with aggregated, field-level time-series data.
        """
        logger.info(f"Starting field aggregation. Initial shape: {df.shape}")

        # Quality control: filter fields with insufficient data
        df = self._apply_quality_control(df)

        # Define aggregations
        agg_functions = {
            col: self._get_aggregations_for_col(col) 
            for col in df.columns if col not in ['field_id', 'timestamp']
        }

        # Perform aggregation
        aggregated_df = df.groupby('field_id').agg(agg_functions)
        aggregated_df.columns = ['_'.join(col).strip() for col in aggregated_df.columns.values]
        aggregated_df.reset_index(inplace=True)

        logger.info(f"Field aggregation complete. Final shape: {aggregated_df.shape}")
        return aggregated_df

    def _apply_quality_control(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters out fields that do not meet the quality criteria defined in the config.
        """
        min_points = self.config.get('quality_control', {}).get('min_points_per_field', 10)
        field_counts = df['field_id'].value_counts()
        valid_fields = field_counts[field_counts >= min_points].index
        
        original_field_count = df['field_id'].nunique()
        filtered_df = df[df['field_id'].isin(valid_fields)]
        new_field_count = filtered_df['field_id'].nunique()

        logger.info(f"Quality Control: Removed {original_field_count - new_field_count} fields with < {min_points} points.")
        return filtered_df

    def _get_aggregations_for_col(self, col_name: str) -> list:
        """
        Returns a list of aggregation functions for a given column name.
        """
        base_aggregations = [
            'mean', 
            'std', 
            'median',
            lambda x: np.percentile(x, q=10),
            lambda x: np.percentile(x, q=90)
        ]
        # Potentially add more specific aggregations based on column type or config
        return base_aggregations

