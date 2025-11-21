import pathlib
from typing import Optional

from .satellite_data import SatelliteDataPreprocessor
from ...utils import basic_file_utils
from . import preprocessor

class PreprocessingRouter:
    """
    Router for selecting the appropriate preprocessor based on feature names.

    This class centralizes the mapping between a feature and the corresponding
    Preprocessor implementation. It allows easy retrieval of the correct
    preprocessing logic for different data types (e.g., satellite-derived
    features, meteorological data).

    Attributes:
        satellite_features: List of feature names that should be handled
            by the SatelliteDataPreprocessor.
    """
    _DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent.parent / 'config' / 'preprocessing_router_config.yaml'


    def __init__(self, configuration: dict = None):
        if configuration is None:
            configuration = basic_file_utils.load_yaml(self._DEFAULT_CONFIG_PATH)
        self.satellite_features: list[str] = configuration['satellite_features']

    def get_preprocessor(self, feature_name: str) -> Optional[preprocessor.Preprocessor]:
        if feature_name in self.satellite_features:
            return SatelliteDataPreprocessor()
        return None
