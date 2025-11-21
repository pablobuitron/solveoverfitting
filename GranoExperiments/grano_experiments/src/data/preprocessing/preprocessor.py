import abc
from typing import Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike


class Preprocessor(metaclass=abc.ABCMeta):
    """
    Abstract base class for preprocessing raw data arrays.

    This class defines a common interface for all preprocessors
    that transform input data into a cleaned and optionally resampled representation.
    """
    @abc.abstractmethod
    def preprocess(self,
                   data: np.array,
                   x: ArrayLike = None,
                   upsampled_x: ArrayLike = None) -> np.array:
        """
        Apply preprocessing to an input array.

        This method belongs to a high-level preprocessor interface.
        A preprocessor is designed to transform input arrays into output
        arrays, with transformations depending on the specific type of
        data handled by the concrete implementation (e.g., vegetation
        indices, meteorological series, soil features).

        All preprocessors are expected to support at least an upsampling
        operation along the temporal axis.

        Args:
            data: A 2D numpy array of shape (n_samples, n_timesteps).
                  Each row represents a temporal sequence associated with
                  the data type under preprocessing.
            x: Optional array representing the original time axis of `data`.
               If None, time is assumed as integer indices
               (0, 1, 2... n_timesteps-1).
            upsampled_x: Optional array representing the target time axis
                         for resampling or interpolation. Requires `x` to
                         be provided.

        Returns:
            np.ndarray: The preprocessed array. The shape may differ
            depending on the resampling or upsampling strategy applied.

        Raises:
            ValueError: If `upsampled_x` is provided but `x` is None.
        """
        ...