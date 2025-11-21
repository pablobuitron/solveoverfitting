from typing import Optional
import numpy as np
from scipy.optimize import curve_fit

from . import preprocessor


def nullify_outliers(array):
    is_growing = True
    latest_valid_value = - np.inf if np.isnan(array[0]) else array[0]
    maximum_idx = np.argmax(array)
    for idx, val in enumerate(array):

        if idx == 0:
            continue
        if idx == len(array) - 1:
            continue

        if idx == maximum_idx:
            is_growing = False
            latest_valid_value = val
            continue

        previous_val = array[idx - 1]
        if np.isnan(previous_val) or previous_val < latest_valid_value:
            previous_val = latest_valid_value

        if is_growing and array[idx] < previous_val:
            array[idx] = np.nan
        elif not is_growing and array[idx] > previous_val:
            array[idx] = np.nan
        else:
            latest_valid_value = val

    return array


def get_next_valid(current_idx, array):
    if current_idx == len(array) - 1:
        return 0

    for i in range(current_idx + 1, len(array)):
        if not np.isnan(array[i]):
            return array[i]

    raise RuntimeError('No valid value found')


def replace_nan_values(array, method: str = 'interp'):
    array = array.copy()
    if method == 'mean':
        previous_valid_value = 0
        for idx, val in enumerate(array):

            if np.isnan(val):
                next_valid_value = get_next_valid(idx, array)
                array[idx] = (previous_valid_value + next_valid_value) / 2
                previous_valid_value = array[idx]
            else:
                previous_valid_value = val
    elif method == 'interp':
        x = np.arange(len(array))
        nans = np.isnan(array)
        array[nans] = np.interp(x[nans], x[~nans], array[~nans])
    else:
        raise RuntimeError('Unknown method {}'.format(method))
    return array


def nullify_outliers_bidirectional(array):
    # remove inf
    array[np.isinf(array)] = np.nan

    maximum_idx = np.argmax(array)

    # --- First phase: pre-maximum (growing) ---
    pre = array[:maximum_idx + 1].copy()
    pre_clean = nullify_outliers(pre)

    # --- Second phase: post-maximum (decaying) ---
    post = array[maximum_idx:].copy()

    # Invert the order so we can apply the same algorithm as "growing"
    post_flipped = post[::-1]
    post_clean_flipped = nullify_outliers(post_flipped)

    # Flip back to original order
    post_clean = post_clean_flipped[::-1]

    # --- Recombine ---
    cleaned = np.concatenate([pre_clean[:-1], post_clean])

    return cleaned


def beta_rescaled(t, a, alpha, beta, t0, t1):
    # Normalize t into [0,1]
    x = (t - t0) / (t1 - t0)
    x = np.clip(x, 1e-6, 1 - 1e-6)  # avoid div by zero
    return a * (x ** (alpha - 1)) * ((1 - x) ** (beta - 1))


def double_logistic(t, c, a, k1, t_s, k2, t_e):
    """
    Double logistic function for NDVI seasonal dynamics.

    f(t) = c + a * [1 / (1 + exp(-k1 * (t - t_s)))
                    - 1 / (1 + exp(-k2 * (t - t_e)))]

    Args:
        t   : numpy array of time steps
        c   : baseline NDVI (minimum level)
        a   : amplitude (NDVI_max - NDVI_min)
        k1  : growth rate of green-up
        t_s : start time of green-up
        k2  : decline rate of senescence
        t_e : end time of senescence

    Returns:
        numpy array with same shape as t
    """
    # Avoid overflow in exp
    z1 = np.clip(-k1 * (t - t_s), -500, 500)
    z2 = np.clip(-k2 * (t - t_e), -500, 500)
    return c + a * (1 / (1 + np.exp(z1)) - 1 / (1 + np.exp(z2)))


def generalized_gaussian(t, a, mu, sigma, p):
    """
    Generalized Gaussian function.

    f(t) = a * exp( - | (t - mu) / sigma | ** p )

    Args:
        t : numpy array of time steps
        a : amplitude (peak height)
        mu : position of the peak
        sigma : width (scale)
        p : shape parameter (>0);
            p=2 is normal Gaussian,
            p<2 is broader peak,
            p>2 is sharper peak

    Returns:
        numpy array with same shape as t
    """
    t = np.array(t, dtype=float)
    return a * np.exp(-np.abs((t - mu) / sigma) ** p)


class SatelliteDataPreprocessor(preprocessor.Preprocessor):
    """
    Preprocessor for satellite time series data.

    This class specializes the generic Preprocessor interface for
    satellite-derived temporal signals (e.g., vegetation indices like NDVI).

    The preprocessing pipeline includes:
      - Outlier removal.
      - Missing value imputation.
      - Optional curve fitting (e.g., double logistic, generalized Gaussian).
      - Optional upsampling on the temporal axis, either through interpolation
        or by evaluating the fitted curve on a denser time grid.
    """
    CURVE_FIT_FALLBACK_RETURN_AS_IS = 'return_as_is'  # return the array without any processing
    CURVE_FIT_FALLBACK_NULLIFY = 'nullify'  # return an array of np.nan

    @staticmethod
    def _interp_row(row: np.ndarray, upsampled_x: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Interpolate one row of data to upsampled_x."""
        res = np.interp(upsampled_x, x, row)
        return res

    def __init__(self,
                 curve_to_fit=None,
                 curve_fit_maxfev=10 * 1000 * 1000,
                 curve_fit_fallback='return_as_is'
                 ):
        """
        Preprocesses arrays of satellite data. It removes outliers (see nullify_outliers),
        replaces values using linear interpolation and then tries to adjust data using curve_to_fit.
        :param curve_to_fit: the curve used for fitting data. If None, no curve wll be fitted.
                             To upsample data, just linear interpolation will be used.
                             Some curve that can be used for fittin are double_logistic and generalized gaussian,
                             which are inside this module
        :param curve_fit_maxfev: the maximum number of iterations for curve fit
        :param curve_fit_fallback: what to do when the curve fits fails.
                                   Look at this class constants to see the implemented fallback strategies.
        """
        self.curve_to_fit = curve_to_fit
        self.curve_fit_maxfev = curve_fit_maxfev
        self.curve_fit_fallback = curve_fit_fallback

    def preprocess(self,
                   data: np.array,
                   x: np.typing.ArrayLike = None,
                   upsampled_x: np.typing.ArrayLike = None) -> np.array:

        x = x if x is None else np.array(x)
        upsampled_x = upsampled_x if upsampled_x is None else np.array(upsampled_x)
        if upsampled_x is not None and x is None:
            raise ValueError('upsampled_x cannot be used if x is not provided')

        y = np.apply_along_axis(nullify_outliers_bidirectional, 1, data)
        y = np.apply_along_axis(replace_nan_values, 1, y)

        if self.curve_to_fit:
            y = np.apply_along_axis(self._fit_curve, 1, y, x, upsampled_x)
        elif upsampled_x is not None:
            # no curve has been fitted but data still need to be upsampled
            y = np.apply_along_axis(self._interp_row, 1, y, upsampled_x, x)
        return y

    def _fit_curve(self,
                   y: np.array,
                   x: Optional[np.ndarray] = None,
                   upsampled_x: Optional[np.ndarray] = None) -> np.array:

        x = np.arange(len(y)) if x is None else x
        try:

            # p0 is a list that contains the starting parameters for the curve that will be fitted
            p0 = self._calculate_p0(x, y)

            # suppressing inspection due to wrong warning
            # noinspection PyTupleAssignmentBalance
            params, _ = curve_fit(self.curve_to_fit, x, y, p0=p0, maxfev=self.curve_fit_maxfev)
            result = self.curve_to_fit(x, *params) if upsampled_x is None else self.curve_to_fit(upsampled_x, *params)

        except RuntimeError:

            if self.curve_fit_fallback == self.CURVE_FIT_FALLBACK_RETURN_AS_IS:
                result = y.copy() if upsampled_x is None else np.interp(upsampled_x, x, y)
            elif self.curve_fit_fallback == self.CURVE_FIT_FALLBACK_NULLIFY:
                result = np.full(y.shape, np.nan) if upsampled_x is None else np.full(upsampled_x.shape, np.nan)
            else:
                raise RuntimeError(f'Unknown curve fit fallback {self.curve_fit_fallback}')

        return result

    def _calculate_p0(self, x: np.array, y: np.array) -> list:
        """The algorithm that fits the curve needs a starting point for the arguments that it's going to fit"""
        if self.curve_to_fit == double_logistic:
            return [
                np.min(y),  # c ~ baseline
                np.max(y) - np.min(y),  # a ~ amplitude
                0.1,  # k1
                np.median(x) - 30,  # t_s
                0.1,  # k2
                np.median(x) + 30  # t_e
            ]
        elif self.curve_to_fit == generalized_gaussian:
            return [
                np.max(y),
                np.argmax(y),
                len(x) / 5, 2
            ]
        else:
            raise RuntimeError(f'Invalid curve to fit {generalized_gaussian.__name__}')
