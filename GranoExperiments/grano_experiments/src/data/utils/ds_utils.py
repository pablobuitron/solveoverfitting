import pathlib

import numpy as np
import pandas as pd
from typing import Optional


def find_point_id_in_dataframe(
        lon: float,
        lat: float,
        df: pd.DataFrame,
        tol: float = 1e-10,
        lon_col: str = "lon",
        lat_col: str = "lat",
) -> Optional[int]:
    """
    Find point_id in a DataFrame given lon and lat.

    Step 1: Try exact match.
    Step 2: If no exact match, fallback to closest point in Euclidean distance.

    Args:
        lon: Longitude of target point.
        lat: Latitude of target point.
        df: DataFrame with columns [lon_col, lat_col, id_col].
        tol: Max Euclidean distance allowed for fallback.
        lon_col: Column name for longitude.
        lat_col: Column name for latitude.

    Returns:
        point_id if found, else None.

    Raises:
        ValueError if multiple exact matches are found.
    """
    # Step 1: exact match
    exact_matches = df[(df[lon_col] == lon) & (df[lat_col] == lat)]
    if len(exact_matches) == 1:
        return exact_matches.iloc[0].name
    elif len(exact_matches) > 1:
        raise ValueError(f"Multiple exact matches found for ({lon}, {lat})")

    # Step 2: fallback to nearest point
    diffs = df[[lon_col, lat_col]].to_numpy() - np.array([lon, lat])
    distances = np.linalg.norm(diffs, axis=1)
    best_idx = np.argmin(distances)
    if distances[best_idx] <= tol:
        # noinspection PyUnresolvedReferences
        return df.iloc[best_idx].name

    return None


def map_coordinate_to_point_id(positions: np.array,
                               dataframe: pd.DataFrame,
                               tol: float = 1e-10,
                               lon_col: str = "lon",
                               lat_col: str = "lat") -> dict:
    #
    if positions.ndim != 2 or positions.shape[1] != 2:
        if positions.ndim == 2 and positions.shape[0] == 2:
            raise ValueError(f"`positions` seems transposed: shape={positions.shape}. Expected (N, 2).")
        else:
            raise ValueError(f"Unexpected shape for positions: {positions.shape}. Expected (N, 2).")

    result = {}
    for pos in  positions[:]:
        lon, lat = pos[0], pos[1]
        id_ = find_point_id_in_dataframe(lon, lat, dataframe, tol, lon_col,lat_col)
        if id_ is None:
            raise RuntimeError(f"No point identifier found for ({lon}, {lat})")
        result[(lon, lat)] = id_

    return result


def get_feature_name(filename: str) -> str:
    stem = pathlib.Path(filename).stem
    return stem.split('|')[3]

# def find_point_index(
#         lon: np.float64,
#         lat: np.float64,
#         positions: h5py.Dataset,
#         tol: float = 1e-10
# ) -> Optional[int]:
#     """
#     Find index of a (lon, lat) point inside positions dataset.
#
#     Step 1: try exact match on both coordinates.
#     Step 2: if no exact match, fallback to closest point in Euclidean distance.
#
#     Parameters
#     ----------
#     lon : float
#         Target longitude.
#     lat : float
#         Target latitude.
#     positions : h5py.Dataset
#         Dataset with shape (N, 2).
#     tol : float
#         Maximum Euclidean distance allowed for fallback.
#
#     Returns
#     -------
#     Optional[int]
#         Index of the matching point or None if not found.
#
#     Raises
#     ------
#     ValueError
#         - If positions has wrong shape or is transposed.
#         - If multiple exact matches are found.
#     """
#     # Ensure shape (N,2)
#     if positions.ndim != 2 or positions.shape[1] != 2:
#         if positions.ndim == 2 and positions.shape[0] == 2:
#             raise ValueError(f"`positions` seems transposed: shape={positions.shape}. Expected (N, 2).")
#         else:
#             raise ValueError(f"Unexpected shape for positions: {positions.shape}. Expected (N, 2).")
#
#     pos = positions[:]  # load numpy (N,2)
#
#     # Step 1: try exact match
#     exact_mask = (pos[:, 0] == lon) & (pos[:, 1] == lat)
#     exact_candidates = np.where(exact_mask)[0]
#     if len(exact_candidates) == 1:
#         return int(exact_candidates[0])
#     elif len(exact_candidates) > 1:
#         raise ValueError(f"Multiple exact matches found for ({lon}, {lat})")
#
#     # Step 2: fallback - closest point in Euclidean distance
#     diffs = pos - np.array([lon, lat])
#     distances = np.linalg.norm(diffs, axis=1)
#     best_idx = np.argmin(distances)
#     if distances[best_idx] <= tol:
#         return int(best_idx)
#
#     return None