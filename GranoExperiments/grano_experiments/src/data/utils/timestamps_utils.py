import calendar
import datetime
from typing import Dict, Optional

import numpy as np


def parse_timestamp(timestamp):
    timestamp = timestamp.decode()
    if is_valid_date_format(timestamp, '%Y-%m-%d %H:%M:%S'):
        timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
    elif is_valid_date_format(timestamp, '%Y-%m-%d %H:%M:%S%z'):
        timestamp = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S%z')
        timestamp = timestamp.replace(tzinfo=None)
    else:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    return timestamp


def is_valid_date_format(date_string: str, date_format: str) -> bool:
    try:
        datetime.datetime.strptime(date_string, date_format)
        return True
    except ValueError:
        return False


def get_season_start_date(timestamp: datetime.datetime, start_month: int = 10) -> datetime.datetime:
    start_year = timestamp.year if start_month <= timestamp.month <= 12 else timestamp.year - 1
    start_date = datetime.datetime(year=start_year, month=start_month, day=1)
    return start_date


def get_season_end_date(timestamp: datetime.datetime, end_month: int = 7) -> datetime.datetime:
    end_year = timestamp.year if 1 <= timestamp.month <= end_month else timestamp.year + 1
    end_date = datetime.datetime(year=end_year, month=end_month, day=calendar.monthrange(end_year, end_month)[1])
    return end_date


def normalize_timestamp(timestamp: datetime.datetime, start_month: int = 10, end_month: int = 7):
    start_date = get_season_start_date(timestamp, start_month)
    end_date = get_season_end_date(timestamp, end_month)
    max_date_range = (end_date - start_date).days

    timestamp_date_range = (timestamp.date() - start_date).days

    normalized_timestamp = timestamp_date_range / max_date_range

    return normalized_timestamp


def calculate_timestamp_id_mapping(start_month: int, end_month: int, start_year: int) -> Dict[datetime.datetime, int]:
    start_date = datetime.datetime(year=start_year, month=start_month, day=1)

    end_day = calendar.monthrange(start_year + 1, end_month)[1]
    end_date = datetime.datetime(year=start_year + 1, month=end_month, day=end_day)

    mapping = {}
    current_date = start_date
    while current_date <= end_date:
        mapping[current_date] = timestamp_to_id(current_date, start_month, end_month)
        current_date += datetime.timedelta(days=1)

    return mapping


def timestamp_to_id(timestamp: datetime.datetime, start_month: int = 10, end_month: int = 7, collide_leap_days: bool = False) -> int:
    """Timestamp will be converted into an ID, calculated as the number of days past the start of the season
    (first day of start_month)."""

    if not isinstance(timestamp, datetime.datetime):
        timestamp = parse_timestamp(timestamp)

    if collide_leap_days and timestamp.month == 2 and timestamp.day == 29:
    # to keep indices consistent, 29 of Feb is indexed as 28. Later, data is corrected to manage this collision.
        timestamp = timestamp.replace(day=28)

    season_start_date = get_season_start_date(timestamp, start_month)
    season_end_date = get_season_end_date(timestamp, end_month)
    season_28_feb = season_end_date.replace(month=2, day=28)

    indexed_timestamp = (timestamp.replace(tzinfo=None) - season_start_date).days
    is_leap_year = season_end_date.year % 4 == 0
    if is_leap_year and collide_leap_days and (season_28_feb < timestamp < season_end_date):
        indexed_timestamp = indexed_timestamp - 1

    return indexed_timestamp


def find_day(timestamps: list[datetime.datetime], month: int, day: int) -> Optional[int]:
    """
    Finds the index of the first occurrence of a specific month/day in a list of timestamps.

    Args:
        timestamps: List of datetime objects.
        month: Month to search (1-12).
        day: Day to search (1-31).

    Returns:
        Index of the matching timestamp, or None if not found.
    """
    for idx, ts in enumerate(timestamps):
        if ts.month == month and ts.day == day:
            return idx
    return None


def fix_leap_day_data(
        parsed_timestamps: list[datetime.datetime],
        timestamps_ids: list[int],
        data: np.array,
        is_missing: np.array
) -> tuple:
    """
    Fixes leap day (Feb 29) entries in timeseries data.

    Args:
        parsed_timestamps: List of datetime objects.
        timestamps_ids: List of timestamps turned into integer indices.
        data: Numpy array of values corresponding to timestamps.
        is_missing: Numpy boolean array for missing data.

    Returns:
        A tuple (timestamps, timestamps_ids, data, is_missing) with leap day handled.
    """
    feb29_position = find_day(parsed_timestamps, 2, 29)
    if feb29_position is None:
        return parsed_timestamps, timestamps_ids, data, is_missing

    feb28_position = find_day(parsed_timestamps, 2, 28)
    if feb28_position is None:
        # No Feb 28: keep Feb 29 as-is (will be mapped to Feb 28 index later)
        return parsed_timestamps, timestamps_ids, data, is_missing

    # Ensure we work on NumPy arrays, not HDF5 datasets
    data = np.asarray(data)
    is_missing = np.asarray(is_missing)

    if data.ndim == 2:
        # Merge Feb 28 and Feb 29
        data[feb28_position, :] = np.nanmean(
            [data[feb28_position, :], data[feb29_position, :]], axis=0
        )
        is_missing[feb28_position, :] = (
            is_missing[feb28_position, :] & is_missing[feb29_position, :]
        )

        # Remove Feb 29 from arrays
        del parsed_timestamps[feb29_position]
        del timestamps_ids[feb29_position]
        data = np.delete(data, feb29_position, axis=0)
        is_missing = np.delete(is_missing, feb29_position, axis=0)

    elif data.ndim == 3:
        # Merge Feb 28 and Feb 29
        data[:, feb28_position, :] = np.nanmean(
            [data[:, feb28_position, :], data[:, feb29_position, :]], axis=0
        )
        is_missing[:, feb28_position, :] = (
            is_missing[:, feb28_position, :] & is_missing[:, feb29_position, :]
        )

        # Remove Feb 29 from arrays
        del parsed_timestamps[feb29_position]
        del timestamps_ids[feb29_position]
        data = np.delete(data, feb29_position, axis=1)  # axis=1 is timestamp axis in 3D
        is_missing = np.delete(is_missing, feb29_position, axis=1)

    else:
        raise ValueError(
            f"Unsupported data dimensionality {data.ndim}. Expected 2D or 3D array."
        )

    return parsed_timestamps, timestamps_ids, data, is_missing

