import numpy as np
import pandas as pd

from GranoExperiments.grano_experiments.src.data import timeseries_dataset


# Columns that should never be treated as features
META_COLS = {
    timeseries_dataset.COLUMN_GROUP_ID,
    timeseries_dataset.COLUMN_TIME_IDX,
    timeseries_dataset.COLUMN_TARGET,
    timeseries_dataset.COLUMN_TMP_GROUP_ID_STR,
}


def _add_normalized_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalized time index per group_id: t_norm in [0, 1].
    This approximates phenological progression (early / mid / late season).
    """
    df = df.copy()
    # Ensure sorted by time
    df = df.sort_values([timeseries_dataset.COLUMN_GROUP_ID,
                         timeseries_dataset.COLUMN_TIME_IDX])

    df["fe_t_rank"] = df.groupby(timeseries_dataset.COLUMN_GROUP_ID).cumcount()
    max_rank = df.groupby(timeseries_dataset.COLUMN_GROUP_ID)["fe_t_rank"].transform("max")
    max_rank = max_rank.replace(0, 1)  # avoid division by zero
    df["fe_t_norm"] = df["fe_t_rank"] / max_rank

    return df


def _add_window_aggregates(df: pd.DataFrame,
                           feature_cols: list[str],
                           windows: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """
    For each dynamic feature in feature_cols, compute per-group aggregates
    within normalized-time windows: mean, max, min, std.
    Aggregates are merged back to each row via group_id.
    """
    if "fe_t_norm" not in df.columns:
        raise RuntimeError("fe_t_norm not found in dataframe. Call _add_normalized_time first.")

    gid = timeseries_dataset.COLUMN_GROUP_ID
    dfs_agg = []

    for w_name, (lo, hi) in windows.items():
        mask = (df["fe_t_norm"] >= lo) & (df["fe_t_norm"] < hi)
        sub = df.loc[mask, [gid] + feature_cols].copy()
        if sub.empty:
            continue

        g = sub.groupby(gid)[feature_cols].agg(["mean", "max", "min", "std"])

        # Flatten columns: feature_stat_window
        g.columns = [
            f"fe_{feat}_{stat}_{w_name}"
            for feat, stat in g.columns.to_flat_index()
        ]
        dfs_agg.append(g)

    # Global aggregates over full season
    full = df.groupby(gid)[feature_cols].agg(["mean", "max", "min", "std"])
    full.columns = [
        f"fe_{feat}_{stat}_full"
        for feat, stat in full.columns.to_flat_index()
    ]
    dfs_agg.append(full)

    # Merge everything
    agg_all = dfs_agg[0]
    for extra in dfs_agg[1:]:
        agg_all = agg_all.join(extra, how="outer")

    df = df.merge(agg_all, on=gid, how="left")
    return df


def _add_growth_features(df: pd.DataFrame,
                         feature_name: str) -> pd.DataFrame:
    """
    Add shape/growth features for a monotonic-ish vegetation index (e.g., NDVI).
    Features per group_id:
      - max, min, range
      - t_norm at max
      - simple slope from first to last timestep
    """
    if feature_name not in df.columns:
        return df

    df = df.copy()
    gid = timeseries_dataset.COLUMN_GROUP_ID

    def group_growth(g):
        vals = g[feature_name].values
        if len(vals) == 0 or np.all(np.isnan(vals)):
            return pd.Series({
                f"fe_{feature_name}_max": np.nan,
                f"fe_{feature_name}_min": np.nan,
                f"fe_{feature_name}_range": np.nan,
                f"fe_{feature_name}_t_norm_at_max": np.nan,
                f"fe_{feature_name}_slope_full": np.nan,
            })

        idx_max = np.nanargmax(vals)
        t_norm_vals = g["fe_t_norm"].values if "fe_t_norm" in g.columns else np.linspace(0, 1, len(vals))
        t_norm_at_max = t_norm_vals[idx_max]

        first = vals[0]
        last = vals[-1]
        slope = (last - first) / max(len(vals) - 1, 1)

        return pd.Series({
            f"fe_{feature_name}_max": np.nanmax(vals),
            f"fe_{feature_name}_min": np.nanmin(vals),
            f"fe_{feature_name}_range": np.nanmax(vals) - np.nanmin(vals),
            f"fe_{feature_name}_t_norm_at_max": t_norm_at_max,
            f"fe_{feature_name}_slope_full": slope,
        })

    growth_df = df.groupby(gid, as_index=True).apply(group_growth)
    df = df.merge(growth_df, on=gid, how="left")
    return df


def _add_lagged_sums(df: pd.DataFrame,
                     col: str,
                     windows: list[int]) -> pd.DataFrame:
    """
    Add rolling sum features over time for a given column (per group_id).
    Example: precipitation over last 7 or 14 days.
    """
    if col not in df.columns:
        return df

    df = df.sort_values([timeseries_dataset.COLUMN_GROUP_ID,
                         timeseries_dataset.COLUMN_TIME_IDX])

    gid = timeseries_dataset.COLUMN_GROUP_ID

    def apply_roll(x: pd.Series, w: int):
        return x.rolling(window=w, min_periods=1).sum()

    for w in windows:
        df[f"fe_{col}_sum{w}"] = (
            df.groupby(gid)[col]
            .transform(lambda x: apply_roll(x, w))
        )

    return df


def _add_stress_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build stress-like ratios, e.g. evaporation / radiation, etc.
    All columns are prefixed with fe_.
    """
    df = df.copy()

    # Example 1: evapotranspiration / solar radiation
    evap_cols = [
        "observed_evaporation_from_vegetation_transpiration_mean",
        "observed_evaporation_from_bare_soil_mean",
    ]
    rad_col = "observed_surface_net_solar_radiation_mean"

    if rad_col in df.columns:
        for e_col in evap_cols:
            if e_col in df.columns:
                df[f"fe_{e_col}_over_rad"] = df[e_col] / (df[rad_col] + 1e-6)

    # Example 2: soil moisture vs AWC if static_awc exists
    if "static_awc" in df.columns:
        for sm_col in [
            "observed_volumetric_soil_water_layer_1_mean",
            "observed_volumetric_soil_water_layer_2_mean",
            "observed_volumetric_soil_water_layer_3_mean",
        ]:
            if sm_col in df.columns:
                df[f"fe_{sm_col}_over_awc"] = df[sm_col] / (df["static_awc"] + 1e-6)

    return df


def _select_base_feature_cols(df: pd.DataFrame, ts_cfg: dict) -> list[str]:
    """
    Reproduce the same feature selection as the original TimeSeriesDataSet config:
      - static_categoricals
      - static_reals
      - time_varying_known_reals
      - time_varying_unknown_reals
    """
    feature_cols: list[str] = []
    for key in [
        "static_categoricals",
        "static_reals",
        "time_varying_known_reals",
        "time_varying_unknown_reals",
    ]:
        cols = ts_cfg.get(key, [])
        for c in cols:
            if c in df.columns and c not in feature_cols:
                feature_cols.append(c)
    return feature_cols


def build_rf_features(df: pd.DataFrame, ts_cfg: dict):
    """
    Main entry point:
      - starts from the long dataframe used by the LSTM (one row per group_id, time_idx)
      - adds engineered features:
         * normalized time within season (fe_t_norm)
         * window aggregates (early/mid/late/full) for key dynamic features
         * growth/shape metrics for NDVI-like index
         * lagged sums for meteorological variables
         * simple stress ratios
      - returns X, y and list of feature columns used.
    """
    df = df.copy()

    # 1) Base feature cols from config (same as LSTM)
    base_feature_cols = _select_base_feature_cols(df, ts_cfg)

    # 2) Add normalized time index
    df = _add_normalized_time(df)

    # 3) Choose dynamic features we want to engineer more
    dynamic_candidates = [
        "observed_ndvi",
        "observed_ndre",
        "observed_gndvi",
        "observed_volumetric_soil_water_layer_1_mean",
        "observed_volumetric_soil_water_layer_2_mean",
        "observed_volumetric_soil_water_layer_3_mean",
        "observed_total_precipitation_mean",
        "observed_evaporation_from_vegetation_transpiration_mean",
        "observed_evaporation_from_bare_soil_mean",
        "observed_surface_net_solar_radiation_mean",
        "observed_skin_temperature_mean",
    ]
    dyn_cols = [c for c in dynamic_candidates if c in df.columns]

    # 4) Window aggregates: early / mid / late / full
    windows = {
        "early": (0.0, 0.2),
        "mid": (0.2, 0.6),
        "late": (0.6, 1.01),
    }
    if dyn_cols:
        df = _add_window_aggregates(df, dyn_cols, windows)

    # 5) Growth features for NDVI / NDRE / GNDVI (if present)
    for veg_col in ["observed_ndvi", "observed_ndre", "observed_gndvi"]:
        df = _add_growth_features(df, veg_col)

    # 6) Lagged sums (e.g. 7 and 14 days) for precip and evapotranspiration
    df = _add_lagged_sums(df, "observed_total_precipitation_mean", windows=[7, 14])
    df = _add_lagged_sums(df, "observed_evaporation_from_vegetation_transpiration_mean", windows=[7, 14])

    # 7) Stress-like ratios
    df = _add_stress_ratios(df)

    # 8) Collect final feature columns:
    #    base features + all engineered ones starting with "fe_"
    engineered_cols = [
        c for c in df.columns
        if c.startswith("fe_") and c not in META_COLS
    ]

    feature_cols = base_feature_cols + engineered_cols

    # 9) Build X, y
    y = df[timeseries_dataset.COLUMN_TARGET].values
    X = df[feature_cols].copy()

    return X, y, feature_cols
