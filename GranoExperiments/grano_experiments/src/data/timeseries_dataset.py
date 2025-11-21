import logging
import pathlib
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import pytorch_forecasting
from pytorch_forecasting import TimeSeriesDataSet

from .utils import timestamps_utils, ds_utils, file_cache
from . import dataset_index, dataset_constants
from .splits import dataset_split, splits_generator
from ..utils import basic_file_utils
from . import preprocessing

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

COLUMN_GROUP_ID = 'group_id'
COLUMN_TMP_GROUP_ID_STR = 'tmp_group_id_str'
COLUMN_TIME_IDX = 'time_idx'
COLUMN_TARGET = 'target'
STATIC_FEATURES_PREFIX = 'static_'
DYNAMIC_FEATURES_PREFIX = 'observed_'


class DatasetBuilder:
    """
    Builder for a PyTorch Forecasting TimeSeriesDataSet tailored to the Grano.IT pipeline.

    The class translates the project's dataset index and files (static and dynamic
    features stored in HDF5-like caches) into a single DataFrame formatted to the
    expectations of `pytorch_forecasting.TimeSeriesDataSet`. It performs:
      - per-field iteration and filtering,
      - creation of (group_id, time_idx) cartesian product rows,
      - mapping of yields (targets) and features,
      - final cleaning and dataset instantiation.

    Normalization Handling:
    One split (by default "train") is used to estimate normalization parameters
    for features and target. Other splits reuse these
    parameters via `TimeSeriesDataSet.from_dataset`, ensuring consistent scaling
    across training, validation and test sets.

    If `split_for_normalization` is set to None, target normalization is disabled
    and each dataset is built independently, letting PyTorch Forecasting apply its
    own per-split feature scaling.
    """

    _DEFAULT_CONFIG_PATH = pathlib.Path(__file__).parent.parent / 'config' / 'timeseries_dataset_config.yaml'

    @staticmethod
    def _init_field_dataframe(ds_index_field_rows: pd.DataFrame, timestamps_to_ids: dict) -> pd.DataFrame:
        """
        Initialize a DataFrame that represents the cartesian product between field points and time indices.

        This creates a DataFrame that contains one row for each combination of
        a point (group) belonging to the given wheat field and every time index
        (as defined by `timestamps_to_ids`).

        Args:
            ds_index_field_rows: DataFrame with rows corresponding to points (contains longitude and latitude).
                The original DataFrame index is used as the point identifier and will be reset into
                a column named `COLUMN_GROUP_ID`.
            timestamps_to_ids: Mapping (e.g. datetime -> integer id) that provides the time id
                for each timestamp. Time ids are integers that correspond to elapsed days since the season start, which has id=0.

        Returns:
            A DataFrame with columns [group_id, time_idx] plus the original lon/lat columns.
            Each (group_id, time_idx) pair appears exactly once.
        """
        points_df = ds_index_field_rows[[dataset_index.COLUMN_LON, dataset_index.COLUMN_LAT]]
        points_df = points_df.reset_index(names=COLUMN_GROUP_ID)
        timestamps_df = pd.DataFrame(sorted(timestamps_to_ids.values()), columns=[COLUMN_TIME_IDX])
        points_df['tmp_merging_key'] = 1
        timestamps_df['tmp_merging_key'] = 1
        field_dataframe = pd.merge(points_df, timestamps_df, on=['tmp_merging_key'])
        field_dataframe = field_dataframe.drop(['tmp_merging_key'], axis=1)
        return field_dataframe

    @staticmethod
    def _build_dynamic_data_dataframe(points_ids: list,
                                      timestamps_ids: list,
                                      features: list,
                                      data: np.array) -> pd.DataFrame:
        """
        Construct a long-format DataFrame from a 3D dynamic data array (n_points, n_times, n_features).

        The function reshapes the provided `data` into a table with columns:
        ['group_id', 'time_idx', <feature_1>, <feature_2>, ...].

        Args:
            points_ids: List of point identifiers corresponding to the first dimension of `data`.
            timestamps_ids: List of integer time indices corresponding to the second dimension of `data`.
                These indices map to the `time_idx` column used downstream.
            features: List of column names for the last dimension of `data` (length f).
            data: 3D numpy array with shape (n_points, n_timestamps_in_input, n_features).

        Returns:
            DataFrame in long format: each row corresponds to (group_id, time_idx) with feature columns.
        """
        n, t, f = len(points_ids), len(timestamps_ids), len(features)
        reshaped_data = np.reshape(data, (n * t, f))

        p_ids_repeated = np.repeat(points_ids, t)
        timestamps_ids_tiled = np.tile(timestamps_ids, n)

        result = pd.DataFrame(reshaped_data, columns=features)
        result.insert(0, 'time_idx', timestamps_ids_tiled)
        result.insert(0, 'group_id', p_ids_repeated)

        return result

    def __init__(self, root_dir: pathlib.Path,
                 ds_index: dataset_index.DatasetIndex,
                 ds_splits: List[dataset_split.DatasetSplit] = None,
                 ds_splits_generator: splits_generator.SplitsGenerator = None,
                 configuration: dict = None,
                 quick_debug: bool = False):
        """
        Create a DatasetBuilder.

        Loads configuration (from provided dict or default YAML) and initializes caches and preprocessing router.

        Args:
            root_dir: Root directory of the dataset used by caches and for resolving files.
            ds_index: DatasetIndex instance that provides metadata about wheat fields, points and associated files.
            ds_splits: DatasetSplit objects that associates points to splits such ad training, validation and test sets.
            ds_splits_generator: a SplitsGenerator object, that can be used to generate DatasetSplit instances.
            configuration: Optional dictionary with configuration values. If None, loads from _DEFAULT_CONFIG_PATH.
            quick_debug: allows to create a dataset with very few samples just to quickly test code.

        Raises:
            ValueError: If both `ds_splits` and `ds_splits_generator` are provided.
        """
        if configuration is None:
            configuration = basic_file_utils.load_yaml(self._DEFAULT_CONFIG_PATH)

        if ds_splits and ds_splits_generator:
            raise ValueError("ds_splits and ds_splits_generator cannot be used simultaneously")

        # instance attributes
        self.root_dir = root_dir
        self.ds_index = ds_index
        self.ds_splits = ds_splits
        self.ds_splits_generator = ds_splits_generator
        self.quick_debug = quick_debug

        # attributes from config
        self.wheat_fields_to_exclude: List[str] = configuration['wheat_fields_to_exclude']
        self.features_to_exclude: List[str] = configuration['features_to_exclude']
        # self.field_level_aggregation: bool = configuration['field_level_aggregation'] commented because it's going to be replaced by better logic that acts upon dataset creation

        self.season_start_month = configuration['season_start_month']  # will be used to calculate ids
        self.season_end_month = configuration['season_end_month']
        self.static_data_dtype = np.float64

        self.static_categoricals = configuration['static_categoricals']
        self.static_reals = configuration['static_reals']
        self.time_varying_unknown_reals = configuration['time_varying_unknown_reals']
        self.allow_missing_timesteps = configuration['allow_missing_timesteps']
        self.add_relative_time_idx = configuration['add_relative_time_idx']

        # dependencies
        self._static_data_cache = file_cache.Hdf5FilesCache(root_dir)
        self._dynamic_data_cache = file_cache.Hdf5FilesCache(root_dir)
        self._preprocessing_router = preprocessing.router.PreprocessingRouter()

    def build(self) -> dict[str, TimeSeriesDataSet]:
        """
        Build a TimeSeriesDataSet ready for training/evaluation.

        The method performs:
          - building a consolidated DataFrame with static and dynamic features (following pytorch_forecasting specifics),
          - final preprocessing: casting, replacing infinities and dropping invalid rows,
          - creation of a pytorch_forecasting.TimeSeriesDataSet using class configuration.

        Returns:
            Configured TimeSeriesDataSet instance.
        """
        try:
            data = self._build_dataframe()
        except Exception as e:
            raise e
        finally:
            # ensure caches are closed after building
            self._static_data_cache.close_files()
            self._dynamic_data_cache.close_files()

        data = self._preprocess_final_dataframe(data, backup_group_id_col=True)
        data = self._divide_in_splits(data, use_group_id_backup_col=True)
        result = self._instantiate_timeseries_datasets(data)

        return result

    def _build_dataframe(self) -> pd.DataFrame:
        """Iterate through wheat fields and build the full dataset DataFrame.
        Returns:
            DataFrame containing all fields with columns:
            ['group_id', 'time_idx', 'target', <static_* columns>, <observed_* columns>]
        """
        final_dataframe = None

        # iterate over wheat fields known by the dataset index
        wheat_fields = self.ds_index.get_wheat_fields()
        for i, wheat_field in enumerate(wheat_fields):

            if self.quick_debug and i > 2:
                break

            # optionally skip fields flagged in configuration
            if wheat_field in self.wheat_fields_to_exclude:
                logger.info(f'Skipping wheat field {wheat_field}')
                continue

            logger.info(f"Building {wheat_field}")

            # retrieve rows in the index corresponding to this wheat field (points and metadata)
            field_rows = self.ds_index.get_rows(wheat_field)

            # compute mapping datetime -> integer time_idx for the season spanning the configured months
            field_year = dataset_index.get_year(field_rows.iloc[0])
            t_id_mapping = timestamps_utils.calculate_timestamp_id_mapping(self.season_start_month,
                                                                           self.season_end_month,
                                                                           field_year - 1)

            # init a dataframe containing data for this wheat field, that will be merged with the final dataframe
            field_dataframe = self._init_field_dataframe(field_rows, t_id_mapping)

            # map point id -> yield value
            yield_map = dataset_index.get_point_to_yields(field_rows)
            field_dataframe[COLUMN_TARGET] = field_dataframe[COLUMN_GROUP_ID].map(yield_map)

            # noinspection PyTypeChecker
            static_files, dynamic_files = dataset_index.get_associated_files(field_rows.iloc[0])

            # loading and merging static features
            static_data = self._load_static_data(static_files, field_rows)
            for column, data_mapping in static_data.items():
                field_dataframe[column] = field_dataframe[COLUMN_GROUP_ID].map(data_mapping)

            # loading and merging dynamic features
            dynamic_data = self._load_dynamic_data(dynamic_files, field_rows, t_id_mapping)
            field_dataframe = pd.merge(field_dataframe, dynamic_data, how='left', on=[COLUMN_GROUP_ID, COLUMN_TIME_IDX])

            # commented because it's going to be replaced by better logic that acts upon dataset creation
            # if self.field_level_aggregation:
            #     field_dataframe = self._aggregate_field_dataframe(field_dataframe)

            # append to final dataframe (concat across fields)
            if final_dataframe is None:
                final_dataframe = field_dataframe
            else:
                final_dataframe = pd.concat([final_dataframe, field_dataframe])

        return final_dataframe

    def _preprocess_final_dataframes(self, dataframes: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        result = {}
        for split_name, df in dataframes:
            result[split_name] = self._preprocess_final_dataframe(df)
        return result

    def _preprocess_final_dataframe(self, df: pd.DataFrame, backup_group_id_col: bool = False) -> pd.DataFrame:

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.reset_index(drop=True)

        # static categorical operations
        cols_to_remove = [col for col in self.static_categoricals if col not in df.columns]
        [self._remove_static_feature(col)for col in cols_to_remove]

        df = df.dropna(subset=self.static_categoricals)
        df = self._cast_static_categoricals(df)

        # time varying unknown reals
        cols_to_remove = [col for col in self.time_varying_unknown_reals if col not in df.columns]
        self._remove_dynamic_features(cols_to_remove)
        df = df.dropna(subset=self.time_varying_unknown_reals)

        # Ensure numeric columns are float32 for MPS compatibility, but keep indices as int
        # TODO |> rendere configurabili i fix di compatibilità con metal (?)
        df[COLUMN_TIME_IDX] = df[COLUMN_TIME_IDX].astype(int)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for c in numeric_cols:
            if c not in [COLUMN_TIME_IDX, COLUMN_GROUP_ID]:
                df[c] = df[c].astype(np.float32)

        # Map string group_ids to integer ids to avoid leakage when using metal
        if backup_group_id_col:
            df[COLUMN_TMP_GROUP_ID_STR] = df[COLUMN_GROUP_ID]
        unique_group_ids = df[COLUMN_GROUP_ID].unique()
        group_id_mapping = {gid: i for i, gid in enumerate(unique_group_ids)}
        df[COLUMN_GROUP_ID] = df[COLUMN_GROUP_ID].map(group_id_mapping).astype(int)
        return df



    def _divide_in_splits(self, data: pd.DataFrame, use_group_id_backup_col: bool = True) -> Dict[str, pd.DataFrame]:
        result = {}
        col_to_use = COLUMN_TMP_GROUP_ID_STR if use_group_id_backup_col else COLUMN_GROUP_ID
        splits = self.ds_splits or self.ds_splits_generator.generate_splits(list(data[col_to_use]))
        for split in splits.values():

            split_data = data[data[col_to_use].isin(split.ids)]
            if use_group_id_backup_col:
                # col_to_use is the backup one, it can be dropped now
                split_data = split_data.drop(columns=[col_to_use])

            result[split.split_name] = split_data

        return result

    def _instantiate_timeseries_datasets(
            self,
            splits_to_data: dict,
            split_for_normalization: Optional[str] = 'train',
    ) -> Dict[str, TimeSeriesDataSet]:
        """
        Instantiate PyTorch Forecasting TimeSeriesDataSet objects for all provided splits,
        handling normalization in a consistent and configurable way.

        This method acts as the orchestration layer for dataset creation:
          - If `split_for_normalization` is a split name (e.g., "train"), that split is used
            to compute all normalization and scaling parameters (such as mean/std for features
            and group-based normalization for the target). All other splits are then built
            from those.
          - If `split_for_normalization` is None, every split is built independently with
            `target_normalizer=None`. In this mode, no target normalization is applied, and
            PyTorch Forecasting will handle feature scaling internally for each dataset
            based on its own data distribution.

        Args:
            splits_to_data: Mapping {split_name -> DataFrame} containing all splits.
            split_for_normalization: Split name to use as normalization reference (default "train").
                If None, disables target normalization entirely.

        Returns:
            Dict[str, TimeSeriesDataSet]: A dictionary mapping split names to dataset instances.

        Raises:
            ValueError: If `split_for_normalization` is specified but not found in splits_to_data.
        """
        if split_for_normalization is None:
            return self._build_datasets_without_normalization(splits_to_data)

        if split_for_normalization not in splits_to_data:
            raise ValueError(
                f"`split_for_normalization='{split_for_normalization}'` not found in splits: "
                f"{list(splits_to_data.keys())}"
            )

        ref_name = split_for_normalization
        reference_ds = self._build_reference_dataset(splits_to_data[ref_name], ref_name)

        return self._build_other_datasets_from_reference(
            splits_to_data=splits_to_data,
            reference_ds=reference_ds,
            reference_name=ref_name,
        )

    def _cast_static_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        for static_categorical in self.static_categoricals:
            df[static_categorical] = df[static_categorical].astype(int).astype(str)
        return df

    def _load_static_data(self, static_files: List[str], field_rows: pd.DataFrame):
        """Load static features for a given field from cached files.

        Args:
            static_files: list of files relative paths (keys) to retrieve from the static cache.
            field_rows: DataFrame rows for the current field, used to map coordinates -> group ids.

        Returns:
            Dictionary where keys are column names (prefixed with STATIC_FEATURES_PREFIX) and values
            are dicts mapping point_id -> feature value.
        """
        coordinate_to_p_id = None
        new_columns = {}
        for filename, file in self._static_data_cache.iter_files(static_files):

            feature_name = ds_utils.get_feature_name(filename)
            column_name = STATIC_FEATURES_PREFIX + feature_name

            if feature_name in self.features_to_exclude:
                self._remove_static_feature(column_name)
                continue

            # read positions and data arrays from cached file object (HDF5-like)
            positions = file[dataset_constants.KEY_POSITIONS]
            data = file[dataset_constants.KEY_DATA]

            # build mapping from coordinate -> point id only once per file group:
            # if the dataset is consistent the mapping should be the same for every file
            if coordinate_to_p_id is None:
                coordinate_to_p_id = ds_utils.map_coordinate_to_point_id(positions[:], field_rows)

            # map positions to point ids (missing positions get NaN)
            p_ids = [coordinate_to_p_id.get((lon, lat), np.nan) for lon, lat in positions]

            # convert data to python list to be safe for creating dict mapping
            data = data[:].tolist()
            data_mapping = dict(zip(p_ids, data))

            new_columns[column_name] = data_mapping

        return new_columns

    def _remove_static_feature(self, column_name: str):
        if column_name in self.static_categoricals:
            self.static_categoricals.remove(column_name)
        elif column_name in self.static_reals:
            self.static_reals.remove(column_name)
        else:
            raise RuntimeError(
                f'Column {column_name} cannot be found in any of the resulting dataframe columns')

    def _load_dynamic_data(self, dynamic_files: List[str], field_rows: pd.DataFrame, timestamps_id_mapping: dict):
        """Load, preprocess and upsample dynamic (time-varying) features from cached files.

        Args:
            dynamic_files: list of files relative paths (keys) to retrieve from the dynamic cache.
            field_rows: DataFrame with points for the current field.
            timestamps_id_mapping: mapping datetime -> integer time_idx for the full season.

        Returns:
            DataFrame with columns ['group_id', 'time_idx', <observed_feature_columns...>]
            containing the outer-merged dynamic features across all provided dynamic files.
        """
        coordinate_to_p_id = None
        result_dataframe = pd.DataFrame(columns=['group_id', 'time_idx'])

        for filename, file in self._dynamic_data_cache.iter_files(dynamic_files):

            # macro feature e.g. "total_precipitation"
            macro_feature = ds_utils.get_feature_name(filename)

            positions = file[dataset_constants.KEY_POSITIONS]
            timestamps = file[dataset_constants.KEY_TIMESTAMPS]
            subfeatures = file[dataset_constants.KEY_FEATURES_NAMES]
            data = file[dataset_constants.KEY_DATA]
            missing_data_mask = file[dataset_constants.KEY_IS_MISSING]

            # build features columns names: observed_<macro> or observed_<macro>_<subfeature>
            features = [DYNAMIC_FEATURES_PREFIX + macro_feature + '_' + sf.decode()
                        for sf in subfeatures] if len(subfeatures) > 1 else [DYNAMIC_FEATURES_PREFIX + macro_feature]

            # self.features_to_exclude can contain the name of the macro feature...
            if macro_feature in self.features_to_exclude:
                self._remove_dynamic_features(features)
                continue

            # ...but also the name of specific features (e.g. 'observed_total_precipitation_mean')
            # so here it's calculated both the names of features to remove and their indices along the f axis of the data array
            features_to_remove, subfeatures_to_remove = self._find_features_to_be_removed(features)
            if features_to_remove:
                self._remove_dynamic_features(features_to_remove)
                features = [f for f in features if f not in features_to_remove]
                # subfeatures = np.delete(subfeatures, subfeatures_to_remove) useless since subfeatures is not used, but pay attention
                data = np.delete(data, subfeatures_to_remove, axis=2)  # supposing shape n,t,f

            # map coordinates -> point ids (only once per file group:
            # if the dataset is consistent the mapping should be the same for every file)
            if coordinate_to_p_id is None:
                coordinate_to_p_id = ds_utils.map_coordinate_to_point_id(positions[:], field_rows)

            # list of point ids for this file's positions
            p_ids = [coordinate_to_p_id.get((lon, lat), np.nan) for lon, lat in positions]

            # parse timestamps (bytes/string -> datetime) and map to integer ids using the provided mapping
            parsed_timestamps = [timestamps_utils.parse_timestamp(t) for t in timestamps]
            timestamps_ids = [timestamps_id_mapping[t] for t in parsed_timestamps]

            # memo: for now the code doesn't adjust timestamps_id_mapping, it should.
            # parsed_timestamps, timestamps_ids, data, _ = timestamps_utils.fix_leap_day_data(parsed_timestamps,
            #                                                                                 timestamps_ids,
            #                                                                                 data,
            #                                                                                 missing_data_mask)


            timestamps_ids, data = self._preprocess_and_upsample_arrays(macro_feature,
                                                                        parsed_timestamps,
                                                                        timestamps_ids,
                                                                        data,
                                                                        timestamps_id_mapping)

            dataframe_to_merge = self._build_dynamic_data_dataframe(p_ids, timestamps_ids, features, data)
            result_dataframe = pd.merge(result_dataframe, dataframe_to_merge, on=['group_id', 'time_idx'], how='outer')

        return result_dataframe

    # def _aggregate_field_dataframe(self, field_dataframe: pd.DataFrame):
    #     # to aggregate field points data, for each timestamp compute the mean of numerical features across all points
    #     numerical_columns = self.static_reals + self.time_varying_unknown_reals + [COLUMN_TARGET, 'lon',
    #                                                                                'lat']  # |> gestire nomi lon-lat
    #     aggregated_data = field_dataframe.groupby(COLUMN_TIME_IDX, as_index=False)[numerical_columns].mean()
    #
    #     for static_categorical_column in self.static_categoricals:
    #         # |> si potrebbe pensare a scompattare i dati se il dato categorico è differente
    #         mode = field_dataframe[static_categorical_column].mode().iloc[0]
    #         aggregated_data[static_categorical_column] = mode
    #
    #     base_group = field_dataframe[COLUMN_GROUP_ID].iloc[0]
    #     year, field, _ = base_group.split(dataset_index.ID_SPLITTER)
    #     #  |> problema di design: per come è pensato adesso,
    #     #         gli id degli split e gli id generati da questo metodo devono combaciare
    #     #            la soluzione potrebbe essere la creazione degli id aggregati in un modulo a parte
    #     aggregated_group_id = dataset_index.ID_SPLITTER.join([year, field, dataset_constants.AGGREGATED_KEYWORD])
    #     aggregated_data[COLUMN_GROUP_ID] = aggregated_group_id
    #
    #     return aggregated_data

    def _find_features_to_be_removed(self, features: List[str]) -> Tuple[List[str], List[int]]:
        """:returns: indices for subfeatures that must be removed and features names to be removed"""
        features_to_be_rm = []
        subfeatures_to_rm = []
        for i, feature in enumerate(features):
            if feature in self.features_to_exclude:
                subfeatures_to_rm.append(i)
                features_to_be_rm.append(feature)
        return features_to_be_rm, subfeatures_to_rm

    def _remove_dynamic_features(self, columns_names: List[str]):
        # memo: column_names correspond to the features names
        for column_name in columns_names:
            if column_name in self.time_varying_unknown_reals:
                self.time_varying_unknown_reals.remove(column_name)

    def _preprocess_and_upsample_arrays(self,
                                        macro_feature: str,
                                        parsed_timestamps: list,
                                        timestamps_ids: list,
                                        data: np.ndarray,
                                        timestamps_id_mapping: dict):
        """Preprocess and upsample a dynamic data array.

        If a specific preprocessor exists for the macro feature,
        it delegates the preprocessing + upsampling to that preprocessor (which can implement
        outlier removal, interpolation, smoothing, custom upsampling, etc.). If no preprocessor
        is registered, it will upsample by inserting NaNs for missing timestamps.

        Args:
            macro_feature: name of the macro feature (e.g. 'ndvi', used to obtain a preprocessor).
            parsed_timestamps: list of parsed datetime objects corresponding to the input `data` timesteps.
            timestamps_ids: list of integer ids corresponding to the parsed_timestamps (input time axis).
            data: numpy array with shape (n_points, n_input_timesteps, n_subfeatures).
            timestamps_id_mapping: mapping datetime -> integer id for the full season axis.

        Returns:
            A tuple (upsampled_timestamps_ids, upsampled_data) where:
              - upsampled_timestamps_ids: sorted list of integer time indices for the full season;
              - upsampled_data: numpy array upsampled to shape (n_points, n_full_timesteps, n_subfeatures).
        """
        upsampled_timestamps_ids = sorted(timestamps_id_mapping.values())
        preprocessor = self._preprocessing_router.get_preprocessor(macro_feature)
        if preprocessor:
            data = preprocessor.preprocess(data, timestamps_ids, upsampled_timestamps_ids)
        else:
            data = self._upsample_with_nan(data, parsed_timestamps, timestamps_id_mapping)

        return upsampled_timestamps_ids, data

    @staticmethod
    def _upsample_with_nan(data: np.ndarray, parsed_timestamps: list, timestamps_id_mapping: dict) -> np.ndarray:
        n, _, f = data.shape
        upsampled_data = np.full((n, len(timestamps_id_mapping), f), np.nan)
        for t_idx, t in enumerate(parsed_timestamps):
            upsampled_data[:, timestamps_id_mapping[t], :] = data[:, t_idx, :]
        return upsampled_data

    def _base_timeseries_kwargs(self, split_name: str) -> dict:
        """Common kwargs for TimeSeriesDataSet to avoid duplication."""
        return dict(
            time_idx=COLUMN_TIME_IDX,
            target=COLUMN_TARGET,
            group_ids=[COLUMN_GROUP_ID],
            static_categoricals=self.static_categoricals,
            static_reals=self.static_reals,
            time_varying_unknown_reals=self.time_varying_unknown_reals,
            allow_missing_timesteps=self.allow_missing_timesteps,
            add_relative_time_idx=self.add_relative_time_idx,
            predict_mode=self._decide_predict_mode(split_name),
            max_prediction_length=1,
        )

    def _build_reference_dataset(self, df, split_name: str) -> TimeSeriesDataSet:
        """Build the dataset used to estimate normalization params (usually 'train')."""
        kwargs = self._base_timeseries_kwargs(split_name)
        # Memo: normalization may give problems on MPS if the output is in float64 instead of float32
        reference_ds = TimeSeriesDataSet(
            data=df,
            target_normalizer=pytorch_forecasting.data.TorchNormalizer(method='robust'),
            **kwargs,
        )
        return reference_ds

    def _build_other_datasets_from_reference(
            self,
            splits_to_data: dict,
            reference_ds: TimeSeriesDataSet,
            reference_name: str,
    ) -> Dict[str, TimeSeriesDataSet]:
        """Reuse the reference dataset's normalization/scalers for all other splits."""
        result: Dict[str, TimeSeriesDataSet] = {reference_name: reference_ds}
        for split_name, df in splits_to_data.items():
            if split_name == reference_name:
                continue
            result[split_name] = TimeSeriesDataSet.from_dataset(
                reference_ds,
                df,
                predict=self._decide_predict_mode(split_name),
            )
        return result

    def _build_datasets_without_normalization(self, splits_to_data: dict) -> Dict[str, TimeSeriesDataSet]:
        """Create independent datasets, explicitly disabling target normalization."""
        result: Dict[str, TimeSeriesDataSet] = {}
        for split_name, df in splits_to_data.items():
            kwargs = self._base_timeseries_kwargs(split_name)
            ds = TimeSeriesDataSet(
                data=df,
                target_normalizer=None,  # explicit: do not normalize target
                **kwargs,
            )
            result[split_name] = ds
        return result

    # noinspection PyMethodMayBeStatic
    def _decide_predict_mode(self, split: str) -> bool:
        if split == 'train':
            return False
        else:
            return True
