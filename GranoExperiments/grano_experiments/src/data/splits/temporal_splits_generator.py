import pathlib
import networkx
import pandas as pd

from . import dataset_split
from .splits_generator import SplitsGenerator
from .. import dataset_index
from .generators_factory import register_splits_generator
from ..utils.file_cache import GeoDataFrameFilesCache

@register_splits_generator
class TemporalSplitsGenerator(SplitsGenerator):

    DEFAULT_IOU_THRESHOLD = 0.25
    DEFAULT_KEEP_UNCLUSTERED_FIELDS = False
    DEFAULT_TRAIN_SPLIT_RATIO = 0.7
    DEFAULT_VAL_SPLIT_RATIO = 0.15

    @staticmethod
    def _fields_to_rows_integrity_check(fields_to_rows):
        for field, rows in fields_to_rows.items():
            yield_files = dataset_index.get_yield_files(rows, to_list=False)
            if len(yield_files) and len(yield_files.drop_duplicates()) != 1:
                raise RuntimeError('A wheat field should be associated to a single yield file,'
                                   ' but this is not the case. Check the dataset index.')

    @staticmethod
    def iou(box1, box2):
        """
        Compute Intersection over Union between two bounding boxes.
        Each box is [xmin, ymin, xmax, ymax].
        """

        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        inter_w = max(0, inter_xmax - inter_xmin)
        inter_h = max(0, inter_ymax - inter_ymin)
        intersection = inter_w * inter_h

        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)

        union = area1 + area2 - intersection
        if union == 0:
            return 0.0

        return intersection / union

    @classmethod
    def from_dict(cls, input_dict) -> 'SplitsGenerator':
        dataset_root = input_dict.get('dataset_root')
        if not dataset_root:
            raise ValueError('The dataset root must be provided.')

        keep_unclustered_fields = input_dict.get('keep_unclustered_fields')
        if keep_unclustered_fields is None:
            keep_unclustered_fields = cls.DEFAULT_KEEP_UNCLUSTERED_FIELDS

        iou_threshold = input_dict.get('iou_threshold')
        if iou_threshold is None:
            iou_threshold =  cls.DEFAULT_IOU_THRESHOLD

        train_split_ratio = input_dict.get('train_split_ratio')
        if train_split_ratio is None:
            train_split_ratio = cls.DEFAULT_TRAIN_SPLIT_RATIO

        val_split_ratio = input_dict.get('val_split_ratio')
        if val_split_ratio is None:
            val_split_ratio = cls.DEFAULT_VAL_SPLIT_RATIO

        instance = cls(dataset_root,
                       iou_threshold=iou_threshold,
                       keep_unclustered_fields=keep_unclustered_fields,
                       train_split_ratio=train_split_ratio,
                       val_spilt_ratio=val_split_ratio)

        return instance

    def __init__(self,
                 dataset_root: pathlib.Path,
                 iou_threshold: float = DEFAULT_IOU_THRESHOLD,
                 keep_unclustered_fields: bool = DEFAULT_KEEP_UNCLUSTERED_FIELDS,
                 train_split_ratio: float = DEFAULT_TRAIN_SPLIT_RATIO,
                 val_spilt_ratio: float = DEFAULT_VAL_SPLIT_RATIO):
        # TODO <doc>
        # instance vars set beforehand
        self.dataset_root = dataset_root
        self.iou_threshold = iou_threshold
        self.keep_unclustered_fields = keep_unclustered_fields
        self.train_split_ratio = train_split_ratio
        self.val_spilt_ratio = val_spilt_ratio
        # instance vars that are set at runtime
        self.dataset_index = None

    def generate_splits_files(self, dataset_index_path: str | pathlib.Path, output_dir: str | pathlib.Path):
        self.dataset_index = dataset_index.DatasetIndex(dataset_index_path)
        super().generate_splits_files(dataset_index_path, output_dir)
        self.dataset_index = None

    def generate_splits(self, ids: list[str]) -> dict[str, dataset_split.DatasetSplit]:
        if self.dataset_index is None:
            raise RuntimeError('This method requires the dataset index to be initialized.')
        return super().generate_splits(ids)


    def _create_splits(self, ids: list[str]) -> tuple[list, list, list]:
        if self.dataset_index is None:
            raise RuntimeError('Dataset index not initialized.')

        fields_to_rows = {field: self.dataset_index.get_rows(field) for field in self.dataset_index.get_wheat_fields()}
        self._fields_to_rows_integrity_check(fields_to_rows)

        fields, bboxes = self._fields_and_bboxes(fields_to_rows)

        clusters = self._cluster_bboxes(bboxes)

        train_split, val_split, test_split = self._clusters_to_splits(fields, clusters, fields_to_rows)

        if self.keep_unclustered_fields:
            for c in clusters:
                if len(c) != 1:
                    continue
                unclustered_field = list(c)[0]
                unclustered_ids = list(fields_to_rows[unclustered_field].index)
                train_split += unclustered_ids

        train_split = [sample for sample in train_split if sample in ids]
        val_split = [sample for sample in val_split if sample in ids]
        test_split = [sample for sample in test_split if sample in ids]

        return train_split, val_split, test_split

    def _fields_and_bboxes(self, fields_to_rows: dict[str, pd.DataFrame]) -> tuple[list, list]:
        fields = []
        bboxes = []
        with GeoDataFrameFilesCache(self.dataset_root) as cache:
            for field, rows in fields_to_rows.items():
                row = rows.iloc[0]
                yield_file = dataset_index.get_yield_file(row)
                gdf = cache.get_file(yield_file)
                fields.append(field)
                bboxes.append(gdf.total_bounds)
        return fields, bboxes

    def _cluster_bboxes(self, bboxes: list) -> list[set]:
        """
        Build a graph where each bbox is a node and edges connect overlapping boxes.
        Connected components correspond to clusters of related fields.
        """

        n = len(bboxes)
        graph = networkx.Graph()
        graph.add_nodes_from(range(n))

        for i in range(n):
            for j in range(i + 1, n):
                if self.iou(bboxes[i], bboxes[j]) >= self.iou_threshold:
                    graph.add_edge(i, j)

        components = list(networkx.connected_components(graph))
        return components

    def _clusters_to_splits(self, fields: list, clusters: list, fields_to_rows: dict) -> tuple[list, list, list]:
        #
        train_split = []
        val_split = []
        test_split = []

        for cluster in clusters:
            if len(cluster) < 2:
                continue

            # to make sure that training set has the most samples
            indices_by_samples = sorted(list(cluster), key= lambda idx: len(fields_to_rows[fields[idx]]), reverse=True)

            if len(indices_by_samples) == 2:
                training_field = fields[indices_by_samples[0]]
                val_field = fields[indices_by_samples[1]]
                train_split.extend(list(fields_to_rows[training_field].index))
                val_split.extend(list(fields_to_rows[val_field].index))
                continue

            split_indices = self._split_train_val_test(indices_by_samples,
                                                       train_split_ratio=self.train_split_ratio,
                                                       val_split_ratio=self.val_spilt_ratio)

            training_fields = [fields[si] for si in split_indices[0]]
            for tf in training_fields:
                train_split += (list(fields_to_rows[tf].index))

            val_fields = [fields[si] for si in split_indices[1]]
            for vf in val_fields:
                val_split += (list(fields_to_rows[vf].index))

            test_fields = [fields[si] for si in split_indices[2]]
            for tf in test_fields:
                test_split += (list(fields_to_rows[tf].index))

        return train_split, val_split, test_split

