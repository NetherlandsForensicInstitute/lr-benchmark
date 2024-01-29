import csv
import logging
import os
import urllib.request
from abc import ABC
from typing import Optional, List, Set, Union, Mapping, Iterator, Iterable, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.pairing import CartesianPairing, BasePairing
from lrbenchmark.typing import PathLike

LOG = logging.getLogger(__name__)


class Dataset(ABC):
    def __init__(self,
                 measurements: Optional[List[Measurement]] = None,
                 holdout_source_ids: Optional[Iterable[Union[int, str]]] = None):
        """
        :param holdout_source_ids: provide the precise sources to include in the holdout data.
        """
        super().__init__()
        self.measurements = measurements
        self.holdout_source_ids = holdout_source_ids

    @property
    def source_ids(self) -> Set[int]:
        return {m.source.id for m in self.measurements}

    def get_x(self) -> np.ndarray:
        return np.array([m.get_x() for m in self.measurements])

    def get_y(self) -> np.ndarray:
        return np.array([m.get_y() for m in self.measurements])

    def get_splits(self,
                   train_size: Optional[Union[float, int]] = None,
                   validate_size: Optional[Union[float, int]] = None,
                   n_splits: Optional[int] = 1,
                   seed: int = None) -> Iterator['Dataset']:
        # TODO: allow specific source splits
        """
        This function splits the measurements in a dataset into two splits, as specified by the
        provided parameters. Every source is in exactly one split.
        :param train_size: size of the train set. Can be a float, to indicate a fraction, or an integer to indicate an
                           absolute number of sources in each
                           split. If not specified, is the complement of the validate_size if provided, else 0.8.
        :param validate_size: size of the validation set. Can be a float, to indicate a fraction, or an integer to
                          indicate an absolute number of sources
                          in each split. If not specified, is the complement of the train_size if provided, else 0.2.

        :param n_splits: number of splits to ...
        :param seed: seed to ensure repeatability of the split

        """
        source_ids = [m.source.id for m in self.measurements]

        s = GroupShuffleSplit(n_splits=n_splits, random_state=seed, train_size=train_size, test_size=validate_size)

        for split in s.split(self.measurements, groups=source_ids):
            yield [Dataset(measurements=list(map(lambda i: self.measurements[i], split_idx))) for split_idx in
                   split]

    def split_off_holdout_set(self) -> Tuple[Optional['Dataset'], 'Dataset']:
        """
        if holdout source ids were provided, returns the dataset with those sources, and the dataset of the complement

        if no hold source ids were provided, returns None and this set itself
        """
        if self.holdout_source_ids:
            holdout_measurements = [measurement for measurement in self.measurements if
                                    measurement.source.id in self.holdout_source_ids]
            other_measurements = [measurement for measurement in self.measurements if
                                  measurement.source.id not in self.holdout_source_ids]
            return Dataset(measurements=holdout_measurements), Dataset(measurements=other_measurements)
        return None, self

    def get_pairs(self,
                  seed: Optional[int] = None,
                  pairing_function: BasePairing = CartesianPairing(),
                  split_trace_reference: Optional[bool] = False) -> List[MeasurementPair]:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        if split_trace_reference and all(m.is_like_reference is None for m in self.measurements):
            raise ValueError("Want to split trace and reference measurements, but attribute 'is_like_reference' is not "
                             "set for any measurement.")
        return pairing_function.transform(self.measurements, seed=seed, split_trace_reference=split_trace_reference)


class XTCDataset(Dataset):
    def __init__(self, measurements_path, **kwargs):
        self.measurements_path = measurements_path
        super().__init__(**kwargs)

        with open(self.measurements_path, "r") as f:
            reader = csv.DictReader(f)
            measurements = [Measurement(source=Source(id=int(row['batchnumber']), extra={}),
                                        extra={'Repeat': int(row['measurement'])},
                                        value=np.array(list(map(float, row.values()))[2:])) for row in reader]
        self.measurements = measurements

    def __repr__(self):
        return "XTC dataset"


class GlassDataset(Dataset):
    def __init__(self, measurements_folder: Optional[PathLike] = None, **kwargs):
        self.measurements_folder = measurements_folder
        super().__init__(**kwargs)

        glass_files = ['duplo.csv', 'training.csv', 'triplo.csv']
        if not self.measurements_folder:
            path = 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/'
            glass_folder = os.path.join('resources', 'glass')
            for file in glass_files:
                download_dataset_file(glass_folder, file, os.path.join(path, file))
            self.measurements_folder = glass_folder

        measurements = []
        max_item = 0
        for file in glass_files:
            path = os.path.join(self.measurements_folder, file)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                measurements_tmp = [Measurement(source=Source(id=int(row['Item']) + max_item, extra={}),
                                                extra={'Piece': int(row['Piece'])},
                                                # the values consist of measurements of ten elemental compositions,
                                                # which start at the fourth position of each row
                                                value=np.array(list(map(float, row.values()))[3:])) for row in reader]
                # The item values start with 1 in each file, this is making it ascending across different files
                max_item = measurements_tmp[-1].source.id
                measurements.extend(measurements_tmp)
        self.measurements = measurements

    def __repr__(self):
        return "Glass dataset"


class ASRDataset(Dataset):
    """
    A dataset containing paired measurements for the purpose of automatic speaker recognition.
    """

    def __init__(self, scores_path, meta_info_path, source_filter, reference_typicalities, trace_typicalities, **kwargs):
        self.scores_path = scores_path
        self.meta_info_path = meta_info_path
        self.source_filter = source_filter or {}
        self.reference_typicalities = reference_typicalities or {}
        self.trace_typicalities = trace_typicalities or {}
        super().__init__(**kwargs)

        with open(self.scores_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        header_measurement_data = np.array(data[0][1:])
        measurement_data = np.array(data)[1:, 1:]

        recording_data = self.load_recording_annotations()

        measurements = []
        for i in tqdm(range(measurement_data.shape[0]), desc='Reading recording measurement data'):
            filename_a = header_measurement_data[i]
            source_id_a, duration = self.get_source_id_duration_from_filename(filename_a)
            info_a = recording_data.get(filename_a.replace('_' + str(duration) + 's', ''))
            if info_a and all([info_a.get(key) == val for key, val in self.source_filter.items()]):
                if all([info_a.get(key) == val for key, val in self.reference_typicalities.items()]):
                    is_like_reference = True
                else:
                    is_like_reference = False

                if all([info_a.get(key) == val for key, val in self.trace_typicalities.items()]):
                    is_like_trace = True
                else:
                    is_like_trace = False

                measurements.append(Measurement(
                                Source(id=source_id_a, extra={'sex': info_a['sex'], 'age': info_a['beller_leeftijd']}),
                                is_like_reference=is_like_reference, is_like_trace=is_like_trace,
                                extra={'filename': filename_a, 'net_duration': float(info_a['net duration']),
                                       'actual_duration': duration, 'auto': info_a['auto']}))
        self.measurements = measurements

    @staticmethod
    def get_source_id_duration_from_filename(filename: str) -> Tuple[str, int]:
        """
        Retrieve the source id and actual duration of the recording from the file name.
        """
        source_id, _, duration = filename.split("_")
        duration = duration.split("s")[0]
        return source_id, int(duration)


    def load_recording_annotations(self) -> Mapping[str, Mapping[str, str]]:
        """
        Read annotations containing information of the recording and speaker.
        """
        with open(self.meta_info_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            data = list(reader)

        return {elt['filename']: elt for elt in data}

    def __repr__(self):
        return "ASR dataset"


def download_dataset_file(folder: str, file: str, url: str):
    location = os.path.join(folder, file)
    if not os.path.isfile(location):
        print(f'downloading {file}')
        try:
            urllib.request.urlretrieve(url, location)
        except Exception as e:
            print(f"Could not download {file} because of: {e}")
