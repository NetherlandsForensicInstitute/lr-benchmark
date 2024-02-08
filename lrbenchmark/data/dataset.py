import csv
import logging
import os
import urllib.request
from abc import ABC
from typing import Optional, List, Set, Union, Mapping, Iterator, Iterable, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, LeavePGroupsOut
from tqdm import tqdm

from lrbenchmark.data.models import Measurement, Source, MeasurementPair, Sample
from lrbenchmark.pairing import CartesianPairing, BasePairing
from lrbenchmark.typing import PathLike
from lrbenchmark.utils import complies_with_filter_requirements

LOG = logging.getLogger(__name__)


class Dataset(ABC):
    def __init__(self,
                 measurements: Optional[List[Measurement]] = None,
                 holdout_source_ids: Optional[Iterable[Union[int, str]]] = None,
                 trace_reference_properties: Optional[Mapping[str, Mapping[str, Any]]] = None):
        """
        :param holdout_source_ids: provide the precise sources to include in the holdout data.
        :param trace_reference_properties: properties of trace and reference measurements
        """
        super().__init__()
        self.measurements = measurements
        self.holdout_source_ids = holdout_source_ids
        self.trace_reference_properties = trace_reference_properties or {}

    @property
    def source_ids(self) -> Set[int]:
        return {m.source.id for m in self.measurements}

    def get_x(self) -> np.ndarray:
        return np.array([m.get_x() for m in self.measurements])

    def get_y(self) -> np.ndarray:
        return np.array([m.get_y() for m in self.measurements])

    def get_splits(self,
                   split_type: Optional[str] = 'simple',  # 'simple' or 'leave_one_out'
                   train_size: Optional[Union[float, int]] = None,
                   validate_size: Optional[Union[float, int]] = None,
                   seed: int = None) -> Iterator['Dataset']:
        # TODO: allow specific source splits
        """
        This function splits the measurements in a dataset into two splits, as specified by the
        provided parameters. Every source is in exactly one split. Either the size parameters or leave_two_out
        scheme should be used
        :param split_type: split type can be a single 'simple' split, or the 'leave_one_out' method
                    In the latter case, for computational efficiency, this is implemented by making all pairs with
                    1 or 2 sources as validation, and selecting same source or different source pairs in
                    the get_pairs() function.
        :param train_size: size of the train set. Can be a float, to indicate a fraction, or an integer to indicate an
                           absolute number of sources in each
                           split. If not specified, is the complement of the validate_size if provided, else 0.8.
        :param validate_size: size of the validation set. Can be a float, to indicate a fraction, or an integer to
                          indicate an absolute number of sources
                          in each split. If not specified, is the complement of the train_size if provided, else 0.2.
        :param seed: seed to ensure repeatability of the split

        """
        leave_one_out = split_type == 'leave_one_out'
        if (train_size or validate_size) and leave_one_out:
            raise ValueError('Either the size parameters or leave_two_out scheme should be used')

        source_ids = [m.source.id for m in self.measurements]

        if leave_one_out:
            # a group is a source. This class provides all combinations of one or two sources as
            # validation splits. get_pairs should handle creating same-source (1-group) or different-source (2-groups)
            # pairs.
            for n_groups in [1, 2]:
                lpgo = LeavePGroupsOut(n_groups=n_groups)
                for i, (train_index, test_index) in enumerate(lpgo.split(self.measurements, groups=source_ids)):
                    # if one source, we need at least two measurements
                    # if two sources, they both need at least one measurement
                    # if fewer than this there is no validation, and for
                    # computational reasons we do not return the split
                    if (n_groups == 1 and len(test_index) > 1) or \
                            (n_groups == 2 and len(set(map(lambda i: source_ids[i], test_index))) == 2):
                        yield [Dataset(measurements=list(map(lambda i: self.measurements[i], train_index)),
                                       trace_reference_properties=self.trace_reference_properties),
                               Dataset(measurements=list(map(lambda i: self.measurements[i], test_index)),
                                       trace_reference_properties=self.trace_reference_properties)]
        else:
            # set n_splits to 1 as we already have repeats in the outer experimental loop
            s = GroupShuffleSplit(n_splits=1, random_state=seed, train_size=train_size, test_size=validate_size)

            for split in s.split(self.measurements, groups=source_ids):
                yield [Dataset(measurements=list(map(lambda i: self.measurements[i], split_idx)),
                               trace_reference_properties=self.trace_reference_properties) for split_idx in split]

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
            return \
                Dataset(measurements=holdout_measurements,
                        trace_reference_properties=self.trace_reference_properties), \
                Dataset(measurements=other_measurements,
                        trace_reference_properties=self.trace_reference_properties)
        return None, self

    def get_pairs(self,
                  seed: Optional[int] = None,
                  pairing_function: BasePairing = CartesianPairing(),
                  trace_reference_properties: Mapping[str, Mapping[str, Any]] = None) -> List[MeasurementPair]:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        if not trace_reference_properties:
            # allow this variable to be overwritten for this function
            trace_reference_properties = self.trace_reference_properties

        return pairing_function.transform(self.measurements, seed=seed,
                                          trace_reference_properties=trace_reference_properties)


class XTCDataset(Dataset):
    def __init__(self, measurements_path, **kwargs):
        self.measurements_path = measurements_path
        super().__init__(**kwargs)

        with open(self.measurements_path, "r") as f:
            reader = csv.DictReader(f)
            measurements = [Measurement(source=Source(id=int(row['batchnumber']), extra={}), id=1,
                                        sample=Sample(int(row['measurement'])), extra={},
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
        for i, file in enumerate(glass_files):
            path = os.path.join(self.measurements_folder, file)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                measurements_tmp = [Measurement(source=Source(id=int(row['Item']), extra={}),
                                                sample=Sample(int(row['Piece'])), extra={}, id=i,
                                                # the values consist of measurements of ten elemental compositions,
                                                # which start at the fourth position of each row
                                                value=np.array(list(map(float, row.values()))[3:])) for row in reader]
                measurements.extend(measurements_tmp)
        self.measurements = measurements

    def __repr__(self):
        return "Glass dataset"


class ASRDataset(Dataset):
    """
    A dataset containing measurements for the purpose of automatic speaker recognition.
    """

    def __init__(self,
                 scores_path: PathLike,
                 meta_info_path: PathLike,
                 source_filter: Optional[Mapping[str, Any]] = None,
                 limit_n_measurements: Optional[int] = None,
                 properties: Optional[List[str]] = None,
                 **kwargs):
        self.scores_path = scores_path
        self.meta_info_path = meta_info_path
        self.source_filter = source_filter or {}
        self.limit_n_measurements = limit_n_measurements
        self.properties = properties
        super().__init__(**kwargs)

        self.measurements = self.get_measurements_from_file()

    def get_measurements_from_file(self):
        with open(self.scores_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        header_measurement_data = np.array(data[0][1:])
        measurement_data = np.array(data)[1:, 1:]
        recording_data = self.load_recording_annotations()

        measurements = []
        for i in tqdm(range(measurement_data.shape[0]), desc='Reading recording measurement data'):
            if self.limit_n_measurements and len(measurements) >= self.limit_n_measurements:
                return measurements
            filename_a = header_measurement_data[i]
            source_id_a, recording_id_a, duration = self.get_ids_and_duration_from_filename(filename_a)
            filename_in_recording_data = filename_a.replace('_' + str(duration) + 's', '') if duration else filename_a
            info_a = recording_data.get(filename_in_recording_data)
            if not duration and info_a:
                duration = info_a['net_duration']
            extra = {'filename': filename_a, 'actual_duration': duration}
            if info_a and complies_with_filter_requirements(self.source_filter, info_a, {'actual_duration': duration}):
                measurements.append(Measurement(
                    Source(id=source_id_a, extra={key: info_a.get(key) for key in self.source_filter.keys()}),
                    sample=Sample(recording_id_a),
                    id=duration, extra={**info_a, **extra}))  # TODO: only relevant measurement properties?
            elif source_id_a.lower() in ['case', 'zaken', 'zaak']:
                measurements.append(Measurement(Source(id='Case', extra={}), sample=Sample(recording_id_a),
                                                id=duration, extra=extra))
        return measurements

    @staticmethod
    def get_ids_and_duration_from_filename(filename: str) -> Tuple[str, str, Optional[int]]:
        """
        Retrieve the source id, recording id and actual duration (if possible) of the recording from the filename.
        """
        split = filename.split("_")
        if len(split) == 3:
            source_id, recording_id, duration = split
            duration = int(duration.split("s")[0])
        elif len(split) == 2:
            source_id, recording_id = split
            duration = None
        else:
            raise ValueError(f"Invalid filename found: {filename}")
        return source_id, recording_id, duration

    def load_recording_annotations(self) -> Dict[str, Dict[str, str]]:
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
