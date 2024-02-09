import csv
import logging
import os
import urllib.request
from abc import ABC
from typing import Optional, List, Set, Union, Mapping, Iterator, Iterable, Tuple, Dict, Any

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, LeavePGroupsOut
from tqdm import tqdm

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.pairing import CartesianPairing, BasePairing
from lrbenchmark.typing import PathLike
from lrbenchmark.utils import complies_with_filter_requirements

LOG = logging.getLogger(__name__)


class Dataset(ABC):
    def __init__(self,
                 measurements: Optional[List[Measurement]] = None,
                 holdout_source_ids: Optional[Iterable[Union[int, str]]] = None,
                 filter_on_trace_reference_properties: bool = False):
        """
        :param holdout_source_ids: provide the precise sources to include in the holdout data.
        :param filter_on_trace_reference_properties: whether or not measurements consisting of two types should be mixed
                in pairing
        """
        super().__init__()
        self.measurements = measurements
        self.holdout_source_ids = holdout_source_ids
        self.filter_on_trace_reference_properties = filter_on_trace_reference_properties

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
                                       filter_on_trace_reference_properties=self.filter_on_trace_reference_properties),
                               Dataset(measurements=list(map(lambda i: self.measurements[i], test_index)),
                                       filter_on_trace_reference_properties=self.filter_on_trace_reference_properties)]
        else:
            # set n_splits to 1 as we already have repeats in the outer experimental loop
            s = GroupShuffleSplit(n_splits=1, random_state=seed, train_size=train_size, test_size=validate_size)

            for split in s.split(self.measurements, groups=source_ids):
                yield [Dataset(measurements=list(map(lambda i: self.measurements[i], split_idx)),
                               filter_on_trace_reference_properties=self.filter_on_trace_reference_properties) for
                       split_idx in
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
            return \
                Dataset(measurements=holdout_measurements,
                        filter_on_trace_reference_properties=self.filter_on_trace_reference_properties), \
                Dataset(measurements=other_measurements,
                        filter_on_trace_reference_properties=self.filter_on_trace_reference_properties)
        return None, self

    def get_pairs(self,
                  seed: Optional[int] = None,
                  pairing_function: BasePairing = CartesianPairing(),
                  filter_on_trace_reference_properties: Optional[bool] = None) -> List[MeasurementPair]:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        if filter_on_trace_reference_properties is None:
            # allow this variable to be overwritten for this function
            filter_on_trace_reference_properties = self.filter_on_trace_reference_properties

        return pairing_function.transform(self.measurements, seed=seed,
                                          filter_on_trace_reference_properties=filter_on_trace_reference_properties)


class XTCDataset(Dataset):
    def __init__(self, measurements_path, **kwargs):
        self.measurements_path = measurements_path
        super().__init__(**kwargs)

        with open(self.measurements_path, "r") as f:
            reader = csv.DictReader(f)
            measurements = [Measurement(source=Source(id=int(row['batchnumber']), extra={}), id=int(row['measurement']),
                                        extra={},
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
                                                extra={'Piece': int(row['Piece'])}, id=int(row['id']),
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
    A dataset containing measurements for the purpose of automatic speaker recognition.
    """

    def __init__(self,
                 scores_path: PathLike,
                 meta_info_path: PathLike,
                 source_filter: Optional[Mapping[str, Any]] = None,
                 reference_properties: Optional[Mapping[str, Any]] = None,
                 trace_properties: Optional[Mapping[str, Any]] = None,
                 limit_n_measurements: Optional[int] = None,
                 **kwargs):
        self.scores_path = scores_path
        self.meta_info_path = meta_info_path
        self.source_filter = source_filter or {}
        self.reference_properties = reference_properties or {}
        self.trace_properties = trace_properties or {}
        self.limit_n_measurements = limit_n_measurements
        super().__init__(**kwargs)

        self.measurement_header = None
        self.measurement_data = None
        self.measurements = self.get_measurements_from_file()

    def get_measurements_from_file(self):
        with open(self.scores_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        self.measurement_header = np.array(data[0][1:])
        self.measurement_data = np.array(data)[1:, 1:].astype(float)
        recording_data = self.load_recording_annotations()

        measurements = []
        for i in tqdm(range(self.measurement_data.shape[0]), desc='Reading recording measurement data'):
            if self.limit_n_measurements and len(measurements) >= self.limit_n_measurements:
                return measurements
            filename_a = self.measurement_header[i]
            source_id_a, recording_id_a, duration = self.get_ids_and_duration_from_filename(filename_a)
            filename_in_recording_data = filename_a.replace('_' + str(duration) + 's', '') if duration else filename_a
            info_a = recording_data.get(filename_in_recording_data)
            if not duration and info_a:
                duration = info_a['net_duration']
            is_like_reference = complies_with_filter_requirements(self.reference_properties, info_a or {},
                                                                  {'duration': duration})
            is_like_trace = complies_with_filter_requirements(self.trace_properties, info_a or {},
                                                              {'duration': duration})
            extra = {'filename': filename_a, 'actual_duration': duration}
            if info_a and complies_with_filter_requirements(self.source_filter, info_a, {'duration': duration}):
                measurements.append(Measurement(
                    Source(id=source_id_a, extra=self.get_extra_information(info_a, 'source')),
                    id=recording_id_a,
                    is_like_reference=is_like_reference, is_like_trace=is_like_trace,
                    extra={**extra, **self.get_extra_information(info_a, 'measurement')}))
            elif source_id_a.lower() in ['case', 'zaken', 'zaak']:
                measurements.append(Measurement(Source(id='Case', extra={}), id=recording_id_a,
                                                is_like_reference=True, is_like_trace=True,
                                                extra=extra))
        return measurements

    def get_extra_information(self, info: Dict[str, str], object_type: str) -> Dict[str, str]:
        """
        Retrieve the values in `info` for the either keys in the trace and reference properties (if
        'object_type' is `measurement`) or for the keys in the source filter (if 'object_type' is 'source').
        """
        if object_type == 'measurement':
            all_property_keys = {**self.trace_properties, **self.reference_properties}.keys()
        elif object_type == 'source':
            all_property_keys = self.source_filter.keys()
        else:
            raise ValueError(f"Unknown object {object_type} found. Possible object types are measurement or source.")
        return {key: info.get(key) for key in all_property_keys}

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
