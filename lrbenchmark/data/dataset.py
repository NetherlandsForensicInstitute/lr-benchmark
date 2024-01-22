import csv
import os
import urllib.request
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Iterable, Optional, Callable, List, Set, Union, Mapping

import numpy as np
from lir.transformers import InstancePairing, AbsDiffTransformer
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from tqdm import tqdm

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.typing import XYType


class Dataset(ABC):
    @abstractmethod
    def get_splits(self,
                   train_size: Optional[Union[float, int]] = 0.8,
                   validate_size: Optional[Union[float, int]] = 0.2,
                   n_splits: Optional[int] = 1,
                   seed: int = None) -> Iterable['Dataset']:
        """
        Retrieve data from this dataset.

        This function is responsible for splitting the data in subset for
        training and testing in a way that is appropriate for the data set.
        Depending on the implementation, the data set may be returned at once,
        as a K-fold series, or otherwise.

        Parameters
        ----------
        seed : int, optional
            Random seed to be used for splitting. The default is None.
        train_size: int, float, optional
            Fraction or number of data points to use for the training set. The default is 0.8, and if not specified,
            the complement of the test size will be used
        validate_size: int, float, optional
            Fraction or number of data points to use for the validation set. The default is 0.2, and if not specified,
            the complement of the train size will be used
        n_splits: int, optional
            Number of splits to ...

        Returns
        -------
        Iterable['Dataset']
            one or more Datasets as an iterable, each element being a subset
            of the original Dataset consisting of measurements and/or measurement
            pairs.

        """
        raise NotImplementedError


class MeasurementsDataset(Dataset):
    def __init__(self, measurements: Optional[List[Measurement]] = None):
        super().__init__()
        self.measurements = measurements

    @property
    def source_ids(self) -> Set[int]:
        return {m.source.id for m in self.measurements}

    def get_x(self) -> np.ndarray:
        return np.array([m.get_x() for m in self.measurements])

    def get_y(self) -> np.ndarray:
        return np.array([m.get_y() for m in self.measurements])

    def get_splits(self,
                   train_size: Optional[Union[float, int]] = 0.8,
                   validate_size: Optional[Union[float, int]] = 0.2,
                   n_splits: Optional[int] = 1,
                   seed: int = None) -> Iterable[Dataset]:
        # TODO: allow specific source splits
        """
        This function splits the measurements in a dataset into two splits, as specified by the
        provided parameters. Every source is in exactly one split.

        :param train_size: size of the train set. Can be a float, to indicate a fraction, or an integer to indicate an
                           absolute number of sources in each
                           split. If not specified, is the complement of the validate_size.
        :param validate_size: size of the validation set. Can be a float, to indicate a fraction, or an integer to
                          indicate an absolute number of sources
                          in each split. If not specified, is the complement of the train_size.
        :param n_splits: number of splits to ...
        :param seed: seed to ensure repeatability of the split

        """
        s = GroupShuffleSplit(n_splits=n_splits, random_state=seed, train_size=train_size, test_size=validate_size)
        source_ids = [m.source.id for m in self.measurements]

        for split in s.split(self.measurements, groups=source_ids):
            yield [MeasurementsDataset(measurements=list(map(lambda i: self.measurements[i], split_idx)))
                   for split_idx in split]

    def get_x_y_pairs(self,
                      seed: Optional[int] = None,
                      pairing_function: Optional[Callable] = partial(InstancePairing,
                                                                     different_source_limit='balanced'),
                      transformer: Optional[Callable] = AbsDiffTransformer) -> XYType:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        X, y = self.get_x(), self.get_y()
        X_pairs, y_pairs = pairing_function(seed=seed).transform(X, y)
        X_pairs = transformer().transform(X_pairs)
        return X_pairs, y_pairs


class MeasurementPairsDataset(Dataset):
    def __init__(self, measurement_pairs: Optional[List[MeasurementPair]] = None):
        super().__init__()
        self.measurement_pairs = measurement_pairs

    @property
    def source_ids(self) -> Set[int]:
        return set(chain.from_iterable(
            [[mp.measurement_a.source.id, mp.measurement_b.source.id] for mp in self.measurement_pairs]))

    def get_x(self) -> np.ndarray:
        return np.array([mp.get_x() for mp in self.measurement_pairs])

    def get_y(self) -> np.ndarray:
        return np.array([mp.get_y() for mp in self.measurement_pairs])

    def get_splits(self,
                   train_size: Optional[Union[float, int]] = 0.8,
                   validate_size: Optional[Union[float, int]] = 0.2,
                   n_splits: Optional[int] = 1,
                   seed: int = None) -> Iterable[Dataset]:
        # TODO: allow specific source splits
        """
        This function splits the measurement pairs in a dataset into two splits, as specified by the
        provided parameters. All measurement (pairs) from the same source are in the same split.

        :param train_size: size of the train set. Can be a float, to indicate a fraction, or an integer to indicate an
                           absolute number of measurement pairs in each split.  If not
                           specified, is the complement of the validate_size.
        :param validate_size: size of the validation set. Can be a float, to indicate a fraction, or an integer to
                          indicate an absolute number of measurements pairs in each split. If not specified, is the
                           complement of the train_size.
        :param n_splits: number of splits to ...
        :param seed: seed to ensure repeatability of the split
        """
        s = ShuffleSplit(n_splits=n_splits, random_state=seed, train_size=train_size, test_size=validate_size)
        source_ids = list(self.source_ids)
        for split in s.split(source_ids):
            yield [MeasurementPairsDataset(measurement_pairs=list(filter(
                lambda mp: mp.measurement_a.source.id in np.array(source_ids)[
                    split_idx] and mp.measurement_b.source.id in np.array(source_ids)[split_idx],
                self.measurement_pairs))) for split_idx in split]

    def get_x_y_pairs(self,
                      seed: Optional[int] = None,
                      transformer: Optional[Callable] = AbsDiffTransformer) -> XYType:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs. If
        pairs are already available, we return those.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        if 'score' in self.measurement_pairs[0].extra.keys():
            return self.get_x(), self.get_y()
        # If the measurement pair has no score, the values of the individual measurements first need to be
        # transformed to scores
        else:
            # the shape of the measurement values should be (m, f,2), with m=number of pairs, f=number of features
            # and 2 values (for the two measurements), to be compatible with the transformation function
            X_pairs = np.array([mp.get_measurement_values() for mp in self.measurement_pairs])
            y_pairs = np.array([mp.is_same_source for mp in self.measurement_pairs])
            X_pairs = transformer().transform(X_pairs)
            return X_pairs, y_pairs


class XTCDataset(MeasurementsDataset):
    def __init__(self, measurements_path):
        self.measurements_path = measurements_path
        super().__init__()

        with open(self.measurements_path, "r") as f:
            reader = csv.DictReader(f)
            measurements = [Measurement(source=Source(id=int(row['batchnumber']), extra={}),
                                        extra={'Repeat': int(row['measurement'])},
                                        value=np.array(list(map(float, row.values()))[2:])) for row in reader]
        self.measurements = measurements

    def __repr__(self):
        return "XTC dataset"


class GlassDataset(MeasurementsDataset):
    def __init__(self):
        super().__init__()

        datasets = {'duplo.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/'
                                 'elemental_composition_glass/main/duplo.csv',
                    'training.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/'
                                    'elemental_composition_glass/main/training.csv',
                    'triplo.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/'
                                  'elemental_composition_glass/main/triplo.csv'}
        glass_folder = os.path.join('resources', 'glass')

        measurements = []
        max_item = 0
        for file, url in datasets.items():
            download_dataset_file(glass_folder, file, url)
            path = os.path.join(glass_folder, file)
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


class ASRDataset(MeasurementPairsDataset):
    """
    A dataset containing paired measurements for the purpose of automatic speaker recognition.
    """
    def __init__(self, measurements_path, sources_path):
        self.measurements_path = measurements_path  # TODO: besluiten waar data te laten, nu nog inlezen vanaf schijf
        self.sources_path = sources_path
        super().__init__()

        with open(self.measurements_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        header_measurement_data = np.array(data[0][1:])
        measurement_data = np.array(data)[1:, 1:]

        recording_data = self.load_recording_annotations()

        mps = []
        for i in tqdm(range(measurement_data.shape[0]), desc='Reading recording measurement data'):
            filename_a = header_measurement_data[i]
            info_a = recording_data.get(filename_a.replace('_30s', ''))
            source_id_a = filename_a.split("_")[0]
            if info_a:  # check whether there is recording info present for the first file
                for j in range(i, measurement_data.shape[1]):
                    filename_b = header_measurement_data[j]
                    info_b = recording_data.get(filename_b.replace('_30s', ''))
                    source_id_b = filename_b.split("_")[0]
                    if info_b:  # check whether there is recording info present for the other file
                        mps.append(MeasurementPair(Measurement(Source(id=source_id_a,
                                                                      extra={'sex': info_a['sex'],
                                                                             'age': info_a['beller_leeftijd']}),
                                                               extra={'filename': filename_a,
                                                                      'net_duration': float(
                                                                          info_a['net duration'])}),
                                                   Measurement(Source(id=source_id_b,
                                                                      extra={'sex': info_b['sex'],
                                                                             'age': info_a['beller_leeftijd']}),
                                                               extra={'filename': filename_b,
                                                                      'net_duration': float(
                                                                          info_b['net duration'])}),
                                                   extra={'score': float(measurement_data[i, j])}))
        self.measurement_pairs = mps

    def load_recording_annotations(self) -> Mapping[str, Mapping[str, str]]:
        """
        Read annotations containing information of the recording and speaker.
        """
        with open(self.sources_path, 'r') as f:
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
