import csv
import os
import urllib.request
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Iterable, Optional, Callable, List, Set, Union, Mapping

import numpy as np
import pandas as pd
from lir.transformers import InstancePairing, AbsDiffTransformer
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit, StratifiedShuffleSplit
from tqdm import tqdm

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.typing import XYType


class Dataset(ABC):
    @abstractmethod
    def get_splits(self, group_by_source: bool = False, stratified: bool = False,
                   train_size: Optional[Union[float, int]] = 0.8, test_size: Optional[Union[float, int]] = 0.2,
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
        group_by_source: bool, optional
            Whether to split the dataset while keeping groups intact
        train_size: int, float, optional
            Fraction or number of data points to use for the training set. The default is 0.8, and if not specified,
            the complement of the test size will be used
        test_size: int, float, optional
            Fraction or number of data points to use for the test set. The default is 0.2, and if not specified,
            the complement of the train size will be used
        stratified: bool, optional
            Whether to split the dataset while keeping the ratio of classes

        Returns
        -------
        Iterable['Dataset']
            one or more Datasets as an iterable, each element being a subset
            of the original Dataset consisting of measurements and/or measurement
            pairs.

        """
        raise NotImplementedError

    def pop(self, fraction: float, seed: int = None) -> 'Dataset':
        """
        Draws a random sample from the data set.

        The returned data will be removed.

        Parameters
        ----------
        fraction : float
            The size of the sample as a fraction of the _original_ data set
            size, i.e. subsequent calls will return arrays of (approximately)
            the same size.
        seed : int, optional
            Optional random seed. The default is None.

        Raises
        ------
        NotImplementedError
            If the method is not implemented by this data set.

        Returns
        -------
        Dataset
            A 'Dataset' consisting of a subset of the instances of the original
            Dataset.
        """
        raise NotImplementedError


class CommonSourceDataset(Dataset, ABC):
    def __init__(self):
        super().__init__()

    def load(self):
        raise NotImplementedError


class CommonSourceMeasurementsDataset(CommonSourceDataset):
    def __init__(self, measurements: Optional[List[Measurement]] = None):
        super().__init__()
        self.measurements = measurements

        if self.measurements is None:
            self.load()

    @property
    def source_ids(self) -> Set[int]:
        return {m.source.id for m in self.measurements}

    def get_x(self) -> np.ndarray:
        return np.array([m.get_x() for m in self.measurements])

    def get_y(self) -> np.ndarray:
        return np.array([m.get_y() for m in self.measurements])

    def get_splits(self,
                   group_by_source: bool,
                   stratified: bool,
                   train_size: Optional[Union[float, int]],
                   test_size: Optional[Union[float, int]],
                   seed: int) -> Iterable[Dataset]:
        """
        This function splits the measurements in a dataset into two splits, as specified by the
        provided parameters.

        :param stratified: boolean that indicates whether the dataset should be split while preserving the ratio of
                           classes in y in both splits.
        :param group_by_source: boolean that indicates whether the dataset should be split along group lines, ensuring
                                each group to be in only a single split.
        :param train_size: size of the train set. Can be a float, to indicate a fraction, or an integer to indicate an
                           absolute number of measurements (if stratified) or sources (if group_by_source) in each
                           split. If not specified, is the complement of the test_size.
        :param test_size: size of the test set. Can be a float, to indicate a fraction, or an integer to indicate an
                          absolute number of measurements (if stratified) or sources (if group_by_source) in each
                          split. If not specified, is the complement of the train_size.
        :param seed: seed to ensure repeatability of the split

        When splitting measurements, a regular split is performed when both group and stratified are False. If group is
        True the split can be made based on the sources. Stratification is not applicable if splitting on measurements,
        as these do not have a y.
        """
        if stratified:
            raise ValueError('It is not possible to split the dataset stratified, when using measurements')

        if not group_by_source:
            s = ShuffleSplit(n_splits=1, random_state=seed, train_size=train_size, test_size=test_size)
            source_ids = None
        else:
            s = GroupShuffleSplit(n_splits=1, random_state=seed, train_size=train_size, test_size=test_size)
            source_ids = [m.source.id for m in self.measurements]

        split = s.split(self.measurements, groups=source_ids)[0]
        return [CommonSourceMeasurementsDataset(measurements=list(map(lambda i: self.measurements[i], split_idx)))
                for split_idx in split]

    def get_x_y_pairs(self,
                      seed: Optional[int] = None,
                      pairing_function: Optional[Callable] = partial(InstancePairing,
                                                                     different_source_limit='balanced'),
                      transformer: Optional[Callable] = AbsDiffTransformer) -> XYType:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs. If
        pairs are already available, we return those.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        X, y = self.get_x(), self.get_y()
        X_pairs, y_pairs = pairing_function(seed=seed).transform(X, y)
        X_pairs = transformer().transform(X_pairs)
        return X_pairs, y_pairs


class CommonSourceMeasurementPairsDataset(CommonSourceDataset):
    def __init__(self, measurement_pairs: Optional[List[MeasurementPair]] = None):
        super().__init__()
        self.measurement_pairs = measurement_pairs

        if self.measurement_pairs is None:
            self.load()

    @property
    def source_ids(self) -> Set[int]:
        return set(chain.from_iterable(
            [[mp.measurement_a.source.id, mp.measurement_b.source.id] for mp in self.measurement_pairs]))

    def get_x(self) -> np.ndarray:
        return np.array([mp.get_x() for mp in self.measurement_pairs])

    def get_y(self) -> np.ndarray:
        return np.array([mp.get_y() for mp in self.measurement_pairs])

    def get_splits(self, stratified: bool = False, group_by_source: bool = False,
                   train_size: Optional[Union[float, int]] = 0.8, test_size: Optional[Union[float, int]] = 0.2,
                   seed: int = None) -> Iterable[Dataset]:
        """
        This function splits the measurement pairs in a dataset into two splits, as specified by the
        provided parameters.

        :param stratified: boolean that indicates whether the dataset should be split while preserving the ratio of
                           classes in y in both splits.
        :param group_by_source: boolean that indicates whether the dataset should be split along group lines, ensuring
                                each group to be in only a single split.
        :param train_size: size of the train set. Can be a float, to indicate a fraction, or an integer to indicate an
                           absolute amount of measurements, measurement pairs or groups in each split.  If not
                           specified, is the complement of the test_size.
        :param test_size: size of the test set. Can be a float, to indicate a fraction, or an integer to indicate an
                          absolute amount of measurements, measurement pairs or groups in each split. If not specified,
                          is the complement of the train_size.
        :param seed: seed to ensure repeatability of the split
        """
        yield from self.get_splits_measurement_pairs(group_by_source, stratified, train_size, test_size, seed)

    def get_splits(self,
                   group_by_source: bool,
                   stratified: bool,
                   train_size: Optional[Union[float, int]],
                   test_size: Optional[Union[float, int]],
                   seed: int) -> Iterable[Dataset]:
        """
        Iterate overThis function splits the measurement pairs in a dataset into two splits, as specified by the
        provided parameters.

        :param stratified: boolean that indicates whether the dataset should be split while preserving the ratio of
                           classes in y in both splits.
        :param group_by_source: boolean that indicates whether the dataset should be split along group lines, ensuring
                                each group to be in only a single split.
        :param train_size: size of the train set. Can be a float, to indicate a fraction, or an integer to indicate an
                           absolute number of measurement pairs in each split.  If not
                           specified, is the complement of the test_size.
        :param test_size: size of the test set. Can be a float, to indicate a fraction, or an integer to indicate an
                          absolute number of measurements pairs in each split. If not specified,
                          is the complement of the train_size.
        :param seed: seed to ensure repeatability of the split

        When splitting measurement pairs, a regular split is performed when both group and stratified are False. A
        split based on y or the source is made when respectively stratified or group are True. It is not possible to
        split with both group and stratified True, as it is not possible to guarantee grouped splits have a similar
        number of instances for each class.
        """
        if not group_by_source:
            if stratified:
                s = StratifiedShuffleSplit(n_splits=1, random_state=seed, train_size=train_size,
                                           test_size=test_size)
                y = [mp.is_same_source for mp in self.measurement_pairs]
            else:
                s = ShuffleSplit(n_splits=1, random_state=seed, train_size=train_size, test_size=test_size)
                y = None

            for split in s.split(self.measurement_pairs, y):
                yield [CommonSourceMeasurementPairsDataset(measurement_pairs=list(
                    map(lambda i: self.measurement_pairs[i], split_idx))) for split_idx in split]
        if not stratified:
            s = ShuffleSplit(n_splits=1, random_state=seed, train_size=train_size, test_size=test_size)
            source_ids = list(self.source_ids)
            for split in s.split(source_ids):
                yield [CommonSourceMeasurementPairsDataset(measurement_pairs=list(filter(
                    lambda mp: mp.measurement_a.source.id in np.array(source_ids)[
                        split_idx] and mp.measurement_b.source.id in np.array(source_ids)[split_idx],
                    self.measurement_pairs))) for split_idx in split]
        if group_by_source and stratified:
            raise ValueError("Cannot specify both group and stratified when measurement pairs are provided")

    def get_x_y_pairs(self, transformer: Optional[Callable] = AbsDiffTransformer) -> XYType:
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


class XTCDataset(CommonSourceDataset):

    def __init__(self, n_splits):
        super().__init__(n_splits)

    def load(self) -> XYType:
        """
        Loads XTC dataset
        """
        data_file = 'Champ_data.csv'
        url = "https://raw.githubusercontent.com/NetherlandsForensicInstitute/placeholder"  # todo publish to github
        print(f"{self.__repr__()} is not yet available for download")
        xtc_folder = os.path.join('resources', 'drugs_xtc')
        download_dataset_file(xtc_folder, data_file, url)
        df = pd.read_csv(os.path.join(xtc_folder, data_file), delimiter=',')
        features = ["Diameter", "Thickness", "Weight", "Purity"]

        X = df[features].to_numpy()
        y = df['batchnumber'].to_numpy()

        return X, y

    def __repr__(self):
        return "XTC dataset"


class GlassDataset(CommonSourceMeasurementsDataset):

    def __init__(self):
        super().__init__()

    def load(self):
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


class ASRDataset(CommonSourceMeasurementPairsDataset):
    """
    A dataset containing paired measurements for the purpose of automatic speaker recognition.
    """

    def __init__(self, measurements_path, sources_path):
        self.measurements_path = measurements_path  # TODO: besluiten waar data te laten, nu nog inlezen vanaf schijf
        self.sources_path = sources_path
        super().__init__()

    def load(self):
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
