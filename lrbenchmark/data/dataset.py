import csv
import os
import urllib.request
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Iterable, Optional, Callable, List, Set, Union, Mapping, Tuple

import numpy as np
import pandas as pd
from lir.transformers import InstancePairing, AbsDiffTransformer
from sklearn.model_selection import ShuffleSplit, GroupShuffleSplit
from tqdm import tqdm

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.refnorm import refnorm
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

    def load(self):
        raise NotImplementedError


class MeasurementsDataset(Dataset):
    def __init__(self, measurements: Optional[List[Measurement]] = None):
        super().__init__()
        self.measurements = measurements

        if self.measurements is None:
            # TODO: self.measurements = self.load(): make it more explicit
            self.load()

    def load(self):
        raise NotImplementedError

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

        if self.measurement_pairs is None:
            # TODO: self.measurement_pairs = self.load(): make it more explicit
            self.load()

    def load(self):
        raise NotImplementedError

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

    def get_x_y(self,
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

    def get_refnorm_split(self,
                          refnorm_size: Optional[Union[float, int]],
                          seed: int) -> Tuple['MeasurementPairsDataset',
                                              Optional['MeasurementPairsDataset']]:
        """
        Splits the measurement pairs in a dataset (used for training and validation) and a refnorm dataset. The
        split is done based on the source ids. The refnorm dataset is then further processed to contain only those
        measurement pairs for which exactly one of the source ids of the measurements is in the `refnorm dataset`, and
        the other is in the `dataset`.

        :param refnorm_size: The size of the refnorm set, this can be a float, to indicate a fraction, or an integer
                             to indicate an absolute amount of source_ids in each split. If not provided, the dataset
                             will not be split and the refnorm dataset will be None
        :param seed: Ensures repeatability of the experiments
        """
        if not refnorm_size:
            return self, None
        dataset, refnorm_dataset = list(
            self.get_splits(train_size=None, validate_size=refnorm_size, seed=seed))[0]
        refnorm_measurement_pairs = list(filter(lambda x: (x.measurement_a.source.id in refnorm_dataset.source_ids) ^
                                                          (x.measurement_b.source.id in refnorm_dataset.source_ids),
                                                self.measurement_pairs))
        refnorm_dataset = MeasurementPairsDataset(measurement_pairs=refnorm_measurement_pairs)
        return dataset, refnorm_dataset

    @staticmethod
    def select_refnorm_measurement_pairs(
            measurement: Measurement,
            source_ids_to_exclude: List[Union[int, str]],
            refnorm_dataset: 'MeasurementPairsDataset') -> List[MeasurementPair]:
        """
        Finds in the refnorm dataset the measurement pairs for which one of the measurements is equal to the provided
        measurement, and the other measurement has a source_id that is not in the list of source ids to exclude.

        :param measurement: the measurement which should be present in all selected refnorm measurement pairs
        :param source_ids_to_exclude: source ids from which measurements should not be selected as the complementary
        :param refnorm_dataset: the dataset to select the appropriate measurement pairs from.
        """
        selected_measurement_pairs = []

        for rn_pair in refnorm_dataset.measurement_pairs:
            # either measurement a in the refnorm measurement pair should be equal to the provided measurement
            if ((rn_pair.measurement_a == measurement and
                 rn_pair.measurement_b.source.id not in source_ids_to_exclude) or
                    # or measurement b in the refnorm measurement pair should be equal to the provided measurement
                    (rn_pair.measurement_b == measurement and
                     rn_pair.measurement_a.source.id not in source_ids_to_exclude)):
                selected_measurement_pairs.append(rn_pair)
        return selected_measurement_pairs

    def perform_refnorm(self,
                        refnorm_dataset: 'MeasurementPairsDataset',
                        source_ids_to_exclude: List[Union[int, str]]):
        """
        Transform the scores of the measurement pairs with reference normalization. For each measurement in the
        measurement pair, the appropriate refnorm measurement pairs are selected (i.e. all pairs of which one of the
        measurements is equal to the measurement that has to be normalized, and the other measurement has a source_id
        that is not in the `source_ids_to_exclude` list or equal to the source ids in the measurement pair).
        Once the refnorm pairs are selected, their scores are extracted and used for the transformation. The normalized
        score is replaced in the measurement pair.

        :param refnorm_dataset: the dataset from which to select measurement pairs to perform the refnorm transformation
        :param source_ids_to_exclude: list of source_ids which the complementary measurement is not allowed to have.
        """
        for mp in tqdm(self.measurement_pairs, desc="Performing reference normalization", position=0):
            refnorm_pairs_m_a = self.select_refnorm_measurement_pairs(
                measurement=mp.measurement_a,
                source_ids_to_exclude=[mp.measurement_a.source.id, mp.measurement_b.source.id] + source_ids_to_exclude,
                refnorm_dataset=refnorm_dataset)
            scores_m_a = [mp.score for mp in refnorm_pairs_m_a]
            refnorm_pairs_m_b = self.select_refnorm_measurement_pairs(
                measurement=mp.measurement_b,
                source_ids_to_exclude=[mp.measurement_a.source.id, mp.measurement_a.source.id] + source_ids_to_exclude,
                refnorm_dataset=refnorm_dataset)
            scores_m_b = [mp.score for mp in refnorm_pairs_m_b]
            normalized_score = refnorm(mp.score, scores_m_a, scores_m_b)
            mp.extra['score'] = normalized_score


class XTCDataset(MeasurementsDataset):
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


class GlassDataset(MeasurementsDataset):
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


class ASRDataset(MeasurementPairsDataset):
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
                for j in range(i + 1, measurement_data.shape[1]):
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
