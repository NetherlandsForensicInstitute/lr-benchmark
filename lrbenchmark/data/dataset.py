import csv
import os
import urllib.request
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Iterable, Optional, Callable, List, Set

import numpy as np
import pandas as pd
from lir.transformers import InstancePairing, AbsDiffTransformer
from sklearn.model_selection import KFold, GroupShuffleSplit

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.typing import XYType


class Dataset(ABC):
    @abstractmethod
    def get_splits(self, seed: int = None) -> Iterable['Dataset']:
        """
        Retrieve data from this dataset.

        This function is responsible for splitting the data in subset for
        training and testing in a way that is appropriate for the data set.
        Depending on the implementation, the data set may be returned at once,
        as a K-fold series, or otherwise.

        Parameters
        ----------
        seed : int, optional
            Optional random seed to be used for splitting. The default is None.

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


class CommonSourceKFoldDataset(Dataset, ABC):
    def __init__(self,
                 n_splits: Optional[int],
                 measurements: Optional[List[Measurement]] = None,
                 measurement_pairs: Optional[List[MeasurementPair]] = None):
        super().__init__()
        self.n_splits = n_splits
        self.measurements = measurements
        self.measurement_pairs = measurement_pairs

        if self.measurements and self.measurement_pairs:
            raise ValueError("Dataset cannot have both measurements and"
                             "measurement pairs.")
        if self.measurements is None and self.measurement_pairs is None:
            self.load()

    def load(self):
        raise NotImplementedError

    @property
    def source_ids(self) -> Set[int]:
        if self.measurements:
            return set([m.source.id for m in self.measurements])
        else:
            return set(chain.from_iterable([[mp.measurement_a.source.id,
                                             mp.measurement_b.source.id] for mp in self.measurement_pairs]))

    def get_x_measurement(self) -> np.ndarray:
        return np.array([m.get_x() for m in self.measurements])

    def get_y_measurement(self) -> np.ndarray:
        return np.array([m.get_y() for m in self.measurements])

    def get_x_y_measurement(self) -> XYType:
        return self.get_x_measurement(), self.get_y_measurement()

    def get_x_measurement_pair(self) -> np.ndarray:
        return np.array([mp.get_x() for mp in self.measurement_pairs])

    def get_y_measurement_pair(self) -> np.ndarray:
        return np.array([mp.get_y() for mp in self.measurement_pairs])

    def get_x_y_measurement_pair(self) -> XYType:
        return self.get_x_measurement_pair(), self.get_y_measurement_pair()

    def get_splits(self, seed: int = None) -> Iterable[Dataset]:
        if self.measurements:  # split the measurements if available
            cv = GroupShuffleSplit(n_splits=self.n_splits, random_state=seed)
            for splits in cv.split(self.measurements, groups=[m.source.id for m in self.measurements]):
                yield [CommonSourceKFoldDataset(n_splits=None,
                                                measurements=[self.measurements[i] for i in split]) for split in splits]
        else:  # split the measurement pairs
            kf = KFold(n_splits=self.n_splits)
            source_ids = list(self.source_ids)
            for splits in kf.split(source_ids):
                yield [CommonSourceKFoldDataset(
                    n_splits=None,
                    measurement_pairs=list(filter(lambda mp: mp.measurement_a.source in source_ids[split] and
                                                             mp.measurement_b.source in source_ids[split],
                                                  self.measurement_pairs))) for split in splits]

    def get_x_y_pairs(self,
                      seed: Optional[int] = None,
                      pairing_function: Optional[Callable] = partial(InstancePairing, different_source_limit='balanced'),
                      transformer: Optional[Callable] = AbsDiffTransformer) -> XYType:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs. If
        pairs are already available, we return those.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        if self.measurement_pairs:
            if 'score' in self.measurement_pairs[0].extra.keys():
                return self.get_x_y_measurement_pair()
            # If the measurement pair has no score, the values of the individual measurements first need to be
            # transformed to scores
            else:
                # the shape of the measurement values should be (m, f,2), with m=number of pairs, f=number of features
                # and 2 values (for the two measurements), to be compatible with the transformation function
                X_pairs = np.array([mp.get_measurement_values() for mp in self.measurement_pairs])
                y_pairs = np.array([mp.is_same_source for mp in self.measurement_pairs])
                X_pairs = transformer().transform(X_pairs)
                return X_pairs, y_pairs
        else:
            X, y = self.get_x_y_measurement()
            X_pairs, y_pairs = pairing_function(seed=seed).transform(X, y)
            X_pairs = transformer().transform(X_pairs)
            return X_pairs, y_pairs


class XTCDataset(CommonSourceKFoldDataset):

    def __init__(self, n_splits):
        super().__init__(n_splits)

    def load(self) -> XYType:
        """
        Loads XTC dataset
        """
        data_file = 'Champ_data.csv'
        url = "https://raw.githubusercontent.com/NetherlandsForensicInstitute/placeholder"  # @todo publish dataset to github
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


class GlassDataset(CommonSourceKFoldDataset):

    def __init__(self, n_splits):
        super().__init__(n_splits)

    def load(self):
        datasets = {
            'duplo.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/duplo.csv',
            'training.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/training.csv',
            'triplo.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/triplo.csv'
        }
        glass_folder = os.path.join('resources', 'glass')

        measurements = []
        max_item = 0
        for file, url in datasets.items():
            download_dataset_file(glass_folder, file, url)
            path = os.path.join(glass_folder, file)
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                measurements_tmp = [
                    Measurement(source=Source(id=int(row['Item']) + max_item, extra={}),
                                extra={'Piece': int(row['Piece'])},
                                # the values consists of measurements of ten elemental
                                # compositions, which start at the fourth position of
                                # each row
                                value=np.array(
                                    list(map(float, row.values()))[3:])) for
                    row in reader]
                # The item values start with 1 in each file,
                # this is making it ascending across different files
                max_item = measurements_tmp[-1].source.id
                measurements.extend(measurements_tmp)
        self.measurements = measurements

    def __repr__(self):
        return "Glass dataset"


def download_dataset_file(folder: str, file: str, url: str):
    location = os.path.join(folder, file)
    if not os.path.isfile(location):
        print(f'downloading {file}')
        try:
            urllib.request.urlretrieve(url, location)
        except Exception as e:
            print(f"Could not download {file} because of: {e}")
