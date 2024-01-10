import csv
import os
import urllib.request
from abc import ABC, abstractmethod
from functools import partial
from itertools import chain
from typing import Iterable, Optional, Callable, List

import numpy as np
import pandas as pd
from lir.transformers import InstancePairing, AbsDiffTransformer
from sklearn.model_selection import StratifiedGroupKFold, KFold, GroupKFold, GroupShuffleSplit

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.typing import TrainTestPair, XYType


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
        Iterable[TrainTestPair]
            one or more subsets as an iterable, each element being a tuple
            `((X_train, y_train), (X_test, y_test))`, where:
                - `X_train` is a `numpy.ndarray` of features for records in the training set
                - `y_train` is a `numpy.ndarray` of labels for records in the training set
                - `X_test` is a `numpy.ndarray` of features for records in the test set
                - `y_test` is a `numpy.ndarray` of labels for records in the test set

        """
        raise NotImplementedError

    def pop(self, fraction: float, seed: int = None) -> XYType:
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
        XYType
            A tuple of `(X, y)`, with `X` being numpy arrays of features and
            `y` the corresponding labels.
        """
        raise NotImplementedError


class CommonSourceKFoldDataset(Dataset, ABC):
    def __init__(self,
                 n_splits,
                 measurements: Optional[List[Measurement]] = None,
                 measurement_pairs: Optional[List[MeasurementPair]] = None):
        super().__init__()
        self.n_splits = n_splits
        self.measurements = measurements
        self.measurement_pairs = measurement_pairs

        if self.measurements is None and self.measurement_pairs is None:
            self.load()

    def load(self) -> Iterable[Measurement]:
        raise NotImplementedError

    @property
    def source_ids(self):
        if self.measurements:
            return set([m.source.id for m in self.measurements])
        else:
            return set(chain.from_iterable([[m.measurement_a.source.id,
                                             m.measurement_b.source.id] for m in self.measurement_pairs]))

    def get_x_measurement(self) -> np.ndarray:
        return np.array([m.get_x() for m in self.measurements])

    def get_y_measurement(self) -> np.ndarray:
        return np.array([m.get_y() for m in self.measurements])

    def get_x_y_measurement(self) -> XYType:
        return self.get_x_measurement(), self.get_y_measurement()

    def get_splits(self, seed: int = None) -> Iterable[Dataset]:
        # X, y = self.get_x_y()
        # sources = self.sources
        # cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
        #                           random_state=seed)
        # # cv.split requires an x, y and groups. We don't have y yet, therefore we set it to -1.
        # for train_idxs, test_idxs in cv.split(X, y=np.array([-1] * len(X)),
        #                                       groups=y):
        #     yield (X[train_idxs], y[train_idxs]), (X[test_idxs], y[test_idxs])

        if self.measurements:
            cv = GroupShuffleSplit(n_splits=self.n_splits, random_state=seed)
            for splits in cv.split(self.measurements, groups=[m.source.id for m in self.measurements]):
                yield [CommonSourceKFoldDataset(n_splits=None,
                                                measurements=[self.measurements[i] for i in split]) for split in splits]
        else:
            kf = KFold(n_splits=self.n_splits)
            source_ids = list(self.source_ids)
            for splits in kf.split(source_ids):
                yield [CommonSourceKFoldDataset(
                    n_splits=None,
                    measurement_pairs=list(filter(lambda mp: mp.measurement_a.source in source_ids[split] and
                                                             mp.measurement_b.source in source_ids[split],
                                                  self.measurement_pairs))) for split in splits]

    def get_x_y_pairs(self,
                      pairing_function: Optional[Callable]=partial(InstancePairing, different_source_limit='balanced', seed=42),
                      transformer: Optional[Callable]=AbsDiffTransformer):
        """
        Transforms a basic X y dataset into same source and different source pairs and returns
        an X y dataset where the X is the absolute difference between the two pairs.

        Note that this method is different from sklearn TransformerMixin because it also transforms y.
        """
        if self.measurement_pairs:
            return self.measurement_pairs.get_x(), self.measurement_pairs.get_y()
        else:
            X, y = self.get_x_y_measurement()
            X_pairs, y_pairs = pairing_function().transform(X, y)
            # X_pairs = transformer().transform(X_pairs)
            return X_pairs, y_pairs


class InMemoryCommonSourceKFoldDataset(CommonSourceKFoldDataset):
    def __init__(self, X, y, n_splits):
        self._X = X
        self._y = y
        super().__init__(n_splits=n_splits)

    def load(self) -> XYType:
        return self._X, self._y

    def __repr__(self):
        return "InMemoryDataset"


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
                                value=np.array(
                                    list(map(float, row.values()))[3:])) for
                    row in reader]
                max_item = measurements_tmp[-1].source.id
                measurements.extend(measurements_tmp)
        self.measurements = measurements  # X, y

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
