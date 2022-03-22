import os
import urllib.request
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from lrbenchmark.typing import TrainTestPair, XYType


class Dataset(ABC):
    @abstractmethod
    def get_splits(self, seed: int = None) -> Iterable[TrainTestPair]:
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

    @property
    @abstractmethod
    def is_commonsource(self) -> bool:
        """
        Binary flag to indicate whether this data set is designed to develop
        common-source models.

        A data set is designed to either develop specific-source models or
        common-source models. A specific-source data set typically has two
        class labels: `0` indicates the defense's hypothesis and `1` for the
        prosecutor's hypothesis. Both the training set and the test set sample
        from both classes. A common-source data set has multiple class labels,
        and if the data set is split, a class label should not appear in more
        than one split.

        Returns
        -------
        bool
            `True` if the data set is designed to develop common-source models;
            `False` if the data set is designed to develop specific-source models.

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
    def __init__(self, n_splits):
        super().__init__()
        self.n_splits = n_splits
        self.preprocessor = None
        self._data = None
        self._preprocessed_data = None

    @abstractmethod
    def load(self) -> XYType:
        raise NotImplementedError

    def get_x_y(self) -> XYType:
        if self._data is None:
            X, y = self.load()
            self._data = (X, y)
        else:
            (X, y) = self._data

        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)
        self._preprocessed_data = (X, y)
        return self._preprocessed_data

    def get_splits(self, seed: int = None) -> Iterable[TrainTestPair]:
        X, y = self.get_x_y()

        cv = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                  random_state=seed)
        # cv.split requires an x, y and groups. We don't have y yet, therefore we set it to -1.
        for train_idxs, test_idxs in cv.split(X, y=np.array([-1] * len(X)),
                                              groups=y):
            yield (X[train_idxs], y[train_idxs]), (X[test_idxs], y[test_idxs])

    @property
    def is_commonsource(self) -> bool:
        return True


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

    def load(self) -> XYType:
        datasets = {
            'duplo.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/duplo.csv',
            'training.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/training.csv',
            'triplo.csv': 'https://raw.githubusercontent.com/NetherlandsForensicInstitute/elemental_composition_glass/main/triplo.csv'
        }
        glass_folder = os.path.join('resources', 'glass')

        features = ["K39", "Ti49", "Mn55", "Rb85", "Sr88", "Zr90", "Ba137",
                    "La139", "Ce140", "Pb208"]
        df = None

        for file, url in datasets.items():
            download_dataset_file(glass_folder, file, url)
            df_temp = pd.read_csv(os.path.join(glass_folder, file),
                                  delimiter=',')
            # The Item column starts with 1 in each file,
            # this is making it ascending across different files
            df_temp['Item'] = df_temp['Item'] + max(
                df['Item']) if df is not None else df_temp['Item']
            # the data from all 3 files is added together to make one dataset
            df = pd.concat([df, df_temp]) if df is not None else df_temp

        X = df[features].to_numpy()
        y = df['Item'].to_numpy()

        return X, y

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
