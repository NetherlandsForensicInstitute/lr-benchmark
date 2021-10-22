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
        raise NotImplementedError

    @property
    @abstractmethod
    def is_commonsource(self) -> bool:
        raise NotImplementedError


class CommonSourceKFoldDataset(Dataset, ABC):
    def __init__(self, n_splits):
        super().__init__()
        self.n_splits = n_splits
        self.preprocessor = None
        self._data = None

    @abstractmethod
    def load(self) -> XYType:
        raise NotImplementedError

    def get_x_y(self) -> XYType:
        if self._data is None:
            X, y = self.load()
            if self.preprocessor:
                X = self.preprocessor.fit_transform(X)
            self._data = (X, y)
        return self._data

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
