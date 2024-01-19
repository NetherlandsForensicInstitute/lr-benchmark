import csv
import os
import urllib.request
from abc import ABC
from typing import Iterable, Optional, List, Set, Union, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.pairing import CartesianPairing, BasePairing


class Dataset(ABC):
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
                   seed: int = None) -> Iterable['Dataset']:
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
            yield [Dataset(measurements=list(map(lambda i: self.measurements[i], split_idx))) for split_idx in split]

    def get_pairs(self,
                  seed: Optional[int] = None,
                  pairing_function: BasePairing = CartesianPairing()) -> List[MeasurementPair]:
        """
        Transforms a dataset into same source and different source pairs and
        returns two arrays of X_pairs and y_pairs where the X_pairs are by
        default transformed to the absolute difference between two pairs.

        Note that this method is different from sklearn TransformerMixin
        because it also transforms y.
        """
        return pairing_function.transform(self.measurements, seed=seed)


class XTCDataset(Dataset):
    def __init__(self, n_splits):
        super().__init__(n_splits)

        data_file = 'Champ_data.csv'
        url = "https://raw.githubusercontent.com/NetherlandsForensicInstitute/placeholder"  # todo publish to github
        print(f"{self.__repr__()} is not yet available for download")
        xtc_folder = os.path.join('resources', 'drugs_xtc')
        download_dataset_file(xtc_folder, data_file, url)
        df = pd.read_csv(os.path.join(xtc_folder, data_file), delimiter=',')
        features = ["Diameter", "Thickness", "Weight", "Purity"]

        X = df[features].to_numpy()
        y = df['batchnumber'].to_numpy()

        self.measurements = None, X, y

    def __repr__(self):
        return "XTC dataset"


class GlassDataset(Dataset):
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


class ASRDataset(Dataset):
    """
    A dataset containing paired measurements for the purpose of automatic speaker recognition.
    """

    def __init__(self, scores_path, meta_info_path):
        self.scores_path = scores_path
        self.meta_info_path = meta_info_path
        super().__init__()

        with open(self.scores_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        header_measurement_data = np.array(data[0][1:])
        measurement_data = np.array(data)[1:, 1:]

        recording_data = self.load_recording_annotations()

        measurements = []
        for i in tqdm(range(measurement_data.shape[0]), desc='Reading recording measurement data'):
            filename_a = header_measurement_data[i]
            info_a = recording_data.get(filename_a.replace('_30s', ''))
            source_id_a = filename_a.split("_")[0]
            if info_a:
                measurements.append(Measurement(
                                Source(id=source_id_a, extra={'sex': info_a['sex'], 'age': info_a['beller_leeftijd']}),
                                extra={'filename': filename_a, 'net_duration': float(info_a['net duration'])}))
        self.measurements = measurements

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
