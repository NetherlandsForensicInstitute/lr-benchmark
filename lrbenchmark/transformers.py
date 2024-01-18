import csv
from collections import defaultdict
from typing import Optional, Iterable, Mapping

import numpy as np
import sklearn
from lir.transformers import AbsDiffTransformer, InstancePairing
from sklearn.base import TransformerMixin
from tqdm import tqdm

from lrbenchmark.data.models import MeasurementPair, Measurement, Source
from lrbenchmark.typing import XYType


class DummyTransformer(TransformerMixin):
    """
    Simply returns the incoming data
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class DummyClassifier(TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        if len(X.shape) > 1 and X.shape[-1] != 1:
            raise ValueError(f"The dummy classifier can only be used on scores, which have one score per pair, not on "
                             f"data with shape {X.shape}.")
        return X


def pair_absdiff_transform(X: np.ndarray, y: np.ndarray, seed: Optional[int] = None) -> XYType:
    """
    Transforms a basic X y dataset into same source and different source pairs and returns
    an X y dataset where the X is the absolute difference between the two pairs.

    Note that this method is different from sklearn TransformerMixin because it also transforms y.
    """
    pairing = InstancePairing(different_source_limit='balanced', seed=seed)
    transformer = AbsDiffTransformer()
    X_pairs, y_pairs = pairing.transform(X, y)
    return transformer.transform(X_pairs), y_pairs


class PrecalculatedScorer:
    def __init__(self):
        self.scores_path = '/mnt/projects/fbda/Profi_ASR/data/scorematrix.csv'
        self.meta_info_path = '/mnt/projects/fbda/Profi_ASR/data/recordings_anon.txt'
        self.scores = None

    def fit(self, measurement_pairs: Iterable[MeasurementPair]):
        self.scores = defaultdict()
        with open(self.scores_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        header_measurement_data = np.array(data[0][1:])
        measurement_data = np.array(data)[1:, 1:]

        recording_data = self.load_recording_annotations()

        for i in tqdm(range(measurement_data.shape[0]), desc='Reading recording measurement data', position=0):
            filename_a = header_measurement_data[i]
            info_a = recording_data.get(filename_a.replace('_30s', ''))
            if info_a:  # check whether there is recording info present for the first file
                for j in range(i + 1, measurement_data.shape[1]):
                    filename_b = header_measurement_data[j]
                    info_b = recording_data.get(filename_b.replace('_30s', ''))
                    if info_b:  # check whether there is recording info present for the other file
                        self.scores[(filename_a, filename_b)] = float(measurement_data[i, j])
                        self.scores[(filename_b, filename_a)] = float(measurement_data[i, j])

    def transform(self, measurement_pairs: Iterable[MeasurementPair]):
        return [self.scores[measurement_pair.measurement_a.extra['filename'], measurement_pair.measurement_b.extra['filename']] for measurement_pair in measurement_pairs]

    def fit_transform(self, measurement_pairs: Iterable[MeasurementPair]):
        self.fit(measurement_pairs)
        return self.transform(measurement_pairs)

    def load_recording_annotations(self) -> Mapping[str, Mapping[str, str]]:
        """
        Read annotations containing information of the recording and speaker.
        """
        with open(self.meta_info_path, 'r') as f:
            reader = csv.DictReader(f, delimiter='\t')
            data = list(reader)

        return {elt['filename']: elt for elt in data}


class MeasurementPairScorer:
    def __init__(self, transformer, scorer):
        self.transformer = transformer()
        self.scorer = scorer()

    def fit(self, measurement_pairs: Iterable[MeasurementPair]):
        X = self.transformer.transform(np.array([mp.get_x() for mp in measurement_pairs]))
        y = [mp.is_same_source for mp in measurement_pairs]
        self.scorer.fit(X, y)

    def transform(self, measurement_pairs: Iterable[MeasurementPair]):
        X = self.transformer.transform(np.array([mp.get_x() for mp in measurement_pairs]))
        return self.scorer.predict_proba(X)[:,1]

    def fit_transform(self, measurement_pairs: Iterable[MeasurementPair]):
        self.fit(measurement_pairs)
        return self.transform(measurement_pairs)
