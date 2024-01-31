import csv
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

import numpy as np
from confidence import Configuration
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from lrbenchmark.data.models import MeasurementPair
from lrbenchmark.typing import PathLike


class DummyTransformer(TransformerMixin):
    """
    Simply returns the incoming data
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class BaseScorer(BaseEstimator, ClassifierMixin, ABC):
    @abstractmethod
    def fit(self, measurement_pairs: Iterable[MeasurementPair]):
        raise NotImplementedError

    @abstractmethod
    def predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def fit_predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        raise NotImplementedError


class PrecalculatedScorerASR(BaseScorer):
    def __init__(self, scores_path: PathLike):
        self.scores = {}
        self.scores_path = scores_path

    def fit(self, measurement_pairs: Iterable[MeasurementPair]):
        with open(self.scores_path, "r") as f:
            reader = csv.reader(f)
            data = list(reader)
        header_measurement_data = np.array(data[0][1:])
        measurement_data = np.array(data)[1:, 1:]

        for i in tqdm(range(measurement_data.shape[0]), desc='Reading scores from file', position=0):
            filename_a = header_measurement_data[i]
            for j in range(i + 1, measurement_data.shape[1]):
                filename_b = header_measurement_data[j]
                self.scores[(filename_a, filename_b)] = float(measurement_data[i, j])
                self.scores[(filename_b, filename_a)] = float(measurement_data[i, j])

    def predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        return np.array([self.scores[(measurement_pair.measurement_a.extra['filename'],
                                      measurement_pair.measurement_b.extra['filename'])]
                         for measurement_pair in measurement_pairs])

    def fit_predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        self.fit(measurement_pairs)
        return self.predict(measurement_pairs)


class MeasurementPairScorer(BaseScorer):
    def __init__(self, scorer, preprocessors: Optional[Sequence[TransformerMixin]]):
        self.transformer = Pipeline([(v.__class__.__name__, v) for v in preprocessors]) if preprocessors else None
        self.scorer = scorer()

    def fit(self, measurement_pairs: Iterable[MeasurementPair]):
        X = np.array([mp.get_x() for mp in measurement_pairs])
        if self.transformer:
            X = self.transformer.fit_transform(X)
        y = [mp.is_same_source for mp in measurement_pairs]
        self.scorer.fit(X, y)

    def predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        if self.transformer:
            X = self.transformer.transform(np.array([mp.get_x() for mp in measurement_pairs]))
        return self.scorer.predict_proba(X)[:, 1]

    def fit_predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        self.fit(measurement_pairs)
        return self.predict(measurement_pairs)
