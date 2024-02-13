import csv
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

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
        """
        Scorer specifically for ASR that retrieved the predictions from a precalculated matrix of scores.
        The scores are in a symmetrical matrix, read from the csv file in scores_path. The source_indices is a mapping
        of each source_id and the indices of its instances in the scores matrix. The measurement_indices is a mapping of
        each measurement_id to its index in the scores matrix.

        :param scores_path: the path to a csv to read the scores from
        """
        self.scores_path = scores_path
        self.scores = None
        self.source_indices = None
        self.measurement_indices = None

    def fit(self, measurement_pairs: Iterable[MeasurementPair]):
        if not self.scores:
            with open(self.scores_path, "r") as f:
                reader = csv.reader(f)
                data = np.array(list(reader))
            header_measurement_data = data[0, 1:]
            row_header_measurement_data = data[1:, 0]
            if not np.array_equal(header_measurement_data, row_header_measurement_data):
                raise ValueError("Column headers and row headers not equal.")

            self.scores = data[1:, 1:].astype(float)

            sources_list = [s.split('_')[0] for s in header_measurement_data]
            self.source_indices = {x: np.where(np.array(sources_list) == x)[0] for x in set(sources_list)}
            self.measurement_indices = {x: np.where(np.array(header_measurement_data) == x)[0][0] for x in
                                        set(header_measurement_data)}

    def predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        return np.array([self.scores[self.measurement_indices.get(measurement_pair.measurement_a.id),
                                     self.measurement_indices.get(measurement_pair.measurement_b.id)]
                         for measurement_pair in measurement_pairs])

    def fit_predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        # If scores are already available, no need to fit again
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
        X = np.array([mp.get_x() for mp in measurement_pairs])
        if self.transformer:
            X = self.transformer.transform(X)
        return self.scorer.predict_proba(X)[:, 1]

    def fit_predict(self, measurement_pairs: Iterable[MeasurementPair]) -> np.ndarray:
        self.fit(measurement_pairs)
        return self.predict(measurement_pairs)
