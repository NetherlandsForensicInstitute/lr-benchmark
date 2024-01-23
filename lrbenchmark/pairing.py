import itertools
from abc import ABC, abstractmethod
import random
from typing import List, Iterable, Optional

import sklearn.base

from lrbenchmark.data.models import Measurement, MeasurementPair


class BasePairing(sklearn.base.TransformerMixin, ABC):
    @abstractmethod
    def fit(self, measurements: Iterable[Measurement]):
        raise NotImplementedError

    @abstractmethod
    def transform(self, measurements: Iterable[Measurement], seed: Optional[int] = None) -> List[MeasurementPair]:
        raise NotImplementedError


class CartesianPairing(BasePairing):
    """
    Creates pairs of instances.

    This transformer takes a list of Measurement as input, and returns a list of MeasurementPair.
    By default, the list of MeasurementPair contains all possible pairs of Measurement, except the combination
    of a Measurement with itself.
    """

    def fit(self, measurements: Iterable[Measurement]):
        return self

    def transform(self, measurements: Iterable[Measurement], seed: Optional[int] = None) -> List[MeasurementPair]:
        return [MeasurementPair(*mp) for mp in itertools.combinations(measurements, 2)]


class BalancedPairing(BasePairing):
    """
    Creates pairs of instances, with an equal amount of same source and different source pairs.
    """

    def fit(self, measurements: Iterable[Measurement]):
        return self

    def transform(self, measurements: Iterable[Measurement], seed=None) -> List[MeasurementPair]:
        random.seed(seed)
        all_pairs = [MeasurementPair(*mp) for mp in itertools.combinations(measurements, 2)]
        same_source_pairs = [a for a in all_pairs if a.is_same_source]
        different_source_pairs = [a for a in all_pairs if not a.is_same_source]
        n_pairs = min(len(same_source_pairs), len(different_source_pairs))
        same_source_pairs = random.sample(same_source_pairs, k=n_pairs)
        different_source_pairs = random.sample(different_source_pairs, k=n_pairs)
        selected_pairs = same_source_pairs + different_source_pairs
        random.shuffle(selected_pairs)
        return selected_pairs
