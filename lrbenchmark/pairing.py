import itertools
import random
from abc import ABC, abstractmethod
from typing import List, Iterable, Optional, Mapping, Tuple

import sklearn.base

from lrbenchmark.data.models import Measurement, MeasurementPair
from lrbenchmark.utils import pair_complies_with_properties, select_max_measurements_per_source


class BasePairing(sklearn.base.TransformerMixin, ABC):
    @abstractmethod
    def fit(self, measurements: Iterable[Measurement]):
        raise NotImplementedError

    @abstractmethod
    def transform(self,
                  measurements: Iterable[Measurement],
                  pairing_properties: Tuple[Mapping[str, str], Mapping[str, str]] = ({}, {}),
                  max_m_per_source: Optional[int] = None,
                  seed: Optional[int] = None) -> List[MeasurementPair]:
        raise NotImplementedError


class CartesianPairing(BasePairing):
    """
    Creates pairs of instances.

    This transformer takes a list of Measurement as input, and returns a list of MeasurementPair.
    By default, the list of MeasurementPair contains all possible pairs of Measurement, except the combination
    of a Measurement with itself. Pairs are considered symmetric, so pairing of measurements [a,b,c] will return
    a-b, a-c, b-c but not also b-a.
    It is possible to consider 'pairing_properties', meaning only measurement pairs will be
    created of which one measurement is similar to the first part of the `pairing_properties` and the other measurement
    is similar to the other part.
    It is also possible to select a maximum amount of measurements per source via the 'max_m_per_source' parameter.
    """

    def fit(self, measurements: Iterable[Measurement]):
        return self

    def transform(self,
                  measurements: Iterable[Measurement],
                  pairing_properties: Tuple[Mapping[str, str], Mapping[str, str]] = ({}, {}),
                  max_m_per_source: Optional[int] = None,
                  seed: Optional[int] = None) -> List[MeasurementPair]:
        if max_m_per_source:
            measurements = select_max_measurements_per_source(max_m_per_source, measurements, seed, pairing_properties)
        all_pairs = [MeasurementPair(*mp) for mp in itertools.combinations(measurements, 2)]
        all_pairs = [mp for mp in all_pairs if pair_complies_with_properties(mp, pairing_properties)]
        return all_pairs


class LeaveOneTwoOutPairing(BasePairing):
    """
    Specific pairing used to make leave one/two out possible.
    Should be provided the measurements of exactly one or two sources.
    On transform, returns all same source pairs for one source, different source pairs for two sources.
    """

    def fit(self, measurements: Iterable[Measurement]):
        return self

    def transform(self,
                  measurements: Iterable[Measurement],
                  pairing_properties: Tuple[Mapping[str, str], Mapping[str, str]] = ({}, {}),
                  max_m_per_source: Optional[int] = None,
                  seed: Optional[int] = None) -> List[MeasurementPair]:
        # all same source pairs for one source, different source pairs for two sources
        num_sources = len(set(m.source.id for m in measurements))
        if num_sources == 1:
            return CartesianPairing().transform(
                measurements,
                pairing_properties=pairing_properties,
                seed=seed)
        if num_sources == 2:
            pairs = CartesianPairing().transform(
                measurements,
                pairing_properties=pairing_properties,
                seed=seed)
            return [pair for pair in pairs if not pair.is_same_source]
        raise ValueError(f'When pairing and leave one out, there should be 1 or 2'
                         f'sources. Found {num_sources}.')


class BalancedPairing(BasePairing):
    """
    Creates pairs of instances, with an equal amount of same source and different source pairs.
    """

    def fit(self, measurements: Iterable[Measurement]):
        return self

    def transform(self,
                  measurements: Iterable[Measurement],
                  pairing_properties: Tuple[Mapping[str, str], Mapping[str, str]] = ({}, {}),
                  max_m_per_source: Optional[int] = None,
                  seed: Optional[int] = None) -> List[MeasurementPair]:
        random.seed(seed)
        if max_m_per_source:
            measurements = select_max_measurements_per_source(max_m_per_source, measurements, pairing_properties)
        all_pairs = [MeasurementPair(*mp) for mp in itertools.combinations(measurements, 2)]
        all_pairs = [mp for mp in all_pairs if pair_complies_with_properties(mp, pairing_properties)]
        same_source_pairs = [a for a in all_pairs if a.is_same_source]
        different_source_pairs = [a for a in all_pairs if not a.is_same_source]
        n_pairs = min(len(same_source_pairs), len(different_source_pairs))
        same_source_pairs = random.sample(same_source_pairs, k=n_pairs)
        different_source_pairs = random.sample(different_source_pairs, k=n_pairs)
        selected_pairs = same_source_pairs + different_source_pairs
        random.shuffle(selected_pairs)
        return selected_pairs
