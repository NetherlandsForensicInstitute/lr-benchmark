from abc import ABC, abstractmethod
from typing import List

import numpy as np

from lrbenchmark.data.dataset import MeasurementPairsDataset
from lrbenchmark.data.models import Measurement, MeasurementPair, Source


class MeasurementPairsSimulator(ABC):
    @abstractmethod
    def get_pairs(self, **kwargs) -> MeasurementPairsDataset:
        raise NotImplementedError


class NormalPairsSimulator(MeasurementPairsSimulator):
    def __init__(self, mean: float, sigma: float, trace_measurement_stdev: float, seed:int=None):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.trace_measurement_stdev = trace_measurement_stdev
        self.generator = np.random.default_rng(seed=seed)

    def get_pairs(self, n_same_source: int, n_diff_source: int) -> List[MeasurementPair]:
        """
        Generates pairs of measurements with the values of the same source pairs differing by the measurement error, and
        the values of the different source pairs drawn randomly from the distribution. Returns a tuple of a list of same
        source and a list of different source measurement pairs

        :param n_same_source: number of same source pairs to generate
        :param n_diff_source: number of different source pairs to generate
        :return: tuple of same source and different source measurement pairs
        """
        real_value = self.generator.normal(self.mean, self.sigma, n_same_source)
        other_value = self.generator.normal(self.mean, self.sigma, n_diff_source)
        measurement_error = self.generator.normal(0, self.trace_measurement_stdev, n_same_source)
        measured_value = real_value + measurement_error
        return [
                   MeasurementPair(
                       measurement_a=Measurement(source=Source(id=i, extra={}), value=real_value[i], extra={}),
                       measurement_b=Measurement(source=Source(id=i, extra={}), value=measured_value[i], extra={}),
                       extra={}) for i in range(n_same_source)] + [
                   MeasurementPair(
                       measurement_a=Measurement(source=Source(id=n_same_source + i, extra={}),
                                                 value=other_value[i], extra={}),
                       measurement_b=Measurement(source=Source(id=i, extra={}),
                                                 value=real_value[i], extra={}),
                       extra={}) for i in range(min(n_diff_source, n_same_source))]

    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma}, "
                f"trace_measurement_stdev={self.trace_measurement_stdev})")


class SynthesizedNormalDataset(MeasurementPairsDataset):
    def __init__(self, mean: float, sigma: float, trace_measurement_stdev: float, n_same_source: int,
                 n_diff_source: int, seed: int):
        super().__init__()
        self.simulator = NormalPairsSimulator(mean=mean,
                                              sigma=sigma,
                                              trace_measurement_stdev=trace_measurement_stdev,
                                              seed=seed)
        self.measurement_pairs = self.simulator.get_pairs(n_same_source, n_diff_source)
