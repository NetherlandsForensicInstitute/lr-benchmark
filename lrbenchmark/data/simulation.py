from abc import ABC, abstractmethod
from typing import List

import numpy as np

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import Measurement, Source


class MeasurementPairsSimulator(ABC):
    @abstractmethod
    def get_measurements(self, **kwargs) -> Dataset:
        raise NotImplementedError


class NormalPairsSimulator(MeasurementPairsSimulator):
    def __init__(self, mean: float, sigma: float, trace_measurement_stdev: float, seed: int = None):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.trace_measurement_stdev = trace_measurement_stdev
        self.generator = np.random.default_rng(seed=seed)

    def get_measurements(self, n_same_source: int, n_diff_source: int) -> List[Measurement]:
        """
        Generates measurements with the values of the same source measurements differing by the measurement error, and
        the values of the different source measurements drawn randomly from the distribution.

        :param n_same_source: number of same source measurements to generate
        :param n_diff_source: number of different source measurements to generate
        :return: list of all generated measurements
        """
        real_value = self.generator.normal(self.mean, self.sigma, n_same_source)
        other_value = self.generator.normal(self.mean, self.sigma, n_diff_source)
        measurement_error = self.generator.normal(0, self.trace_measurement_stdev, n_same_source)
        measured_value = real_value + measurement_error
        measurements = []
        for i in range(n_same_source):
            measurements.append(Measurement(source=Source(id=i, extra={}), value=np.array(real_value[i]), extra={}))
            measurements.append(Measurement(source=Source(id=i, extra={}), value=np.array(measured_value[i]), extra={}))
        for i in range(min(n_same_source, n_diff_source)):
            measurements.append(Measurement(source=Source(id=n_same_source + i, extra={}),
                                            value=np.array(other_value[i]), extra={}))
        return measurements

    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma}, "
                f"trace_measurement_stdev={self.trace_measurement_stdev})")


class SynthesizedNormalDataset(Dataset):
    def __init__(self, mean: float, sigma: float, trace_measurement_stdev: float, n_same_source: int,
                 n_diff_source: int, seed: int):
        super().__init__()
        self.simulator = NormalPairsSimulator(mean=mean,
                                              sigma=sigma,
                                              trace_measurement_stdev=trace_measurement_stdev,
                                              seed=seed)
        self.measurements = self.simulator.get_measurements(n_same_source, n_diff_source)
