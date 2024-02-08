from abc import ABC, abstractmethod
from typing import List

import numpy as np

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import Measurement, Source, Sample


class MeasurementPairsSimulator(ABC):
    @abstractmethod
    def get_measurements(self, **kwargs) -> Dataset:
        raise NotImplementedError


class NormalPairsSimulator(MeasurementPairsSimulator):
    def __init__(self, population_mean: float, population_std: float, source_std: float, seed: int = None):
        super().__init__()
        self.population_mean = population_mean
        self.population_std = population_std
        self.source_std = source_std
        self.generator = np.random.default_rng(seed=seed)

    def get_measurements(self, n_sources: int, n_measurements_per_source: int) -> List[Measurement]:
        """
        Generates measurements with the values of the same source measurements differing by the measurement error, and
        the values of the different source measurements drawn randomly from the distribution.

        :param n_sources: number of sources to generate
        :param n_measurements_per_source: number of measurements per source to generate
        :return: list of all generated measurements
        """
        n_dimensions = 3
        n_measurements = n_measurements_per_source * n_sources
        source_means = self.generator.normal(self.population_mean, self.population_std, (n_sources, n_dimensions))
        measurements = []
        for i, mean in enumerate(source_means):
            source = Source(id=i, extra={})
            measurement_errors = self.generator.normal(0, self.source_std, (n_measurements, n_dimensions))
            for j, error in enumerate(measurement_errors):
                measurements.append(Measurement(source=source, sample=Sample(id=j),
                                                value=mean+error, extra={}, id=1))
        return measurements

    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.population_mean}, sigma={self.population_std}, "
                f"trace_measurement_stdev={self.source_std})")


class SynthesizedNormalDataset(Dataset):
    def __init__(self, population_mean: float,
                 population_std: float,
                 source_std: float,
                 n_sources: int,
                 n_measurements_per_source: int, seed: int):
        super().__init__()
        self.simulator = NormalPairsSimulator(population_mean=population_mean,
                                              population_std=population_std,
                                              source_std=source_std,
                                              seed=seed)
        self.measurements = self.simulator.get_measurements(n_sources, n_measurements_per_source)
