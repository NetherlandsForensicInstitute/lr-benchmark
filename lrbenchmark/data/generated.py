from typing import Iterable, Tuple, List

import numpy as np

from lrbenchmark.data.dataset import Dataset, CommonSourceMeasurementPairsDataset
from lrbenchmark.data.models import Measurement, MeasurementPair, Source


class SynthesizedNormalDataset(Dataset):
    def __init__(self, mean: float, sigma: float, trace_measurement_stdev: float, n_train_instances: int,
                 n_test_instances: int):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.trace_measurement_stdev = trace_measurement_stdev
        self.n_train_instances = n_train_instances
        self.n_test_instances = n_test_instances

    def get_pairs(self, n_same_source: int, n_diff_source: int) -> Tuple[List[MeasurementPair], List[MeasurementPair]]:
        """
        Generates pairs of measurements with the values of the same source pairs differing by the measurement error, and
        the values of the different source pairs drawn randomly from the distribution. Returns a tuple of a list of same
        source and a list of different source measurement pairs

        :param n_same_source: number of same source pairs to generate
        :param n_diff_source: number of different source pairs to generate
        :return: tuple of same source and different source measurement pairs
        """
        real_value = np.random.normal(self.mean, self.sigma, n_same_source)
        other_value = np.random.normal(self.mean, self.sigma, n_diff_source)
        measurement_error = np.random.normal(0, self.trace_measurement_stdev, n_same_source)
        measured_value = real_value + measurement_error
        return ([
                    MeasurementPair(
                        measurement_a=Measurement(source=Source(id=i, extra={}), value=real_value[i], extra={}),
                        measurement_b=Measurement(source=Source(id=i, extra={}), value=measured_value[i], extra={}),
                        extra={}) for i in range(n_same_source)],
                [
                    MeasurementPair(
                        measurement_a=Measurement(source=Source(id=n_same_source + i, extra={}),
                                                  value=other_value[i], extra={}),
                        measurement_b=Measurement(source=Source(id=i, extra={}),
                                                  value=real_value[i], extra={}),
                        extra={}) for i in range(min(n_diff_source, n_same_source))])

    def generate_data(self, n: int) -> Dataset:
        ss_pairs, ds_pairs = self.get_pairs(n // 2, n // 2)
        pairs = ss_pairs + ds_pairs
        return CommonSourceMeasurementPairsDataset(measurement_pairs=pairs)

    def get_splits(self, seed: int = None, **kwargs) -> Iterable[Dataset]:
        yield [self.generate_data(self.n_train_instances), self.generate_data(self.n_test_instances)]

    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma}, "
                f"trace_measurement_stdev={self.trace_measurement_stdev})")
