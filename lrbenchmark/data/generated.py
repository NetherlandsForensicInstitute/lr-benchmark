from collections import namedtuple
from typing import Iterable, Tuple

import numpy as np
import scipy

from lrbenchmark.dataset import Dataset
from lrbenchmark.typing import XYType, TrainTestPair


class XYWithTrueLRs(namedtuple("XY", ["X", "y"])):
  def __new__(cls, X, y, true_lrs):
    self = super(XYWithTrueLRs, cls).__new__(cls, X, y)
    self.true_lrs = true_lrs
    return self


class SynthesizedNormalDataset(Dataset):
    def __init__(self, mean: float, sigma: float, trace_measurement_stdev: float, n_train_instances: int, n_test_instances: int):
        super().__init__()
        self.mean = mean
        self.sigma = sigma
        self.trace_measurement_stdev = trace_measurement_stdev
        self.n_train_instances = n_train_instances
        self.n_test_instances = n_test_instances

    def get_same_source_pairs(self, n: int) -> np.ndarray:
        real_height = np.random.normal(self.mean, self.sigma, n)
        measurement_error = np.random.normal(0, self.trace_measurement_stdev, n)
        measured_height = real_height + measurement_error
        return np.stack([real_height.reshape(n, 1), measured_height.reshape(n, 1)], axis=2)

    def get_different_source_pairs(self, n: int) -> np.ndarray:
        real_height = np.random.normal(self.mean, self.sigma, n)
        other_height = np.random.normal(self.mean, self.sigma, n)
        measurement_error = np.random.normal(0, self.trace_measurement_stdev, n)
        measured_height = other_height + measurement_error
        return np.stack([real_height.reshape(n, 1), measured_height.reshape(n, 1)], axis=2)

    def calculate_lr(self, pairs: np.ndarray):
        real_height = pairs[:, :, 0]
        measured_height = pairs[:, :, 1]
        diff = np.abs(real_height - measured_height)

        ss_prob = scipy.stats.norm(self.mean, self.trace_measurement_stdev).pdf(diff)
        ds_prob = np.array([scipy.stats.norm(real_height[i], self.sigma).pdf(measured_height[i]) for i in range(len(real_height))])

        return ss_prob / ds_prob

    def generate_data(self, n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        ss_pairs = self.get_same_source_pairs(n//2)
        ds_pairs = self.get_different_source_pairs(n//2)
        pairs = np.concatenate([ss_pairs, ds_pairs])
        labels = np.concatenate([np.ones(n//2), np.zeros(n//2)])
        real_lrs = self.calculate_lr(pairs)
        return XYWithTrueLRs(pairs, labels, real_lrs)

    def get_splits(self, seed: int = None) -> Iterable[TrainTestPair]:
        yield self.generate_data(self.n_train_instances), self.generate_data(self.n_test_instances)

    @property
    def is_binary(self) -> bool:
        return True

    def __repr__(self):
        return f"{self.__class__.__name__}(mean={self.mean}, sigma={self.sigma}, trace_measurement_stdev={self.trace_measurement_stdev})"
