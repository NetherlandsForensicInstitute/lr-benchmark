import itertools
from typing import List

import confidence
import numpy as np
import pytest

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.data.dataset import MeasurementsDataset, MeasurementPairsDataset, GlassDataset, Dataset


@pytest.fixture
def measurements() -> List[Measurement]:
    values = np.reshape(np.array(list(range(50))), (10, 5))
    items = np.array(list(range(10, 60)))
    return [Measurement(source=Source(id=item, extra={}), extra={}, value=value) for value, item in zip(values, items)]


@pytest.fixture
def measurements_set2() -> List[Measurement]:
    values = np.reshape(np.array(list(range(250, 300))), (10, 5))
    items = np.array(list(range(10, 60)))
    return [Measurement(source=Source(id=item, extra={}), extra={}, value=value) for value, item in zip(values, items)]


@pytest.fixture
def measurement_pairs(measurements, measurements_set2) -> List[MeasurementPair]:
    return [MeasurementPair(
        measurement_a=m1,
        measurement_b=m2,
        extra={'score': 0.8}) for m1, m2 in zip(measurements, measurements_set2)]


@pytest.mark.parametrize('train_size, test_size', [(2, 3), (0.5, 0.2), (4, None), (None, 4), (None, None)])
def test_get_splits_measurements(measurements, train_size, test_size):
    dataset = MeasurementsDataset(measurements=measurements)
    for dataset_train, dataset_test in dataset.get_splits(seed=0,
                                                          train_size=train_size, validate_size=test_size, n_splits=3):
        X_train, y_train = dataset_train.get_x(), dataset_train.get_y()
        X_test, y_test = dataset_test.get_x(), dataset_test.get_y()
        assert len(np.intersect1d(X_train, X_test)) == 0
        assert len(np.intersect1d(y_train, y_test)) == 0

        if train_size is None and test_size is None:
            assert len(X_train) + len(X_test) == len(dataset.measurements)
            assert len(y_train) + len(y_test) == len(dataset.measurements)

        assert len(y_train) == train_size if isinstance(train_size, int) else \
            len(y_train) == train_size * len(measurements) if isinstance(train_size, float) else \
                len(y_train) == len(measurements) - test_size if isinstance(test_size, int) else \
                    len(y_train) == test_size * len(measurements) if isinstance(test_size, float) else \
                        len(measurements) > len(y_train) > 0

        assert len(y_test) == test_size if isinstance(test_size, int) else \
            len(y_test) == test_size * len(measurements) if isinstance(test_size, float) else \
                len(y_test) == len(measurements) - train_size if isinstance(train_size, int) else \
                    len(y_test) == train_size * len(measurements) if isinstance(train_size, float) else \
                        len(measurements) > len(y_test) > 0

        train_sources = [m.source.id for m in dataset_train.measurements]
        test_sources = [m.source.id for m in dataset_test.measurements]
        assert not any([train_source in test_sources for train_source in train_sources])


def test_get_splits_measurement_pairs(measurement_pairs):
    dataset = MeasurementPairsDataset(measurement_pairs=measurement_pairs)
    for dataset_train, dataset_test in dataset.get_splits(seed=0, n_splits=3):
        train_measurements = list(
            itertools.chain.from_iterable([mp.measurements for mp in dataset_train.measurement_pairs]))
        test_measurements = list(
            itertools.chain.from_iterable([mp.measurements for mp in dataset_test.measurement_pairs]))

        assert not any([train_measurement in test_measurements for train_measurement in train_measurements])
        train_sources = list(
            itertools.chain.from_iterable([mp.source_ids for mp in dataset_train.measurement_pairs]))
        test_sources = list(
            itertools.chain.from_iterable([mp.source_ids for mp in dataset_test.measurement_pairs]))
        assert not any([train_source in test_sources for train_source in train_sources])


@pytest.mark.parametrize("class_name, config_key", [  # (ASRDataset, 'asr', True),
    (GlassDataset, 'glass'),])
def test_dataset_basic_functions(class_name, config_key):
    config = confidence.load_name('tests/lrbenchmark_test')
    if config_key in config.dataset_test:
        dataset = class_name(**config.dataset_test[config_key])
    else:
        dataset = class_name()

    dataset.load()

    sets = dataset.get_splits()

    for set in sets:
        for fold in set:
            assert isinstance(fold, Dataset)
