import itertools
from typing import List

import confidence
import numpy as np
import pytest

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.data.dataset import MeasurementsDataset, MeasurementPairsDataset, GlassDataset, Dataset
from lrbenchmark.data.simulation import SynthesizedNormalDataset


@pytest.fixture
def measurements() -> List[Measurement]:
    values = np.reshape(np.array(list(range(50))), (10, 5))
    items = np.array(list(range(10, 20)))
    return [Measurement(source=Source(id=item, extra={}), extra={}, value=value) for value, item in zip(values, items)]


@pytest.fixture
def measurements_set2() -> List[Measurement]:
    values = np.reshape(np.array(list(range(250, 300))), (10, 5))
    items = np.array(list(range(10, 15))+list(range(21, 26)))
    return [Measurement(source=Source(id=item, extra={}), extra={}, value=value) for value, item in zip(values, items)]


@pytest.fixture
def measurement_pairs(measurements, measurements_set2) -> List[MeasurementPair]:
    return [MeasurementPair(measurement_a=m1,
                            measurement_b=m2,
                            extra={'score': 0.8}) for m1, m2 in itertools.combinations(measurements +
                                                                                       measurements_set2, 2)]


@pytest.fixture
def dataset(measurement_pairs):
    return MeasurementPairsDataset(measurement_pairs=measurement_pairs)


def test_get_refnorm_split(dataset: MeasurementPairsDataset):
    refnorm_size = 5
    dataset, refnorm_dataset = dataset.get_refnorm_split(refnorm_size=refnorm_size, seed=1)
    source_ids_dataset, source_ids_refnorm = dataset.source_ids, refnorm_dataset.source_ids
    difference = source_ids_refnorm.difference(source_ids_dataset)
    assert len(difference) == refnorm_size
    for mp in refnorm_dataset.measurement_pairs:
        source_ids = mp.source_ids
        assert sum(source_id in source_ids_dataset for source_id in source_ids) == 1


def test_select_refnorm_measurement_pairs(dataset: MeasurementPairsDataset):
    dataset, refnorm_dataset = dataset.get_refnorm_split(5, seed=1)
    source_ids_dataset = dataset.source_ids
    source_ids_to_exclude = list(source_ids_dataset)[:2]
    for mp in dataset.measurement_pairs:
        refnorm_pairs = dataset.select_refnorm_measurement_pairs(mp.measurement_a,
                                                                 source_ids_to_exclude,
                                                                 refnorm_dataset)
        assert all(sum(source_id in source_ids_dataset for source_id in mp.source_ids) == 1 for mp in refnorm_pairs)
        assert all(sum(source_id in source_ids_to_exclude for source_id in mp.source_ids) <= 1 for mp in refnorm_pairs)


def test_select_refnorm_measurement_pairs_leave_two_out(dataset: MeasurementPairsDataset):
    dataset, refnorm_dataset = dataset.get_refnorm_split(refnorm_size=None, seed=1)
    dataset_train, dataset_test = next(dataset.get_splits(seed=0))

    for mp in dataset.measurement_pairs:
        source_ids_to_exclude = list(dataset_train.source_ids) + [mp.measurement_a.source.id,
                                                                  mp.measurement_b.source.id]
        refnorm_pairs = dataset.select_refnorm_measurement_pairs(
            mp.measurement_a,
            source_ids_to_exclude=list(dataset_train.source_ids) + [mp.measurement_a.source.id,
                                                                    mp.measurement_b.source.id],
            refnorm_dataset=dataset)
        assert all(sum(source_id in mp.source_ids for source_id in rn_mp.source_ids) == 1 for rn_mp in refnorm_pairs)
        assert all(sum(source_id in source_ids_to_exclude for source_id in rn_mp.source_ids) <= 1 for rn_mp in
                   refnorm_pairs)


@pytest.mark.parametrize('train_size, test_size', [(2, 3), (0.5, 0.2), (4, None), (None, 4), (None, None)])
def test_get_splits_measurements(measurements, train_size, test_size):
    dataset = MeasurementsDataset(measurements=measurements)
    for dataset_train, dataset_test in dataset.get_splits(seed=0, train_size=train_size, validate_size=test_size):
        X_train, y_train = dataset_train.get_x(), dataset_train.get_y()
        X_test, y_test = dataset_test.get_x(), dataset_test.get_y()
        assert len(np.intersect1d(X_train, X_test)) == 0
        assert len(np.intersect1d(y_train, y_test)) == 0

        if train_size is None and test_size is None:
            assert len(X_train) + len(X_test) == len(measurements)
            assert len(y_train) + len(y_test) == len(measurements)

        assert len(y_train) == train_size if isinstance(train_size, int) else len(y_train) == train_size * len(
            measurements) if isinstance(train_size, float) else len(y_train) == len(
            measurements) - test_size if isinstance(test_size, int) else len(y_train) == test_size * len(
            measurements) if isinstance(test_size, float) else len(measurements) > len(y_train) > 0

        assert len(y_test) == test_size if isinstance(test_size, int) else len(y_test) == test_size * len(
            measurements) if isinstance(test_size, float) else len(y_test) == len(
            measurements) - train_size if isinstance(train_size, int) else len(y_test) == train_size * len(
            measurements) if isinstance(train_size, float) else len(measurements) > len(y_test) > 0

        assert not dataset_train.source_ids.intersection(dataset_test.source_ids)


def test_get_splits_measurement_pairs(dataset):
    for dataset_train, dataset_test in dataset.get_splits(seed=0):
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
    (GlassDataset, 'glass'),
    (SynthesizedNormalDataset, 'normal')])
def test_dataset_basic_functions(class_name, config_key):
    config = confidence.load_name('tests/lrbenchmark_test')
    if config_key in config.dataset_test:
        dataset = class_name(**config.dataset_test[config_key])
    else:
        dataset = class_name()

    sets = dataset.get_splits()

    for set in sets:
        for fold in set:
            assert isinstance(fold, Dataset)
