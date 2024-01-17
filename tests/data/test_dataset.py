import itertools
from typing import List

import numpy as np
import pytest

from lrbenchmark.data.models import Measurement, Source, MeasurementPair
from lrbenchmark.data.dataset import CommonSourceKFoldDataset


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

@pytest.fixture
def dataset(measurement_pairs):
    return CommonSourceKFoldDataset(n_splits=3, measurement_pairs=measurement_pairs)


def test_get_refnorm_split(dataset, refnorm_size=5):
    dataset, refnorm_dataset = dataset.get_refnorm_split(refnorm_size, seed=1)
    source_ids_dataset, source_ids_refnorm = dataset.source_ids, refnorm_dataset.source_ids
    difference = source_ids_dataset.difference(source_ids_refnorm)
    assert len(difference) == refnorm_size
    for mp in refnorm_dataset.measurement_pairs:
        source_ids = mp.source_ids
        assert sum(source_id in source_ids_dataset for source_id in source_ids) == 1


def test_select_refnorm_measurement_pairs(dataset):
    dataset, refnorm_dataset = dataset.get_refnorm_split(5, seed=1)
    source_ids_dataset = dataset.source_ids
    source_ids_to_exclude = list(source_ids_dataset)[:2]
    for mp in dataset.measurement_pairs:
        m_a, m_b = mp.measurement_a, mp.measurement_b
        refnorm_pairs = dataset.select_refnorm_measurement_pairs(m_a, source_ids_to_exclude, refnorm_dataset)
        for mp in refnorm_pairs:
            source_ids = mp.source_ids
            assert sum(source_id in source_ids_dataset for source_id in source_ids) == 1
            assert sum(source_id not in source_ids_to_exclude for source_id in source_ids) <= 1



@pytest.mark.parametrize('stratified', [True, False])
@pytest.mark.parametrize('group_by_source', [True, False])
@pytest.mark.parametrize('train_size, test_size', [(2, 3), (0.5, 0.2), (4, None), (None, 4), (None, None)])
def test_get_splits_measurements(measurements, group_by_source, stratified, train_size, test_size):
    dataset = CommonSourceKFoldDataset(n_splits=3, measurements=measurements)
    if stratified:
        with pytest.raises(ValueError):
            list(dataset.get_splits(seed=0, group_by_source=group_by_source, stratified=stratified))
    for dataset_train, dataset_test in dataset.get_splits(seed=0, group_by_source=group_by_source, train_size=train_size, test_size=test_size):
        X_train, y_train = dataset_train.get_x_y_measurement()
        X_test, y_test = dataset_test.get_x_y_measurement()
        assert len(np.intersect1d(X_train, X_test)) == 0
        assert len(np.intersect1d(y_train, y_test)) == 0

        if train_size is None and test_size is None:
            assert len(X_train) + len(X_test) == len(measurements)
            assert len(y_train) + len(y_test) == len(measurements)

        assert len(y_train) == train_size if isinstance(train_size, int) else \
               len(y_train) == train_size * len(measurements) if isinstance(train_size, float) else \
               len(y_train) == len(measurements)-test_size if isinstance(test_size, int) else \
               len(y_train) == test_size * len(measurements) if isinstance(test_size, float) else \
               len(measurements) > len(y_train) > 0

        assert len(y_test) == test_size if isinstance(test_size, int) else \
               len(y_test) == test_size * len(measurements) if isinstance(test_size, float) else \
               len(y_test) == len(measurements) - train_size if isinstance(train_size, int) else \
               len(y_test) == train_size * len(measurements) if isinstance(train_size, float) else \
               len(measurements) > len(y_test) > 0

        if group_by_source:
            train_sources = [m.source.id for m in dataset_train.measurements]
            test_sources = [m.source.id for m in dataset_test.measurements]
            assert not any([train_source in test_sources for train_source in train_sources])


@pytest.mark.parametrize('group_by_source', [True, False])
@pytest.mark.parametrize('stratified', [True, False])
def test_get_splits_measurement_pairs(dataset, group_by_source, stratified):
    if stratified and group_by_source:
        with pytest.raises(ValueError):
            list(dataset.get_splits(seed=0, group_by_source=group_by_source, stratified=stratified))
    else:
        for dataset_train, dataset_test in dataset.get_splits(seed=0, group_by_source=group_by_source, stratified=stratified):
            train_measurements = list(
                itertools.chain.from_iterable([mp.measurements for mp in dataset_train.measurement_pairs]))
            test_measurements = list(
                itertools.chain.from_iterable([mp.measurements for mp in dataset_test.measurement_pairs]))

            assert not any([train_measurement in test_measurements for train_measurement in train_measurements])

            if group_by_source:
                train_sources = list(itertools.chain.from_iterable([mp.source_ids for mp in dataset_train.measurement_pairs]))
                test_sources = list(itertools.chain.from_iterable([mp.source_ids for mp in dataset_test.measurement_pairs]))
                assert not any([train_source in test_sources for train_source in train_sources])


