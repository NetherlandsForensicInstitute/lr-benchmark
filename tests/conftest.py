import os
from pathlib import Path

from typing import List

import numpy as np
import pytest

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import Measurement, Source, MeasurementPair


ROOT_DIR = Path(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
TEST_DIR = ROOT_DIR / 'tests'

@pytest.fixture
def test_measurement() -> Measurement:
    return Measurement(source=Source(id=1, extra={}), extra={}, value=np.array([1, 1, 1, 1]), id='id_test')


@pytest.fixture
def test_measurement_pair(test_measurement) -> MeasurementPair:
    return MeasurementPair(measurement_a=test_measurement,
                           measurement_b=Measurement(source=Source(id=2, extra={}),
                                                     extra={},
                                                     value=np.array([0, 0, 0, 0]), id='id_b'),
                           extra={'score': 0.8})


@pytest.fixture
def measurements() -> List[Measurement]:
    values = np.reshape(np.array(list(range(50))), (10, 5))
    items = np.array(list(range(10, 20)))
    return [Measurement(source=Source(id=item, extra={}),
                        id=i,
                        extra={'property': 'trace'}, value=value) for i, (value, item) in enumerate(zip(values, items))]


@pytest.fixture
def measurements_set2() -> List[Measurement]:
    values = np.reshape(np.array(list(range(250, 300))), (10, 5))
    items = np.array(list(range(10, 15))+list(range(21, 26)))
    return [Measurement(source=Source(id=item, extra={}), extra={'property': 'reference'},
                        value=value, id=i+9999) for i, (value, item) in enumerate(zip(values, items))]


@pytest.fixture
def dataset(measurements):
    return Dataset(measurements=measurements)
