import numpy as np
import pytest

from lrbenchmark.data.models import Measurement, MeasurementPair, Source


@pytest.fixture
def test_measurement() -> Measurement:
    return Measurement(source=Source(id=1, extra={}),
                       extra={},
                       value=np.array([1, 1, 1, 1]))


@pytest.fixture
def test_measurement_pair(test_measurement) -> MeasurementPair:
    return MeasurementPair(
        measurement_a=test_measurement,
        measurement_b=Measurement(source=Source(id=2, extra={}),
                                  extra={},
                                  value=np.array([0, 0, 0, 0])),
        score=0.8)


def test_get_x_measurement(test_measurement):
    assert np.array_equal(test_measurement.get_x(), np.array([1, 1, 1, 1]))


def test_is_same_source(test_measurement_pair):
    assert not test_measurement_pair.is_same_source


def test_get_x_measurement_pair(test_measurement_pair):
    assert np.array_equal(test_measurement_pair.get_x(), np.array([0.8]))


def test_get_y_measurement_pair(test_measurement_pair):
    assert not test_measurement_pair.get_y()
