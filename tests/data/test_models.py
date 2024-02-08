import numpy as np

from lrbenchmark.data.models import Source, Sample, Measurement


def test_get_x_measurement(test_measurement):
    assert np.array_equal(test_measurement.get_x(), np.array([1, 1, 1, 1]))


def test_get_y_measurement(test_measurement):
    assert test_measurement.get_y() == 1


def test_is_same_source(test_measurement_pair):
    assert not test_measurement_pair.is_same_source


def test_get_x_measurement_pair(test_measurement_pair):
    assert np.array_equal(test_measurement_pair.get_x(), np.array([[1, 0], [1, 0], [1, 0], [1, 0]]))


def test_get_y_measurement_pair(test_measurement_pair):
    assert not test_measurement_pair.get_y()


def test_measurements_equal(test_measurement):
    other_measurement = Measurement(source=Source(id=1, extra={'test': True}), extra={'test': True},
                                    value=np.array([2, 2, 2, 2]), sample=Sample(id='A1'), id='A1_test')
    assert test_measurement == other_measurement
    other_measurement = Measurement(source=Source(id=1, extra={'test': True}), extra={'test': True},
                                    value=np.array([2, 2, 2, 2]), sample=Sample(id='A2'), id='A1_test')
    assert not test_measurement == other_measurement