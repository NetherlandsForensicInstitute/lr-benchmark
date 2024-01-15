from typing import List

import numpy as np
import pytest

from lrbenchmark.data.models import Measurement, Source
from lrbenchmark.data.dataset import CommonSourceKFoldDataset

@pytest.fixture
def measurements() -> List[Measurement]:
    values = np.reshape(np.array(list(range(25))), (5, 5))
    items = np.array([10, 11, 12, 13, 14])
    return [Measurement(source=Source(id=item, extra={}),
                        extra={},
                        value=value) for value, item in zip(values, items)]

def test_get_splits_is_mutually_exclusive(measurements):
    dataset = CommonSourceKFoldDataset(n_splits=3, measurements=measurements)
    for dataset_train, dataset_test in dataset.get_splits(seed=0):
        X_train, y_train = dataset_train.get_x_y_measurement()
        X_test, y_test = dataset_test.get_x_y_measurement()
        assert len(np.intersect1d(X_train, X_test)) == 0 and len(X_train) + len(X_test) == len(dataset.measurements)
        assert len(np.intersect1d(y_train, y_test)) == 0 and len(y_train) + len(y_test) == len(dataset.measurements)




