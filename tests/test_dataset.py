import numpy as np
import pytest

from lrbenchmark.dataset import InMemoryCommonSourceKFoldDataset


@pytest.fixture
def X() -> np.ndarray:
    return np.reshape(np.array(list(range(25))), (5, 5))


@pytest.fixture
def y() -> np.ndarray:
    return np.array(['a', 'b', 'c', 'd', 'e'])


def test_get_splits_is_mutually_exclusive(X, y):
    dataset = InMemoryCommonSourceKFoldDataset(X, y, n_splits=3)
    for (X_train, y_train), (X_test, y_test) in dataset.get_splits(seed=0):
        assert len(np.intersect1d(X_train, X_test)) == 0 and X_train.size + X_test.size == X.size
        assert len(np.intersect1d(y_train, y_test)) == 0 and y_train.size + y_test.size == y.size

