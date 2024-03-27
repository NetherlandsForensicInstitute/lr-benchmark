import numpy as np
import pytest

from lrbenchmark.transformers import DummyTransformer


@pytest.fixture
def X() -> np.ndarray:
    return np.array([
        [1, 1, 11],
        [1, 2, 12],
        [1, 2, 13],
        [1, 3, 99],
        [1, 4, 10]
    ])


@pytest.fixture
def y() -> np.ndarray:
    return np.array(['a', 'a', 'b', 'b', 'c'])


def test_dummy_transformer(X, y):
    X_expected = X
    X_actual = DummyTransformer().fit_transform(X, y)
    assert np.all(X_expected == X_actual)
