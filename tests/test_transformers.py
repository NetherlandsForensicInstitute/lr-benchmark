import numpy as np
import pytest

from lrbenchmark.transformers import DummyTransformer, RankTransformer, pair_absdiff_transform


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


def test_rank_transformer(X, y):
    X_expected = np.array([
        [3., 1.0, 2.],
        [3., 2.5, 3.],
        [3., 2.5, 4.],
        [3., 4.0, 5.],
        [3., 5.0, 1.]
    ])
    X_actual = RankTransformer().fit_transform(X, y)
    assert np.all(X_expected == X_actual)


def test_pair_absdiff_transform(X, y):
    X_transformed, y_transformed = pair_absdiff_transform(X, y)

    assert np.all(X_transformed >= 0)
    assert np.all((y_transformed >= 0) & (y_transformed <= 1))

    max_diff_expected = np.max(X, axis=0) - np.min(X, axis=0)
    max_diff_actual = np.max(X_transformed, axis=0)
    assert np.all(max_diff_actual <= max_diff_expected)


def test_pair_absdiff_transform_after_rank_transform(X, y):
    X = RankTransformer().fit_transform(X, y)
    test_pair_absdiff_transform(X, y)
