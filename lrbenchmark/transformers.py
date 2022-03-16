import numpy as np
import sklearn
from lir.transformers import AbsDiffTransformer, InstancePairing
from scipy.stats import rankdata

from lrbenchmark.typing import XYType


class DummyTransformer(sklearn.base.TransformerMixin):
    """
    Simply returns the incoming data
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class RankTransformer(sklearn.base.TransformerMixin):
    """
    Assign ranks to X, dealing with ties appropriately.
    Expects:
        - X is of shape (n,f) with n=number of instances; f=number of features;
    Returns:
        - X has shape (n, f), and y
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert len(X.shape) == 2
        return rankdata(X, axis=0)


def pair_absdiff_transform(X: np.ndarray, y: np.ndarray, seed:None) -> XYType:
    """
    Transforms a basic X y dataset into same source and different source pairs and returns
    an X y dataset where the X is the absolute difference between the two pairs.

    Note that this method is different from sklearn TransformerMixin because it also transforms y.
    """
    pairing = InstancePairing(different_source_limit='balanced', seed=seed)
    transformer = AbsDiffTransformer()
    X_pairs, y_pairs = pairing.transform(X, y)
    return transformer.transform(X_pairs), y_pairs
