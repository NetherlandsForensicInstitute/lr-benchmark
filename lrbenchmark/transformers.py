from typing import Optional

import numpy as np
import sklearn
from lir.transformers import AbsDiffTransformer, InstancePairing

from lrbenchmark.typing import XYType


class DummyTransformer(sklearn.base.TransformerMixin):
    """
    Simply returns the incoming data
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X


class DummyClassifier(sklearn.base.TransformerMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        if len(X.shape) > 1 and X.shape[-1] != 1:
            raise ValueError(f"The dummy classifier can only be used on scores, which have one score per pair, not on "
                             f"data with shape {X.shape}.")
        return X


def pair_absdiff_transform(X: np.ndarray, y: np.ndarray, seed: Optional[int] = None) -> XYType:
    """
    Transforms a basic X y dataset into same source and different source pairs and returns
    an X y dataset where the X is the absolute difference between the two pairs.

    Note that this method is different from sklearn TransformerMixin because it also transforms y.
    """
    pairing = InstancePairing(different_source_limit='balanced', seed=seed)
    transformer = AbsDiffTransformer()
    X_pairs, y_pairs = pairing.transform(X, y)
    return transformer.transform(X_pairs), y_pairs
