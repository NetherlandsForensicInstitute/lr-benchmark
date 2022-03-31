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


def pair_absdiff_transform(X: np.ndarray,
                           y: np.ndarray,
                           ratio_limit: Optional[int] = None,
                           seed: Optional[int] = None) -> XYType:
    """
    Transforms a basic X y dataset into same source and different source pairs and returns
    an X y dataset where the X is the absolute difference between the two pairs.

    Note that this method is different from sklearn TransformerMixin because it also transforms y.
    """
    pairing = InstancePairing(ratio_limit=ratio_limit, seed=seed)
    transformer = AbsDiffTransformer()
    X_pairs, y_pairs = pairing.transform(X, y)
    return transformer.transform(X_pairs), y_pairs
