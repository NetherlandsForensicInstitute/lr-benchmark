import itertools
from typing import List

import sklearn.base

from lrbenchmark.data.models import Measurement, MeasurementPair


class CartesianPairing(sklearn.base.TransformerMixin):
    def __init__(self,
                 seed=None):
        """
        Creates pairs of instances.

        This transformer takes a list of Measurement as input, and returns a list of MeasurementPair.
        By default, the list of MeasurementPair contains all possible pairs of Measurement.

        Parameters:
            - seed (int or None): seed to make pairing reproducible
        """

    def fit(self, X, y=None):
        return self

    def transform(self, measurements: List[Measurement]):
        """

        """
        return [MeasurementPair(*mp) for mp in itertools.combinations(measurements, 2)]