from dataclasses import dataclass
from typing import Mapping, Any, Union, Optional

import numpy as np


@dataclass
class Source:
    """
    A source that can generate a measurement

    :param id: the id of the source
    :param extra: additional metadata related to the source
    """
    id: int
    extra: Mapping[str, Any]


@dataclass
class Measurement:
    """
    A single measurement that has a source, with optional value and additional
    meta information in the `extra` mapping

    :param source: the source of the measurement
    :param extra: additional metadata related to the measurement
    :param value: the value of the measurement
    """
    source: Source
    extra: Mapping[str, Any]
    value: Optional[Any] = None

    def get_x(self) -> np.ndarray:
        if not isinstance(self.value, np.ndarray):
            raise TypeError('The returned value should be a numpy array.')
        return self.value

    def get_y(self) -> int:
        return self.source.id


@dataclass
class MeasurementPair:
    """
    A pair of two measurements. It always contains the information from the two
    measurements it was created from and additional meta-information in the
    extra mapping.
    """
    measurement_a: Measurement
    measurement_b: Measurement
    extra: Mapping[str, Any]

    @property
    def is_same_source(self) -> bool:
        return self.measurement_a.source.id == self.measurement_b.source.id

    @property
    def score(self) -> Union[None, float, np.ndarray]:
        if 'score' in self.extra.keys():
            return self.extra['score']
        raise ValueError("No score found in the extra mapping.")

    def get_x(self) -> np.ndarray:
        if 'score' in self.extra.keys():
            return np.array([self.score]) if not isinstance(self.score, np.ndarray) else self.score
        else:
            # if score is not available, return the values of the individual measurements
            # the shape of the returned array should be (f,2), with f=number of features and 2 values (for the two
            # measurements), to be compatible with the transformation function
            measurement_a_value = np.array([self.measurement_a.value]) if not (
                isinstance(self.measurement_a.value, np.ndarray)) else self.measurement_a.value
            measurement_b_value = np.array([self.measurement_b.value]) if not (
                isinstance(self.measurement_b.value, np.ndarray)) else self.measurement_b.value
            return np.array(list(zip(measurement_a_value, measurement_b_value)))

    def get_y(self) -> bool:
        return self.is_same_source
