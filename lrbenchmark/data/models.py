from dataclasses import dataclass
from typing import Mapping, Any, Optional

import numpy as np


@dataclass
class Source:
    """
    A source that can generate a measurement

    :param id: the id of the source
    :param extra: Additional metadata related to the source.
    """
    id: int
    extra: Mapping[str, Any]


@dataclass
class Measurement:
    """
    A single measurement that has a source, with optional measurement_value and additional meta information in the
    `extra` mapping

    :param source: the source of the measurement
    :param extra: Additional metadata related to the measurement
    :param value: The value of the measurement
    """
    source: Source
    extra: Mapping[str, Any]
    value: Any = None

    def get_x(self) -> np.ndarray:
        if not isinstance(self.value, np.ndarray):
            raise TypeError('The returned value should be an numpy array')
        return self.value


@dataclass
class MeasurementPair:
    """
    A pair of two measurements. It always contains the information from the two measurements it was created from. An
    optional score can be included if already available
    """
    measurement_a: Measurement
    measurement_b: Measurement
    score: Optional[float, np.ndarray] = None

    @property
    def is_same_source(self) -> bool:
        return self.measurement_a.source.id == self.measurement_b.source.id

    def get_x(self) -> np.ndarray:
        return np.ndarray(self.score) if not isinstance(self.score, np.ndarray) else self.score

    def get_y(self) -> bool:
        return self.is_same_source
