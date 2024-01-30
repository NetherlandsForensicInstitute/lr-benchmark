from dataclasses import dataclass
from typing import Mapping, Any, Union, Optional, List, Tuple

import numpy as np


@dataclass
class Source:
    """
    A source that can generate a measurement

    :param id: the identifier of the source
    :param extra: additional metadata related to the source
    """
    id: Union[int, str]
    extra: Mapping[str, Any]


@dataclass
class Measurement:
    """
    A single measurement that has a source, with optional value and additional
    meta information in the `extra` mapping

    :param source: the source of the measurement
    :param extra: additional metadata related to the measurement
    :param is_like_reference: indication of whether the measurement is similar to reference measurements
    :param is_like_trace: indication of whether the measurement is similar to trace measurements
    :param value: the value of the measurement
    """
    source: Source
    extra: Mapping[str, Any]
    is_like_reference: Optional[bool] = None
    is_like_trace: Optional[bool] = None
    value: Optional[Any] = None

    def get_x(self) -> np.ndarray:
        if not isinstance(self.value, np.ndarray):
            raise TypeError('The returned value should be a numpy array.')
        return self.value

    def get_y(self) -> int:
        return self.source.id

    def __str__(self):
        return f'source_id: {self.source.id}, {self.extra}'


@dataclass
class MeasurementPair:
    """
    A pair of two measurements. It always contains the information from the two
    measurements it was created from and additional meta-information in the
    extra mapping.
    """
    measurement_a: Measurement
    measurement_b: Measurement
    extra: Mapping[str, Any] = None

    @property
    def is_same_source(self) -> bool:
        return self.measurement_a.source.id == self.measurement_b.source.id

    @property
    def source_ids(self) -> List[Union[int, str]]:
        return [self.measurement_a.source.id, self.measurement_b.source.id]

    @property
    def measurements(self) -> Tuple[Measurement, Measurement]:
        return self.measurement_a, self.measurement_b

    def get_x(self) -> np.ndarray:
        if self.measurement_a.get_x() is None or self.measurement_b.get_x() is None:
            raise ValueError('No values present in MeasurementPair. Use PrecalculatedScorer to retrieve scores.')
        return np.vstack([self.measurement_a.get_x(), self.measurement_b.get_x()]).T

    def get_y(self) -> bool:
        return self.is_same_source

    def __str__(self):
        return f'[{self.measurement_a} - {self.measurement_b}]'
