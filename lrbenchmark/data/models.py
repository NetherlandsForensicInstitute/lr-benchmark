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

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Sample:
    """
    A sample from one or more sources that could encompass one or multiple measurements. For example, for ASR
    a sample could be a single recording.
    :param id: the identifier of the sample
    :param extra: additional metadata related to the sample
    """

    id: Union[int, str]
    extra: Mapping[str, Any] = None

    def __eq__(self, other):
        return self.id == other.id


@dataclass
class Measurement:
    """
    A single measurement that has a source, with optional value and additional
    meta information in the `extra` mapping

    :param source: the source of the measurement
    :param extra: additional metadata related to the measurement
    :param id: the identifier of the measurement
    :param sample: the original sample the measurement belongs to
    :param value: the value of the measurement
    """

    source: Source
    extra: Mapping[str, Any]
    id: Union[int, str]
    sample: Sample
    value: Optional[Any] = None

    def get_x(self) -> np.ndarray:
        if not isinstance(self.value, np.ndarray):
            raise TypeError('The returned value should be a numpy array.')
        return self.value

    def get_y(self) -> int:
        return self.source.id

    def __str__(self):
        return f'source_id: {self.source.id}, {self.extra}'

    def __eq__(self, other):
        return self.id == other.id and self.sample == other.sample and self.source == other.source


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

    def __eq__(self, other):
        # equality is symmetric. we ignore extras
        return (other.measurement_a == self.measurement_a and other.measurement_b == self.measurement_b) or \
               (other.measurement_a == self.measurement_b and other.measurement_b == self.measurement_a)
