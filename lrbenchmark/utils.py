from collections import defaultdict
import random
from typing import Any, Optional, Mapping, Tuple, Iterable, List

import itertools

from lrbenchmark.data.models import Measurement, MeasurementPair


def select_max_measurements_per_source(max_m_per_source: int,
                                       measurements: Iterable[Measurement],
                                       seed: Optional[int] = None,
                                       pairing_properties: Tuple[Mapping[str, str], Mapping[str, str]] = ({}, {})) \
        -> List[Measurement]:
    """
    Select at most `max_m_per_source` measurements per source that comply with any of the two`pairing_properties`
    and return the list of remaining measurements. If there are more than `max_m_per_source` measurements for a source,
    we randomly sample `max_m_per_source` measurements.
    """
    random.seed(seed)
    m_per_source = defaultdict(list)
    # gather all measurements that comply with the properties per source
    for m in measurements:
        if measurement_complies_with_properties(m, pairing_properties):
            m_per_source[m.source.id].append(m)
    # randomly sample the measurements per source if there are more than 'max_m_per_source' measurements
    m_per_source = {source: random.sample(ms, max_m_per_source) if len(ms) > max_m_per_source else ms
                    for source, ms in m_per_source.items()}
    return list(itertools.chain(*m_per_source.values()))


def pair_complies_with_properties(measurement_pair: MeasurementPair,
                                  properties: Tuple[Mapping[str, str], Mapping[str, str]]) -> bool:
    """
    Check a measurement pair on two conditions:
    - one of the measurements of the pair must have properties equal to the first properties and the other measurement
    must have properties equal to the second properties
    - the sample id or the source id of the measurements must differ
    """
    m_a, m_b = measurement_pair.measurements
    return ((complies_with_filter_requirements(properties[0], m_a.extra) and
             complies_with_filter_requirements(properties[1], m_b.extra)) or
            (complies_with_filter_requirements(properties[1], m_a.extra) and
            complies_with_filter_requirements(properties[0], m_b.extra))) and \
           (m_a.source.id != m_b.source.id or m_a.sample.id != m_b.sample.id)


def measurement_complies_with_properties(measurement: Measurement,
                                         properties: Tuple[Mapping[str, str], Mapping[str, str]]) -> bool:
    """
    Check whether a measurement has properties equal to any of the two `properties` provided.
    """
    return complies_with_filter_requirements(properties[0], measurement.extra) or \
        complies_with_filter_requirements(properties[1], measurement.extra)


def complies_with_filter_requirements(requirements: Mapping[str, Any], info: Optional[Mapping[str, str]] = None,
                                      extra: Optional[Mapping[str, Any]] = None) -> bool:
    """
    Check whether the values in `info` and `extra` match the values in the `requirements`. The values in `requirements`
    and `extra`, can be any type (a list of strings, an integer, a boolean), while the values in `info` should be
    strings. Therefore, parse the `requirements` and `extra` values to strings or list of strings.
    """
    if info is None:
        info = {}
    info.update(**parse_dict_values_to_str(extra)) if extra else info
    requirements = parse_dict_values_to_str(requirements)
    for key, val in requirements.items():
        if isinstance(val, list):
            if not info.get(key) in val:
                return False
        else:
            if not info.get(key) == val:
                return False
    return True


def parse_dict_values_to_str(mapping_to_parse: Mapping[str, Any]) -> Mapping[str, str]:
    """
    Parse the values in the mapping to strings or list of strings.
    """
    result = {}
    for key, val in mapping_to_parse.items():
        if isinstance(val, list):
            result.update({key: list(map(str, val))})
        else:
            result.update({key: str(val)})
    return result
