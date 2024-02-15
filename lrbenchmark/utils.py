from typing import Dict, Any, Optional, Iterable

from pathlib import Path
from typing import Dict, Any, Optional, Mapping, Tuple
from lrbenchmark.data.models import MeasurementPair


def get_experiment_description(selected_params: Optional[Dict[str, Any]]) -> str:
    """
    Concatenates information about the parameters of the experiment into a string.
    :param selected_params: dict of selected parameters and their values
    :return: experiment description (str)
    """
    if selected_params:
        string_agg = []
        for param_name, param_value in selected_params.items():
            if isinstance(param_value, int):
                string_agg.append(f'{param_name}={param_value}')
            elif hasattr(param_value, 'calibrator'):
                string_agg.append(f'{param_name}={str(param_value.__class__)}_{param_value.calibrator}')
            elif hasattr(param_value, 'first_step_calibrator'):
                string_agg.append(f'{param_name}={str(param_value.__class__)}_{param_value.first_step_calibrator}')
            else:
                string_agg.append(f'{param_name}={param_value}')
        return '_'.join(string_agg)
    else:
        return "defaults"


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
