from pathlib import Path
from typing import Dict, Any, Optional, Iterable

from lrbenchmark.data.models import MeasurementPair


def prepare_output_file(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


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


def filter_pairs_on_trace_reference(measurement_pairs: Iterable[MeasurementPair]) -> Iterable[MeasurementPair]:
    """
    Filter measurement pairs on two conditions:
    - the pair must consist of one 'trace_like' measurement and one 'reference_like' measurement
    - the source id's of the two measurements must differ or the id's of the measurements must differ. We also check
    the source id because it may occur that multiple sources have the same measurement id's (for instance when
    measurement id's are counted from 1 for every source).
    """
    return [mp for mp in measurement_pairs if
            ((mp.measurement_a.is_like_reference and mp.measurement_b.is_like_trace) or
             (mp.measurement_a.is_like_trace and mp.measurement_b.is_like_reference)) and
            (mp.measurement_a.id != mp.measurement_b.id or not mp.is_same_source)]


def complies_with_filter_requirements(filter: Dict[str, Any], info: Optional[Dict[str, str]],
                                      extra: Optional[Dict[str, Any]]) -> bool:
    """
    Check whether the values in `info` and `extra` match the values in the `filter`. The values in `filter` and `extra`,
    can be any type (a list of strings, an integer, a boolean), while the values in `info` should be strings. Therefore,
    parse the `filter` and `extra` values to strings or list of strings.
    """
    if info is None:
        info = {}
    info.update(**parse_dict_values_to_str(extra)) if extra else info
    filter = parse_dict_values_to_str(filter)
    for key, val in filter.items():
        if isinstance(val, list):
            if not info.get(key) in val:
                return False
        else:
            if not info.get(key) == val:
                return False
    return True


def parse_dict_values_to_str(dict: Dict[str, Any]) -> Dict[str, str]:
    """
    Parse the values in the dictionary to strings or list of strings.
    """
    result = {}
    for key, val in dict.items():
        if isinstance(val, list):
            result.update({key: list(map(str, val))})
        else:
            result.update({key: str(val)})
    return result
