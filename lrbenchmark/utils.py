from pathlib import Path
from typing import Dict, Any, Optional


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
