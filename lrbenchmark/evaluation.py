"""
Example of use:

```python
def run_experiment(selected_params, data, clf):
    clf.fit(data.X, data.y)
    ...
    return results


# Parameter search
exp = Setup(run_experiment)
exp.parameter('data', my_data)
exp.parameter('clf', LogisticRegression())
for selected_params, param_values, results in exp.run_parameter_search("clf", [LogisticRegression(), SVC()]):
    # do something sensible with the results
    ...


# Grid search
exp = Setup(run_experiment)
exp.parameter('data', my_data)  # Default value for parameter
exp.parameter('clf', LogisticRegression())  # Default value for parameter
for selected_params, param_values, results in exp.run_full_grid({'n_most_common_words': [5, 10, 20, 30],
                'binning_strategy', [regular_binning_strategy, mismatch_binning_strategy]}):
    # do something sensible with the results
    ...
"""
import collections
import itertools
import logging
from typing import Callable, Optional, List, Any, Dict, Union

import lir.plotting
import numpy as np
from matplotlib import pyplot as plt
from sklearn.base import TransformerMixin

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import MeasurementPair

LOG = logging.getLogger(__name__)


class Setup:
    def __init__(self, evaluate: Callable):
        self._evaluate = evaluate
        self._default_values = {}

    def parameter(self, name: str, default_value=None):
        """
        Defines a parameter with name `name` and optionally a default value.
        """
        self._default_values[name] = default_value
        return self

    def default_values(self):
        """
        Returns a dictionary with default parameter names and values as key/value.
        """
        return {name: value for name, value in self._default_values.items()}

    def run_full_grid(self, parameter_ranges: Dict[str, List[Any]], default_values: Optional[Dict[str, Any]] = None):
        """
        Runs a full grid of experiments along the dimensions in `names`.

        Parameters:
            - parameter_ranges: `dict` of parameters included in gridsearch; keys are parameter names, values are lists
              of parameter values
            - default_values: any predefined parameter values
        """
        parameter_ranges = collections.OrderedDict(parameter_ranges.items())
        combinations = itertools.product(*parameter_ranges.values())

        experiments = [
            collections.OrderedDict([(dim, combi[i]) for i, dim in enumerate(parameter_ranges.keys())])
            for combi in combinations]
        yield from self.run_experiments(experiments, default_values=default_values)

    def run_parameter_search(self, name: str, values: List[Any], default_values: Optional[Dict[str, Any]] = None):
        """
        Runs a series of experiments, varying a single parameter value.

        Parameters:
            - name: name of parameters for the search
            - values: the values to try out in this search
            - default_values: any predefined parameter values
        """
        experiments = [{name: value} for value in values]
        yield from self.run_experiments(experiments, default_values=default_values)

    def run_defaults(self):
        """
        Runs an experiment with the default values and returns the result
        """
        return self.run_experiment({})[2]

    def run_experiment(self, param_set: Dict[str, Any], default_values: Optional[Dict[str, Any]] = None):
        """
        Runs a single experiment.

        Parameters:
        -----------
            - param_set: a dictionary, with parameter names as keys
            - default_values: any predefined parameter values
        """
        defaults = self.default_values()
        if default_values is not None:
            defaults.update(default_values)

        param_values = defaults.copy()
        for name, value in param_set.items():
            param_values[name] = value

        result = self._evaluate(**param_values)
        return param_set, param_values, result

    def run_experiments(self, experiments: Union[List[Dict[str, Any]], Dict[str, Any]], default_values=None):
        """
        Carry out a range of experiments, for example varying one parameter or doing a gridsearch.

        Parameters:
        -----------
            - experiments: a list of experiments to run
            - default_values: any predefined parameter values
        """
        # Run the experiments
        for param_set in experiments:
            try:
                yield self.run_experiment(param_set, default_values)
            except Exception as e:
                LOG.fatal(f"experiment aborted: {e}; params: {param_set}")
                raise


def compute_descriptive_statistics(dataset: Dataset,
                                   holdout_set: Dataset,
                                   all_train_pairs: List['MeasurementPair'],
                                   all_validate_pairs: List['MeasurementPair'], ) -> Dict[str, int]:
    """
    computes some simple statistics, such as number of sources.
    """
    no_ss_validate = len([pair for pair in all_validate_pairs if pair.is_same_source])
    no_ds_validate = len(all_validate_pairs) - no_ss_validate
    no_sources_validate = len(set([pair.measurement_a.source.id for pair in all_validate_pairs] +
                                  [pair.measurement_b.source.id for pair in all_validate_pairs]))
    no_ss_train = len([pair for pair in all_train_pairs if pair.is_same_source])
    no_ds_train = len(all_train_pairs) - no_ss_train
    no_sources_train = len(set([pair.measurement_a.source.id for pair in all_train_pairs] +
                               [pair.measurement_b.source.id for pair in all_train_pairs]))
    no_sources_holdout = 0
    if holdout_set:
        no_sources_holdout = len(holdout_set.source_ids)

    return {'no of sources': len(dataset.source_ids),
            'no of sources train': no_sources_train,
            'no of sources validate': no_sources_validate,
            'no of sources holdout': no_sources_holdout,
            'no of train SS pairs': no_ss_train,
            'no of train DS pairs': no_ds_train,
            'no of validate SS pairs': no_ss_validate,
            'no of validate DS pairs': no_ds_validate, }


def create_figures(calibrator: TransformerMixin,
                   validate_labels: np.ndarray,
                   validate_lrs: np.ndarray,
                   validate_scores: np.ndarray) -> dict:
    """
    creates a set of evaluation plots, and returns them in a dict for later showing/saving
    """
    figs = {}
    fig = plt.figure()
    lir.plotting.lr_histogram(validate_lrs, validate_labels, bins=20)
    figs['lr_distribution'] = fig
    fig = plt.figure()
    lir.plotting.tippett(validate_lrs, validate_labels)
    figs['tippett'] = fig
    fig = plt.figure()
    lir.plotting.calibrator_fit(calibrator, score_range=(min(validate_scores), max(validate_scores)))
    figs['calibrator_fit'] = fig
    fig = plt.figure()
    lir.plotting.score_distribution(validate_scores, validate_labels)
    figs['score distribution and calibrator fit'] = fig
    return figs
