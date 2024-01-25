#!/usr/bin/env python3
import csv
from datetime import datetime
from typing import Dict, Any, Optional

import confidence
import lir.plotting
import lir.util
import matplotlib.pyplot as plt
import numpy as np
from confidence import Configuration
from lir import calculate_lr_statistics, Xy_to_Xn
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from lrbenchmark import evaluation
from lrbenchmark.data.dataset import Dataset
from lrbenchmark.load import get_parser, load_data_config
from lrbenchmark.pairing import BasePairing
from lrbenchmark.refnorm import perform_refnorm
from lrbenchmark.transformers import BaseScorer
from lrbenchmark.utils import get_experiment_description, prepare_output_file
from params import SCORERS, CALIBRATORS, DATASETS, PREPROCESSORS, get_parameters, PAIRING


def evaluate(dataset: Dataset,
             preprocessor: TransformerMixin,
             pairing_function: BasePairing,
             calibrator: BaseEstimator,
             scorer: BaseScorer,
             experiment_config: Configuration,
             selected_params: Dict[str, Any] = None,
             refnorm: Optional[Configuration] = None,
             repeats: int = 1) -> Dict:
    """
    Measures performance for an LR system with given parameters
    """
    validate_lrs = []
    validate_labels = []
    validate_scores = []

    dataset_refnorm = None
    for idx in tqdm(range(repeats), desc=', '.join(map(str, selected_params.values())) if selected_params else ''):
        if refnorm.refnorm_size:
            dataset, dataset_refnorm = next(dataset.get_splits(validate_size=refnorm.refnorm_size, seed=idx))
        for dataset_train, dataset_validate in dataset.get_splits(seed=idx,
                                                                  **experiment_config.experiment.splitting_strategy):
            train_pairs = dataset_train.get_pairs(pairing_function=pairing_function, seed=idx)
            validate_pairs = dataset_validate.get_pairs(pairing_function=pairing_function, seed=idx)

            # todo: what to do with the preprocessor?
            # todo: another way to get the paths in the scorer?
            train_scores = scorer.fit_predict(train_pairs, experiment_config.dataset)
            validation_scores = scorer.predict(validate_pairs)

            if refnorm:
                train_scores = perform_refnorm(train_scores, train_pairs, dataset_refnorm or dataset_train, scorer)
                validation_scores = perform_refnorm(validation_scores, validate_pairs, dataset_refnorm or dataset_train,
                                                  scorer)

            calibrator.fit(train_scores, np.array([mp.is_same_source for mp in train_pairs]))
            validate_lrs.append(calibrator.transform(validation_scores))
            validate_labels.append([mp.is_same_source for mp in validate_pairs])
            validate_scores.append(validation_scores)

    validate_lrs = np.concatenate(validate_lrs)
    validate_labels = np.concatenate(validate_labels)
    validate_scores = np.concatenate(validate_scores)

    # plotting results for a single experiment
    figs = {}
    fig = plt.figure()
    lir.plotting.lr_histogram(validate_lrs, validate_labels, bins=20)
    figs['lr_distribution'] = fig

    lr_metrics = calculate_lr_statistics(*Xy_to_Xn(validate_lrs, validate_labels))

    results = {'desc': get_experiment_description(selected_params),
               'figures': figs, **lr_metrics._asdict(),
               'auc': roc_auc_score(validate_labels, validate_scores)}

    return results


def run(exp: evaluation.Setup, exp_config: Configuration) -> None:
    """
    Executes experiments and saves results to file.
    :param exp: Helper class for execution of experiments.
    :param exp_config: Experiment parameters.
    """
    exp_params = exp_config.experiment
    exp.parameter('repeats', exp_params.repeats)
    exp.parameter('refnorm', exp_params.refnorm)
    exp.parameter('experiment_config', exp_config)
    parameters = {'dataset': get_parameters(exp_config.dataset, DATASETS),
                  'pairing_function': get_parameters(exp_params.pairing, PAIRING),
                  'preprocessor': get_parameters(exp_params.preprocessor, PREPROCESSORS),
                  'scorer': get_parameters(exp_params.scorer, SCORERS),
                  'calibrator': get_parameters(exp_params.calibrator, CALIBRATORS)}

    if [] in parameters.values():
        raise ValueError('Every parameter should have at least one value, '
                         'see README.')

    agg_result, param_sets = [], []
    for param_set, param_values, result in exp.run_full_grid(parameters):
        agg_result.append(result)
        param_sets.append(param_set)

    # create foldername for this run
    folder_name = f'output/{str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))}'

    # write results to file
    with open(prepare_output_file(f'{folder_name}/all_results.csv'), 'w') as file:
        fieldnames = ['desc', 'auc', 'acc', 'cllr', 'cllr_min', 'cllr_cal']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result_row in agg_result:
            writer.writerow({fieldname: value for fieldname, value in result_row.items() if fieldname in fieldnames})

    # save figures and results per parameter set
    for result_row, param_set in zip(agg_result, param_sets):
        for fig_name, fig in result_row['figures'].items():
            short_description = ' - '.join([str(val)[:5] for val in param_set.values()])
            path = f'{folder_name}/{short_description}/{fig_name}'
            prepare_output_file(path)
            fig.savefig(path)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config = Configuration(confidence.load_name('lrbenchmark'), *load_data_config(args.data_config))
    exp = evaluation.Setup(evaluate)

    run(exp, config)
