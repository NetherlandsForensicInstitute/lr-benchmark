#!/usr/bin/env python3
import csv
from datetime import datetime
from typing import Dict, Any

import confidence
import lir.plotting
import lir.util
import matplotlib.pyplot as plt
import numpy as np
from confidence import Configuration
from lir import calculate_lr_statistics, Xy_to_Xn
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from lrbenchmark import evaluation
from lrbenchmark.data.dataset import Dataset
from lrbenchmark.load import get_parser, load_data_config
from lrbenchmark.transformers import DummyClassifier
from lrbenchmark.utils import get_experiment_description, prepare_output_file
from params import SCORERS, CALIBRATORS, DATASETS, PREPROCESSORS, get_parameters


def evaluate(dataset: Dataset,
             preprocessor: TransformerMixin,
             calibrator: TransformerMixin,
             scorer: BaseEstimator,
             splitting_strategy_config: Configuration,
             selected_params: Dict[str, Any] = None,
             repeats: int = 1) -> Dict:
    """
    Measures performance for an LR system with given parameters
    """
    calibrated_scorer = lir.CalibratedScorer(scorer, calibrator)

    test_lrs = []
    test_labels = []
    test_probas = []
    test_predictions = []

    for idx in tqdm(range(repeats), desc=', '.join(map(str, selected_params.values())) if selected_params else ''):
        for dataset_train, dataset_test in dataset.get_splits(seed=idx, **splitting_strategy_config):
            X_train, y_train = dataset_train.get_x_y_pairs(seed=idx)
            X_test, y_test = dataset_test.get_x_y_pairs(seed=idx)

            if preprocessor:
                X_train = preprocessor.fit_transform(X_train)
                X_test = preprocessor.fit_transform(X_test)

            calibrated_scorer.fit(X_train, y_train)
            test_lrs.append(calibrated_scorer.predict_lr(X_test))
            test_labels.append(y_test)

            if not isinstance(calibrated_scorer.scorer, DummyClassifier):
                test_probas.append(calibrated_scorer.scorer.predict_proba(X_test)[:, 1])
                test_predictions.append(calibrated_scorer.scorer.predict(X_test))

    test_lrs = np.concatenate(test_lrs)
    test_labels = np.concatenate(test_labels)

    # plotting results for a single experiment
    figs = {}
    fig = plt.figure()
    lir.plotting.lr_histogram(test_lrs, test_labels, bins=20)
    figs['lr_distribution'] = fig

    lr_metrics = calculate_lr_statistics(*Xy_to_Xn(test_lrs, test_labels))

    results = {'desc': get_experiment_description(selected_params),
               'figures': figs,
               **lr_metrics._asdict()}

    if not isinstance(calibrated_scorer.scorer, DummyClassifier):
        test_probas = np.concatenate(test_probas)
        test_predictions = np.concatenate(test_predictions)
        results['auc'] = roc_auc_score(test_labels, test_probas)
        results['acc'] = accuracy_score(test_labels, test_predictions)

    return results


def run(exp: evaluation.Setup, exp_config: Configuration, data_config: Configuration) -> None:
    """
    Executes experiments and saves results to file.
    :param exp: Helper class for execution of experiments.
    :param exp_config: Experiment parameters.
    :param data_config: Dataset parameters.
    """
    exp_params = exp_config.experiment
    exp.parameter('repeats', exp_params.repeats)
    parameters = {'dataset': get_parameters(data_config.dataset, DATASETS),
                  'splitting_strategy_config': [exp_params.splitting_strategy],
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
    data_config = load_data_config(args.data_config)
    config = confidence.load_name('lrbenchmark')
    exp = evaluation.Setup(evaluate)

    run(exp, config, data_config)
