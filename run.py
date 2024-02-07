#!/usr/bin/env python3
import csv
import logging
from datetime import datetime
from typing import Dict, Any, Mapping

import confidence
import lir.plotting
import lir.util
import matplotlib.pyplot as plt
import numpy as np
from confidence import Configuration
from lir import calculate_lr_statistics, Xy_to_Xn
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from lrbenchmark import evaluation
from lrbenchmark.data.dataset import Dataset
from lrbenchmark.load import get_parser, load_data_config
from lrbenchmark.pairing import BasePairing, CartesianPairing, LeaveOneTwoOutPairing
from lrbenchmark.refnorm import perform_refnorm
from lrbenchmark.transformers import BaseScorer
from lrbenchmark.typing import Result
from lrbenchmark.utils import get_experiment_description, prepare_output_file
from params import parse_config, config_option_dicts

LOG = logging.getLogger(__name__)


def fit_and_evaluate(dataset: Dataset,
                     pairing_function: BasePairing,
                     calibrator: BaseEstimator,
                     scorer: BaseScorer,
                     splitting_strategy: Mapping,
                     properties: Mapping[str, Mapping[str, Any]] = None,
                     selected_params: Dict[str, Any] = None,
                     repeats: int = 1) -> Result:
    """
    Fits an LR system on part of the data, and evaluates its performance on the remainder
    """
    validate_lrs = []
    validate_labels = []
    validate_scores = []

    if splitting_strategy['validation']['split_type'] == 'leave_one_out' \
            and not isinstance(pairing_function, CartesianPairing):
        LOG.warning(f"Leave one out validation will give you cartesian pairing, not {pairing_function}")

    dataset_refnorm = None
    holdout_set = None
    for idx in tqdm(range(repeats), desc=', '.join(map(str, selected_params.values())) if selected_params else ''):
        # split off the sources that should only be evaluated
        holdout_set, dataset = dataset.split_off_holdout_set()
        if splitting_strategy['refnorm']['split_type'] == 'simple':
            dataset, dataset_refnorm = \
                next(dataset.get_splits(validate_size=splitting_strategy['refnorm']['size'], seed=idx))
        splits = dataset.get_splits(seed=idx, **splitting_strategy['validation'])
        if repeats == 1:
            splits = tqdm(list(splits), 'train - validate splits')
        for dataset_train, dataset_validate in splits:

            # if leave one out, take all diff source pairs for 2 sources and all same source pairs for 1 source
            if splitting_strategy['validation']['split_type'] == 'leave_one_out':
                validate_pairs = dataset_validate.get_pairs(pairing_function=LeaveOneTwoOutPairing(), seed=idx)
                # there may be no viable pairs for these sources. If so, go to the next
                if not validate_pairs:
                    continue
            else:
                validate_pairs = dataset_validate.get_pairs(pairing_function=pairing_function, seed=idx)

            train_pairs = dataset_train.get_pairs(pairing_function=pairing_function, seed=idx)

            train_scores = scorer.fit_predict(train_pairs)
            validation_scores = scorer.predict(validate_pairs)

            if splitting_strategy['refnorm']['split_type'] in ('simple', 'leave_one_out'):
                train_scores = perform_refnorm(train_scores, train_pairs, dataset_refnorm or dataset_train, scorer)
                validation_scores = perform_refnorm(validation_scores, validate_pairs, dataset_refnorm or dataset_train,
                                                    scorer)

            calibrator.fit(train_scores, np.array([mp.is_same_source for mp in train_pairs]))
            validate_lrs.append(calibrator.transform(validation_scores))
            validate_labels.append([mp.is_same_source for mp in validate_pairs])
            validate_scores.append(validation_scores)

    # retrain with everything, and apply to the holdout (after the repeat loop)
    if holdout_set:
        holdout_pairs = holdout_set.get_pairs(pairing_function=CartesianPairing(),
                                              trace_reference_properties=properties or {})
        pairs = dataset.get_pairs(pairing_function=pairing_function, seed=idx)
        scores = scorer.fit_predict(pairs)
        holdout_scores = scorer.predict(holdout_pairs)
        if splitting_strategy['refnorm']['split_type'] in ('simple', 'leave_one_out'):
            scores = perform_refnorm(scores, pairs, dataset_refnorm or dataset, scorer)
            holdout_scores = perform_refnorm(holdout_scores, holdout_pairs, dataset_refnorm or dataset,
                                             scorer)
        calibrator.fit(scores, np.array([mp.is_same_source for mp in pairs]))
        holdout_lrs = calibrator.transform(holdout_scores)

    validate_lrs = np.concatenate(validate_lrs)
    validate_labels = np.concatenate(validate_labels)
    validate_scores = np.concatenate(validate_scores)

    # plotting results for a single experiment
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

    descriptive_statistics = {'no of sources': len(dataset.source_ids),
                              'no of pairs train (last repeat)': len(train_pairs),
                              'no of pairs validate (last repeat)': len(validate_pairs)}

    # elub bounds
    lr_metrics = calculate_lr_statistics(*Xy_to_Xn(validate_lrs, validate_labels))

    metrics = {'desc': get_experiment_description(selected_params),
               'cllr': lr_metrics.cllr,
               'cllr_min': lr_metrics.cllr_min,
               'cllr_cal': lr_metrics.cllr_cal,
               'auc': roc_auc_score(validate_labels, validate_scores),
               **descriptive_statistics}

    holdout_results = None
    if holdout_set:
        # holdout set was specified, record LRs. Only takes those from the last repeat.
        holdout_results = {str(pair): lr for pair, lr in zip(holdout_pairs, holdout_lrs)}

    return Result(metrics, figs, holdout_results)


def run(exp: evaluation.Setup, config: Configuration) -> None:
    """
    Executes experiments and saves results to file.
    :param exp: Helper class for execution of experiments.
    :param config: configuration
    """

    config_resolved = parse_config(config, config_option_dicts)
    exp_config = config_resolved['experiment']
    exp.parameter('repeats', exp_config['repeats'])
    exp.parameter('splitting_strategy', exp_config['splitting_strategy'])
    exp.parameter('dataset', config_resolved['dataset'])
    parameters = {'pairing_function': exp_config['pairing'],
                  'scorer': exp_config['scorer'],
                  'calibrator': exp_config['calibrator'],
                  'properties': exp_config.get('properties', [None])}
                  # 'properties': exp_config['properties'] or [None],}

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
    with open(prepare_output_file(f'{folder_name}/all_metrics.csv'), 'w') as file:
        fieldnames = set(agg_result[0].metrics.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result_row in agg_result:
            writer.writerow({fieldname: value for fieldname, value in result_row.metrics.items() if fieldname in fieldnames})

    # write LRs to file
    if agg_result[0].holdout_lrs:
        with open(prepare_output_file(f'{folder_name}/holdout_lrs.csv'), 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['desc', 'pair', 'LR'])
            for result_row in agg_result:
                for pair_desc, lr in result_row.holdout_lrs.items():
                    writer.writerow([result_row.metrics['desc'], pair_desc, lr])

    # save figures and results per parameter set
    for result_row, param_set in zip(agg_result, param_sets):
        for fig_name, fig in result_row.figures.items():
            short_description = ' - '.join([val.__class__.__name__[:5] for val in param_set.values()])
            path = f'{folder_name}/{short_description}/{fig_name}'
            prepare_output_file(path)
            fig.savefig(path)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    config = Configuration(confidence.load_name('lrbenchmark'), load_data_config(args.data_config))
    exp = evaluation.Setup(fit_and_evaluate)

    run(exp, config)
