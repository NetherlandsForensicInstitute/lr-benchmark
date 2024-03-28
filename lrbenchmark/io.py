import csv
from pathlib import Path
from typing import List, Mapping

import numpy as np
from matplotlib import pyplot as plt

from lrbenchmark.typing import Result

LOG_LR = 'log$_{10}$(LR)'


def prepare_output_file(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics(agg_result: List[Result], param_sets: List[Mapping], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/all_metrics.csv'), 'w') as file:
        fieldnames = ['run'] + sorted(param_sets[0].keys()) + sorted(agg_result[0].metrics.keys())
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for i, (result_row, param_set) in enumerate(zip(agg_result, param_sets)):
            params_and_results = dict(param_set, **result_row.metrics)
            params_and_results['run'] = i
            writer.writerow(
                {fieldname: value for fieldname, value in params_and_results.items() if fieldname in fieldnames})


def write_calibration_results(agg_result: List[Result], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/calibration_results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'pair', 'normalized_score', 'is_same_source'])
        for i, result_row in enumerate(agg_result):
            writer.writerows([[i] + r for r in result_row.calibration_results])


def save_holdout_results(agg_result: List[Result], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/holdout_lrs.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'pair', 'LR'])
        Path(f'{folder_name}/holdout_figs/').mkdir(parents=True, exist_ok=True)
        for i, result_row in enumerate(agg_result):
            fig = result_row.figures['lr_distribution']
            for pair_desc, lr in result_row.holdout_lrs.items():
                writer.writerow([i, pair_desc, lr])
                path = f'{folder_name}/holdout_figs/{i}_{pair_desc}.png'
                plot_lr_on_distribution(fig, np.log10(lr), path)


def plot_lr_on_distribution(fig: plt.Figure, log_lr: float, path: str):
    """
    Plot the LR distribution histograms with the actual logLR as vertical line and save the figure.
    """
    vline = fig.axes[0].axvline(log_lr, 0, 0.95)
    fig.axes[0].set_title(f'{LOG_LR}: {log_lr}')
    fig.savefig(path, bbox_inches='tight')
    vline.remove()


def write_refnorm_stats(agg_result: List[Result], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/refnorm_source_ids.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'repeat', 'source_ids'])
        for i, result_row in enumerate(agg_result):
            for repeat, sources in result_row.refnorm_stats.items():
                writer.writerow([i, repeat, *sources])


def save_figures(result: Result, folder_name: str):
    for fig_name, fig in result.figures.items():
        path = f'{folder_name}/{fig_name}'
        prepare_output_file(path)
        fig.savefig(path)
