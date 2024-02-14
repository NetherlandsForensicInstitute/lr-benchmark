import csv
from pathlib import Path
from typing import List, Dict

from lrbenchmark.typing import Result


def prepare_output_file(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics(agg_result: List[Result], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/all_metrics.csv'), 'w') as file:
        fieldnames = sorted(set(agg_result[0].metrics.keys()))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result_row in agg_result:
            writer.writerow(
                {fieldname: value for fieldname, value in result_row.metrics.items() if fieldname in fieldnames})


def write_calibration_results(agg_result: List[Result], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/calibration_results.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['pair', 'normalized_score', 'is_same_source'])
        for result_row in agg_result:
            writer.writerows(result_row.calibration_results)


def write_lrs(agg_result: List[Result], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/holdout_lrs.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['desc', 'pair', 'LR'])
        for result_row in agg_result:
            for pair_desc, lr in result_row.holdout_lrs.items():
                writer.writerow([result_row.metrics['desc'], pair_desc, lr])


def write_refnorm_stats(agg_result: List[Result], folder_name: str):
    with open(prepare_output_file(f'{folder_name}/refnorm_source_ids.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'source_ids'])
        for result_row in agg_result:
            for run, sources in result_row.refnorm_stats.items():
                writer.writerow([f'run_{run}', *sources])


def save_figures_per_param_set(agg_result: List[Result], param_sets: List[Dict], folder_name: str):
    for result_row, param_set in zip(agg_result, param_sets):
        for fig_name, fig in result_row.figures.items():
            short_description = ' - '.join([val.__class__.__name__[:5] for val in param_set.values()])
            path = f'{folder_name}/{short_description}/{fig_name}'
            prepare_output_file(path)
            fig.savefig(path)
