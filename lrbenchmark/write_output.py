import csv
from pathlib import Path


def prepare_output_file(path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    return path


def write_metrics(agg_result, folder_name):
    with open(prepare_output_file(f'{folder_name}/all_metrics.csv'), 'w') as file:
        fieldnames = sorted(set(agg_result[0].metrics.keys()))
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result_row in agg_result:
            writer.writerow(
                {fieldname: value for fieldname, value in result_row.metrics.items() if fieldname in fieldnames})


def write_lrs(agg_result, folder_name):
    with open(prepare_output_file(f'{folder_name}/holdout_lrs.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['desc', 'pair', 'LR'])
        for result_row in agg_result:
            for pair_desc, lr in result_row.holdout_lrs.items():
                writer.writerow([result_row.metrics['desc'], pair_desc, lr])


def write_refnorm_stats(agg_result, folder_name):
    with open(prepare_output_file(f'{folder_name}/refnorm_source_ids.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['run', 'source ids'])
        for result_row in agg_result:
            for run, sources in result_row.refnorm_stats.items():
                writer.writerow([run, *sources])