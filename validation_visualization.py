import ast
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import yaml

import streamlit as st


def get_pairing_properties(experiment_folder: Path, group: str):
    run_folder = list(experiment_folder.glob(f'{group}_*'))
    if len(run_folder) > 1:
        folders = '\n'.join([str(f) for f in run_folder])
        st.warning(
            f"Found more than 1 possible output folder for run group {group}:\n{folders}")
        return "Warning: multiple pairing properties found"
    elif len(run_folder) == 0:
        st.warning(
            f"Found no possible output folder for run group {group} in directory {experiment_folder}")
        return "Warning: no pairing properties found"
    else:
        f = run_folder[0] / 'run_config.yaml'
        props = yaml.safe_load(f.read_text())
        prop_l, prop_r = ast.literal_eval(props['pairing_properties'])
        return ', '.join([f"{key}: {prop_l[key]}/{prop_r[key]}" for key in
                          prop_l.keys()])


@st.cache_data
def get_all_metrics(experiment_folder: Path):
    file_path = experiment_folder / 'all_metrics.csv'
    if file_path.exists():
        data = pd.read_csv(file_path)
        data['run'] = data['run'].astype(str)
        return data
    else:
        return None


def get_counts_per_pairing_properties(experiment_folder: Path, group: str):
    data = get_all_metrics(experiment_folder)
    if type(data) is pd.DataFrame:
        train_counts, val_counts = data[data['run'] == group][
            ['no of sources train', 'no of sources validate']].values[0]
        return f"n_train: {train_counts}, n_val: {val_counts}, n_total: {train_counts + val_counts}"
    else:
        return f"Warning: file 'all_metrics.csv' not found"


@st.cache_data
def get_calibration_results(path: Path, experiment_folder: Path):
    data = pd.read_csv(path)
    if 'lr' in data.columns:
        data['run'] = data['run'].astype(str)
        data['llrs'] = data['lr'].apply(lambda x: np.log10(x))
        distinct_groups = data['run'].unique()
        groups_with_labels = {
            group: {
                "pairing properties": get_pairing_properties(experiment_folder,
                                                             group),
                "counts": get_counts_per_pairing_properties(experiment_folder,
                                                            group)} for group
            in
            distinct_groups}
        data['pairing_property'] = data['run'].apply(
            lambda x: str(groups_with_labels[x]['pairing properties']))
        return data, distinct_groups, groups_with_labels
    else:
        return None, None, None


@st.cache_data
def downsample(data: pd.DataFrame, n_decimals: int = 2):
    data['round_score'] = data['normalized_score'].apply(
        lambda x: round(x, n_decimals))
    downsampled = calibration_results.groupby(['run', 'round_score']).first()
    return downsampled.reset_index()


@st.cache_data
def get_groupdata_as_lists(data: pd.DataFrame, groups: List, target_column: str):
    data_list = []
    for group in groups:
        data_list.append(data[data['run'] == group][target_column].tolist())
    return data_list


experiment_folders = sorted([p.name for p in Path('./output').glob('*')],
                            reverse=True)

experiment = st.selectbox('Select experiment folder', experiment_folders)

experiment_folder = Path(f'./output/{experiment}')
calibration_results_file = experiment_folder / 'calibration_results.csv'

n_decimals = st.selectbox(
    "Downsample specificity: Higher number leads to more data points, 'None' for all data points/no downsampling (might be slow to process)",
    [0, 1, 2, 3, None], index=1)

if calibration_results_file.exists():
    calibration_results, groups, labels = get_calibration_results(
        calibration_results_file, experiment_folder)

    if type(calibration_results) is pd.DataFrame:
        if n_decimals is not None:
            selected_data = downsample(calibration_results, n_decimals)
        else:
            selected_data = calibration_results

        st.write('**Counts per pairing category:**')
        for group in groups:
            f"{labels[group]['pairing properties']}: {labels[group]['counts']}"

        st.header('Figures')
        st.info('Legends of the figures are clickable to show or hide data')

        fig = px.scatter(selected_data, x='normalized_score', y='llrs',
                         color=selected_data['pairing_property'],
                         title='Normalized scores to log10 LR per property pairing',
                         labels={
                             'normalized_score': 'Normalized score',
                             'llrs': 'llr',
                             'pairing_property': 'Pairing property'
                         })

        st.plotly_chart(fig)

        scores_data = get_groupdata_as_lists(selected_data, groups, 'normalized_score')
        kde_score = ff.create_distplot(scores_data, group_labels=selected_data['pairing_property'].unique())
        kde_score.update_layout(title_text='Distplot of normalized scores per pairing category')
        st.plotly_chart(kde_score)

        scores_data = get_groupdata_as_lists(selected_data, groups, 'llrs')
        kde_llr = ff.create_distplot(scores_data, group_labels=selected_data['pairing_property'].unique())
        kde_llr.update_layout(title_text='Distplot of llrs per pairing category')
        st.plotly_chart(kde_llr)

    else:
        st.warning(
            f"File '{calibration_results_file}' does not contain the 'lr' column. "
            f"Run the latest version of run.py to include this column in the results.")

else:
    st.warning(f"File '{calibration_results_file}' not found")
