import ast
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import yaml

import streamlit as st

single_output_tab, multi_output_tab = st.tabs(
    ['Single output', 'Multi output'])


def get_pairing_properties(experiment_folder: Path, group: str):
    """
    Looks for the pairing properties in the 'run_config.yaml' file in
    the run folder. If multiple runs for the same group are found or
    if no folder is found, warnings are printed to the dashboard and returned.
    The property categories are simplified by concatenating them into:
    '{category 1}: {value 1}/{value 2}, {category 2}: {value 1}/{value 2}'
    e.g.:
    'auto: ja/nee, sprekers: nee/nee'
    """
    run_folder = list(experiment_folder.glob(f'{group}_*'))
    if len(run_folder) > 1:
        folders = '\n'.join([str(f) for f in run_folder])
        st.warning(
            f"Found more than 1 possible output folder for run group {group}:"
            f"\n{folders}")
        return "Warning: multiple pairing properties found"
    elif len(run_folder) == 0:
        st.warning(
            f"Found no possible output folder for run group {group} in "
            f"directory {experiment_folder}")
        return "Warning: no pairing properties found"
    else:
        f = run_folder[0] / 'run_config.yaml'
        props = yaml.safe_load(f.read_text())
        prop_l, prop_r = ast.literal_eval(props['pairing_properties'])
        return ', '.join([f"{key}: {prop_l[key]}/{prop_r[key]}" for key in
                          prop_l.keys()])


@st.cache_data
def get_experiment_configs(experiment_folder: Path):
    """
    Get content of experiment configs.yaml if it exists
    """
    config_path = experiment_folder / 'config.yaml'
    if config_path.exists():
        return yaml.safe_load(config_path.read_text())
    else:
        st.warning(f"file '{config_path}' not found")
        return {}


@st.cache_data
def get_all_metrics(experiment_folder: Path):
    """
    Loads the 'all_metrics.csv' file into a dataframe if the file exists.
    Column 'run' is set to string
    """
    file_path = experiment_folder / 'all_metrics.csv'
    if file_path.exists():
        data = pd.read_csv(file_path)
        data['run'] = data['run'].astype(str)
        return data
    else:
        return None


def get_counts_per_run(experiment_folder: Path, group: str):
    """
    Looks up the train, validation and total counts per run in the
    all_metrics.csv file in the output folder.
    Counts are returned as a string, or a warning string is returned if the
    file is not found
    """
    data = get_all_metrics(experiment_folder)
    if type(data) is pd.DataFrame:
        train_counts, val_counts = data[data['run'] == group][
            ['no of sources train', 'no of sources validate']].values[0]
        return (f"n_train: {train_counts}, n_val: {val_counts}, "
                f"n_total: {train_counts + val_counts}")
    else:
        return f"Warning: file 'all_metrics.csv' not found"


@st.cache_data
def get_calibration_results(file_path: Path, experiment_folder: Path):
    """
    Read the data from the csv file.
    - Changes 'run' column into string
    - Adds 'llrs' column by applying log 10 to 'lr' column
    - Adds 'pairing_property' column by getting pairing properties from
    the run_config.yaml in the underlying run folder

    returns data, distinct run id's, and a dictionary of the run id's with
    the 'pairing properties' and 'counts' of that run
    If the column 'lr' does not exist in the data, None values are returned
    """
    data = pd.read_csv(file_path)
    if 'lr' in data.columns:
        data['run'] = data['run'].astype(str)
        data['llrs'] = data['lr'].apply(lambda x: np.log10(x))
        distinct_groups = data['run'].unique()
        groups_with_labels = {
            group: {
                "pairing properties": get_pairing_properties(experiment_folder,
                                                             group),
                "counts": get_counts_per_run(experiment_folder,
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
    """
    Downsample the data by rounding the normalized_score to n_decimals and
    selecting the first row per rounded score
    """
    data['round_score'] = data['normalized_score'].apply(
        lambda x: round(x, n_decimals))
    downsampled = data.groupby(['run', 'round_score']).first()
    return downsampled.reset_index()


@st.cache_data
def get_groupdata_as_lists(data: pd.DataFrame, groups: List,
                           target_column: str) -> List[List]:
    """
    Return a list containing the lists of the target_column per group
    """
    data_list = []
    for group in groups:
        data_list.append(data[data['run'] == group][target_column].tolist())
    return data_list


@st.cache_data
def merge_dataframes(lhs_df: pd.DataFrame, rhs_df: pd.DataFrame,
                     on: str or List, suffixes: Tuple):
    """
    Merge two dataframes on one or multiple columns. Add suffixes to
    overlapping columns.
    Put in a function to be able to cache the data.
    """
    if on is None:
        on = ['pairing_category', 'pair']
    return pd.merge(lhs_df, rhs_df, on=on, suffixes=suffixes)


with single_output_tab:
    st.header('Analyse single experiment data')
    # List all experiment outputfolder names in reversed order (i.e. latest
    # results are selected by default)
    experiment_folders = sorted([p.name for p in Path('./output').glob('*')],
                                reverse=True)

    experiment = st.selectbox('Select experiment output folder',
                              experiment_folders)

    experiment_folder = Path(f'./output/{experiment}')
    calibration_results_file = experiment_folder / 'calibration_results.csv'

    if calibration_results_file.exists():
        calibration_results, groups, labels = get_calibration_results(
            calibration_results_file, experiment_folder)

        if type(calibration_results) is pd.DataFrame:
            n_decimals = st.selectbox(
                "Downsample specificity: Higher number leads to more data points, "
                "'None' for all data points/no downsampling (might be slow to "
                "process)",
                [0, 1, 2, 3, None], index=1)

            if n_decimals is not None:
                selected_data = downsample(calibration_results, n_decimals)
            else:
                selected_data = calibration_results

            st.write('**Counts per pairing category:**')
            for group in groups:
                st.write(
                    f"{labels[group]['pairing properties']}: {labels[group]['counts']}")

            st.header('Figures')
            st.info(
                'Legends of the figures are clickable to show or hide data')

            # Scatterplot
            fig = px.scatter(selected_data, x='normalized_score', y='llrs',
                             color=selected_data['pairing_property'],
                             title='Normalized scores to log10 LR per property pairing',
                             labels={
                                 'normalized_score': 'Normalized score',
                                 'llrs': 'llr',
                                 'pairing_property': 'Pairing property'
                             })

            st.plotly_chart(fig)

            # KDE plot normalized scores
            scores_data = get_groupdata_as_lists(selected_data, groups,
                                                 'normalized_score')
            kde_score = ff.create_distplot(scores_data,
                                           group_labels=selected_data[
                                               'pairing_property'].unique())
            kde_score.update_layout(
                title_text='Distplot of normalized scores per pairing category')
            st.plotly_chart(kde_score)

            # KDE plot llrs. For lrs, change target_column on line 140 to 'lr'
            llrs_data = get_groupdata_as_lists(selected_data, groups, 'llrs')
            kde_llr = ff.create_distplot(llrs_data, group_labels=selected_data[
                'pairing_property'].unique())
            kde_llr.update_layout(
                title_text='Distplot of llrs per pairing category')
            st.plotly_chart(kde_llr)

        else:
            st.warning(
                f"File '{calibration_results_file}' does not contain the 'lr' column. "
                f"Run the latest version of run.py to include this column in the results.")

    else:
        st.warning(f"File '{calibration_results_file}' not found")

with multi_output_tab:
    st.header('Analyse over multiple experiments')
    experiment_folders = sorted([p.name for p in Path('./output').glob('*')],
                                reverse=True)

    # Left hand side
    experiment_lhs = st.selectbox('Select left hand side experiment output folder',
                                  experiment_folders)
    experiment_folder_lhs = Path('.') / 'output' / experiment_lhs
    calibration_results_file_lhs = experiment_folder_lhs / 'calibration_results.csv'
    calibration_results_lhs, _, _ = get_calibration_results(
        calibration_results_file_lhs, Path(f'./output/{experiment_lhs}'))
    st.write(get_experiment_configs(experiment_folder_lhs))

    # Right hand side
    experiment_rhs = st.selectbox('Select right hand side experiment output folder',
                                  experiment_folders)
    experiment_folder_rhs = Path('.') / 'output' / experiment_rhs
    calibration_results_file_rhs = Path(
        f'./output/{experiment_rhs}') / 'calibration_results.csv'
    calibration_results_rhs, _, _ = get_calibration_results(
        calibration_results_file_rhs, Path(f'./output/{experiment_rhs}'))
    st.write(get_experiment_configs(experiment_folder_rhs))

    if experiment_lhs == experiment_rhs:
        st.warning('Same experiment folder selected for left hand side and '
                   'right hand side')
    calibration_results_all = merge_dataframes(
        calibration_results_lhs[['llrs', 'pair', 'pairing_property']],
        calibration_results_rhs[['llrs', 'pair', 'pairing_property']],
        on=['pairing_property', 'pair'], suffixes=("_lhs", "_rhs"))

    lhs_label = st.text_input('Label for lhs axis in figure:', 'llr_lhs')
    lhs_label = lhs_label if lhs_label else 'llr_lhs'
    rhs_label = st.text_input('Label for rhs axis in figure:', 'llr_rhs')
    rhs_label = rhs_label if rhs_label else 'llr_rhs'

    if calibration_results_all.empty:
        st.warning('Unable to merge experiment data. Are you sure the '
                   'experiments use the same dataset?')

    fig = px.scatter(calibration_results_all, x='llrs_lhs', y='llrs_rhs',
                     color='pairing_property',
                     title='llr per pair for different run conditions',
                     labels={
                         'llrs_lhs': lhs_label,
                         'llrs_rhs': rhs_label,
                         'pairing_property': 'Pairing property'
                     })
    st.plotly_chart(fig)
