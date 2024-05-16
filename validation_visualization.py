from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import streamlit as st
from pandas import DataFrame


def get_pairing_properties(experiment_folder: Path, group: str):
    run_folder = list(experiment_folder.glob(f'{group}_*'))
    if len(run_folder) > 1:
        folders = '\n'.join([str(f) for f in run_folder])
        st.warning(f"Found more than 1 possible output folder for run group {group}:\n{folders}")
        return "Warning: multiple pairing properties found"
    elif len(run_folder) == 0:
        st.warning(f"Found no possible output folder for run group {group} in directory {experiment_folder}")
        return "Warning: no pairing properties found"
    else:
        f = run_folder[0] / 'run_config.yaml'
        props = yaml.safe_load(f.read_text())
        return props['pairing_properties']


@st.cache_data
def get_calibration_results(path: Path):
    data = pd.read_csv(path)
    if 'lr' in data.columns:
        data['run'] = data['run'].astype(str)
        data['llrs'] = data['lr'].apply(lambda x: np.log10(x))
        distinct_groups = data['run'].unique()
        groups_with_labels = {
            group: get_pairing_properties(experiment_folder, group) for group in
            distinct_groups}
        return data, distinct_groups, groups_with_labels
    else:
        return None, None, None


@st.cache_data
def downsample(data: DataFrame, n_decimals: int = 2):
    data['round_score'] = data['normalized_score'].apply(lambda x: round(x, n_decimals))
    downsampled = calibration_results.groupby(['run', 'round_score']).first()
    return downsampled.reset_index()


experiment_folders = sorted([p.name for p in Path('./output').glob('*')], reverse=True)

experiment = st.selectbox('Select experiment', experiment_folders)

experiment_folder = Path(f'./output/{experiment}')
calibration_results_file = experiment_folder / 'calibration_results.csv'

n_decimals = st.selectbox("Downsample specificity ('None' for no downsampling, might be slow to process)", [0, 1, 2, 3, None])

if calibration_results_file.exists():
    calibration_results, groups, labels = get_calibration_results(calibration_results_file)

    if type(calibration_results) is DataFrame:
        if n_decimals is not None:
            downsampled_results = downsample(calibration_results, n_decimals)
        else:
            downsampled_results = calibration_results

        st.write('Select groups to compare:')
        for group in groups:
            st.checkbox(f"{group}: {labels[group]}", key=group, value=True)

        selected_groups = [key for key in st.session_state.keys() if st.session_state[key]]

        selected_data = downsampled_results[downsampled_results['run'].isin(selected_groups)]

        st.header('Scores to log10 LR per property pairing')
        st.scatter_chart(selected_data, x='normalized_score', y='llrs', color='run')

    else:
        st.warning(f"File '{calibration_results_file}' does not contain the 'lr' column. "
                   f"Run the latest version of run.py to include this column in the results.")

else:
    st.warning(f"File '{calibration_results_file}' not found")
