from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import streamlit as st


def get_pairing_properties(experiment_folder: Path, group: str):
    run_folder = list(experiment_folder.glob(f'{group}_*'))
    if len(run_folder) > 1:
        print(f"Found more than 1 possible output folder for run group {group}:\n'\n'.join(run_folder)")
        return None
    elif len(run_folder) == 0:
        print(f"Found no possible output folder for run group {group} in directory {experiment_folder}")
        return None
    else:
        f = run_folder[0] / 'run_config.yaml'
        props = yaml.safe_load(f.read_text())
        return props['pairing_properties']


experiment_folders = [p.name for p in Path('./output').glob('*')]

experiment = st.selectbox('Select experiment', experiment_folders)

experiment_folder = Path(f'./output/{experiment}')
calibration_results_path = experiment_folder / 'calibration_results.csv'


@st.cache_data
def get_calibration_results(path):
    data = pd.read_csv(path)
    data['run'] = data['run'].astype(str)
    data['llrs'] = data['lrs'].apply(lambda x: np.log10(x))
    distinct_groups = data['run'].unique()
    groups_with_labels = {
        group: get_pairing_properties(experiment_folder, group) for group in
        distinct_groups}
    return data, distinct_groups, groups_with_labels


calibration_results, groups, labels = get_calibration_results(calibration_results_path)

@st.cache_data
def downsample(data):
    data['round_score'] = data['normalized_score'].apply(lambda x: round(x))
    downsampled = calibration_results.groupby(['run', 'round_score']).first()
    return downsampled.reset_index()


downsampled_results = downsample(calibration_results)

st.write('Select groups to compare:')
for group in groups:
    st.checkbox(f"{group}: {labels[group]}", key=group)

selected_groups = [key for key in st.session_state.keys() if st.session_state[key]]

selected_data = downsampled_results[downsampled_results['run'].isin(selected_groups)]

st.header('Scores to log LR per ')
st.scatter_chart(selected_data, x='normalized_score', y='llrs', color='run')

