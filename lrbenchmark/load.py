import argparse
import json
import os
from pathlib import Path

from confidence import Configuration, loadf
from typing import Optional, Mapping, Tuple, List

import itertools

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.typing import PathLike


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data-config',
        type=str,
        required=True,
        help="The path to a .yaml file containing the configuration for the dataset to be used."
    )
    return parser


def load_data_config(path: PathLike) -> Configuration:
    """
    Load the data config file for the provided `path`. This can either be
    the name of the yaml itself or the full path to a config file.
    """
    # first check whether the path refers to an existing file. If not, we extract the
    # full path and then load the file.
    if not os.path.isfile(path):
        path = get_data_config_path(path)
    return loadf(path)


def get_data_config_path(path: PathLike, root: Optional[PathLike] = ".") -> PathLike:
    """
    Extract the actual path to a yaml file located in the config folder.
    """
    elements = [root]
    if not path.endswith('.yaml'):
        path += '.yaml'
    if not path.startswith('config'):
        elements.append('config')
        if not path.startswith('data'):
            elements.append('data')
    elements.append(path)
    return Path(os.path.join(*elements))


def get_filter_combination_values(dataset: Dataset) -> \
        List[Tuple[Mapping[str, str], Mapping[str, str]]]:
    """
    For the 'filtering_properties' provided in the Dataset, find the values of the hold out measurements belonging
    to those properties. Then return all combinations of two-sided properties to be used in pairing.
    """
    if not dataset.holdout_source_ids or not dataset.filtering_properties:
        return [({}, {})]
    # retrieve all info of the measurements whose source id is a holdout source id
    all_holdout_properties = [m.extra for m in dataset.measurements if m.source.id in dataset.holdout_source_ids]
    # find all unique values corresponding to the filtering properties
    filtering_values = set(
        [json.dumps({filter_prop: prop.get(filter_prop) for filter_prop in dataset.filtering_properties})
         for prop in all_holdout_properties])
    filtering_values = [json.loads(p) for p in filtering_values]
    return list(itertools.combinations_with_replacement(filtering_values, 2))
