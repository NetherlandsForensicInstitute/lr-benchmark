import argparse
import os
from pathlib import Path

from confidence import Configuration, loadf
from typing import Optional, Iterable, Union
from lrbenchmark.typing import PathLike


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d',
        '--data-config',
        nargs='*',
        required=True,
        help="The path to a .yaml file containing the configuration for the dataset to be used. Several datasets can"
             "be provided, eg '-d asr glass'. Use '-d all' to analyse all available datasets."
    )
    return parser


def load_data_config(paths: Iterable[PathLike]) -> Iterable[Configuration]:
    """
    Load the data config files for the provided `path`s. These can either be
    the name of the yaml itself or the full path to a config file.
    """
    # first check whether the path refers to an existing file. If not, we extract the
    # full path and then load the file.
    # interpret ['all'] as all datasets
    if paths[0]=='all':
        for path in get_all_data_config_paths():
            yield loadf(path)
    else:
        for path in paths:
            if not os.path.isfile(path):
                path = get_data_config_path(path)
            yield loadf(path)


def get_all_data_config_paths(root: Optional[PathLike] = ".") -> Iterable[PathLike]:
    """
    Returns all available data yamls.
    """
    path = os.path.join(root, 'config', 'data')
    for file in os.listdir(path):
        yield Path(os.path.join(path, file))



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
