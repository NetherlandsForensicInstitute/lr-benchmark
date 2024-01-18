import argparse
import os
from pathlib import Path

from confidence import Configuration, loadf
from typing import Optional
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
