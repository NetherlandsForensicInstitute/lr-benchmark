import argparse
import os

from confidence import Configuration, loadf


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


def load_data_config(path: str) -> Configuration:
    """
    Load the data config file for the provided `path`. This can either be
    the name of the yaml itself or the full path.
    """
    if not os.path.isfile(path):
        path = get_data_config_path(path)
    return loadf(path)


def get_data_config_path(path: str) -> str:
    """
    Extract the actual path to a yaml file located in the config folder.
    """
    if not path.endswith('.yaml'):
        path += '.yaml'
    if not path.startswith('config'):
        if not path.startswith('data'):
            path = 'data/' + path
        path = 'config/' + path
    return path
