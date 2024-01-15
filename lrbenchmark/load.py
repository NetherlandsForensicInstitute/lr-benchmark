import argparse

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


def load_data_config(config_name: str) -> Configuration:
    """
    Load the data config file for the provided `config_name`. This can either be
    the name of the yaml itself or the full path.
    """
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'
    if not config_name.startswith('config'):
        if not config_name.startswith('data'):
            config_name = 'data/' + config_name
        config_name = 'config/' + config_name
    return loadf(config_name)
