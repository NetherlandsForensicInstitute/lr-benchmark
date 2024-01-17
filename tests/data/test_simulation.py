import confidence
import pytest

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.simulation import SynthesizedNormalDataset


@pytest.mark.parametrize("class_name, config_key", [
    (SynthesizedNormalDataset, 'normal')])
def test_simulation_basic_functions(class_name, config_key):
    config = confidence.load_name('tests/lrbenchmark_test')
    if config_key in config.dataset_test:
        simulator = class_name(**config.dataset_test[config_key])
    else:
        simulator = class_name()

    dataset = simulator.simulate_dataset()

    sets = dataset.get_splits()

    for set in sets:
        for fold in set:
            assert isinstance(fold, Dataset)
