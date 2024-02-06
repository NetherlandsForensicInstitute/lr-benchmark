import confidence
import numpy as np
import pytest

from lrbenchmark.data.dataset import GlassDataset, XTCDataset, Dataset, ASRDataset
from lrbenchmark.data.simulation import SynthesizedNormalDataset
from lrbenchmark.pairing import CartesianPairing, LeaveOneTwoOutPairing
from tests.conftest import TEST_DIR


@pytest.mark.parametrize('train_size, test_size', [(2, 3), (0.5, 0.2), (4, None), (None, 4), (None, None)])
def test_get_splits_measurements(measurements, train_size, test_size):
    dataset = Dataset(measurements=measurements)
    for dataset_train, dataset_test in dataset.get_splits(seed=0, train_size=train_size, validate_size=test_size):
        X_train, y_train = dataset_train.get_x(), dataset_train.get_y()
        X_test, y_test = dataset_test.get_x(), dataset_test.get_y()
        assert len(np.intersect1d(X_train, X_test)) == 0
        assert len(np.intersect1d(y_train, y_test)) == 0

        if train_size is None and test_size is None:
            assert len(X_train) + len(X_test) == len(measurements)
            assert len(y_train) + len(y_test) == len(measurements)

        assert len(y_train) == train_size if isinstance(train_size, int) else len(y_train) == train_size * len(
            measurements) if isinstance(train_size, float) else len(y_train) == len(
            measurements) - test_size if isinstance(test_size, int) else len(y_train) == test_size * len(
            measurements) if isinstance(test_size, float) else len(measurements) > len(y_train) > 0

        assert len(y_test) == test_size if isinstance(test_size, int) else len(y_test) == test_size * len(
            measurements) if isinstance(test_size, float) else len(y_test) == len(
            measurements) - train_size if isinstance(train_size, int) else len(y_test) == train_size * len(
            measurements) if isinstance(train_size, float) else len(measurements) > len(y_test) > 0

        assert not dataset_train.source_ids.intersection(dataset_test.source_ids)


def test_get_splits_loo(measurements, measurements_set2):
    meas = measurements[:5] + measurements_set2[:5]
    dataset = Dataset(measurements=meas)
    n_sources = len(dataset.source_ids)
    # 5 sources each with 2 measurements
    n_single_groups = n_sources
    n_double_groups = n_sources * (n_sources - 1) / 2
    assert len(list(dataset.get_splits(seed=0, split_type='leave_one_out'))) == n_single_groups + n_double_groups
    pairing_function = CartesianPairing()
    validate_pairs = []
    for dataset_train, dataset_validate in dataset.get_splits(seed=0, split_type='leave_one_out'):
        validate_pairs += dataset_validate.get_pairs(pairing_function=LeaveOneTwoOutPairing())
    all_pairs = dataset.get_pairs(pairing_function=pairing_function)
    assert len(validate_pairs) == len(all_pairs), 'should see all pairs in validation'


@pytest.mark.parametrize("class_name, config_key", [(ASRDataset, 'asr'),
                                                    (XTCDataset, 'xtc'),
                                                    (GlassDataset, 'glass'),
                                                    (SynthesizedNormalDataset, 'normal')])
def test_dataset_basic_functions(class_name, config_key):
    config = confidence.load_name(TEST_DIR / 'lrbenchmark_test')
    if config_key in config.dataset_test:
        dataset = class_name(**config.dataset_test[config_key])
    else:
        dataset = class_name()

    datasets = dataset.get_splits()

    for dataset in datasets:
        for fold in dataset:
            assert isinstance(fold, Dataset)
