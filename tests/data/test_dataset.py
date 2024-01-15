from typing import List

import numpy as np
import pytest

from lrbenchmark.data.generated import SynthesizedNormalDataset
from lrbenchmark.data.models import Measurement, Source
from lrbenchmark.data.dataset import CommonSourceKFoldDataset, ASRDataset, GlassDataset
import confidence

@pytest.fixture
def measurements() -> List[Measurement]:
    values = np.reshape(np.array(list(range(25))), (5, 5))
    items = np.array([10, 11, 12, 13, 14])
    return [Measurement(source=Source(id=item, extra={}),
                        extra={},
                        value=value) for value, item in zip(values, items)]

def test_get_splits_is_mutually_exclusive(measurements):
    dataset = CommonSourceKFoldDataset(n_splits=3, measurements=measurements)
    for dataset_train, dataset_test in dataset.get_splits(seed=0):
        X_train, y_train = dataset_train.get_x_y_measurement()
        X_test, y_test = dataset_test.get_x_y_measurement()
        assert len(np.intersect1d(X_train, X_test)) == 0 and len(X_train) + len(X_test) == len(dataset.measurements)
        assert len(np.intersect1d(y_train, y_test)) == 0 and len(y_train) + len(y_test) == len(dataset.measurements)


@pytest.mark.parametrize("class_name, config_key, load", [#(ASRDataset, 'asr', True),
                                               (GlassDataset, 'glass', True),
                                               (SynthesizedNormalDataset, 'normal', False)])
def test_dataset_basic_functions(class_name, config_key, load):
    config = confidence.load_name('tests/lrbenchmark_test')

    dataset = class_name(**config.dataset_test[config_key])
    if load:
        dataset.load()
    else:
        dataset = dataset.generate_data(1000)

    dataset.get_splits()







