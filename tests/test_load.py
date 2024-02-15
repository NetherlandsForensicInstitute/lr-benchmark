import numpy as np
from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import Measurement, Source, Sample
from lrbenchmark.load import get_filter_combination_values


def test_get_filter_combination_values():
    values = np.reshape(np.array(list(range(50))), (10, 5))
    items = np.array(list(range(10, 20)))
    n_items = len(items) // 2
    property1 = ['yes'] * n_items + ['no'] * n_items
    property2 = ['no'] * n_items + ['yes'] * n_items
    dataset = Dataset(measurements=[Measurement(source=Source(id=item, extra={}),
                                                sample=Sample(id=i), id=1,
                                                extra={'property1': property1[i], 'property2': property2[i]},
                                                value=value) for i, (value, item) in enumerate(zip(values, items))],
                      holdout_source_ids=[items[0], items[-1]],
                      filtering_properties=['property1', 'property2'])

    actual_properties = get_filter_combination_values(dataset)
    expected_properties = [({'property1': 'no', 'property2': 'yes'}, {'property1': 'yes', 'property2': 'no'})]
    assert expected_properties == actual_properties