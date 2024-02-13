from itertools import combinations_with_replacement

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
    properties_of_holdout_set = [{'property1': property1[0], 'property2': property2[0]},
                                 {'property1': property1[-1], 'property2': property2[-1]}]
    expected_properties = list(combinations_with_replacement(properties_of_holdout_set, r=2))

    assert len(expected_properties) == len(actual_properties)
    for p1, p2 in actual_properties:
        # check both since ordering of 'actual_properties' may differ
        assert (p1, p2) in expected_properties or (p2, p1) in expected_properties
