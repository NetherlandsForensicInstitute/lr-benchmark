from collections import defaultdict

import numpy as np

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import Measurement, Source, Sample
from lrbenchmark.pairing import CartesianPairing, LeaveOneTwoOutPairing


def test_split_by_properties(measurements, measurements_set2):
    dataset = Dataset(measurements=measurements + measurements_set2)
    pairing_function = CartesianPairing()
    pairs = dataset.get_pairs(pairing_function=pairing_function, seed=0,
                              pairing_properties=({'property': 'trace'}, {'property': 'reference'}))
    for pair in pairs:
        m_a, m_b = pair.measurements
        assert (m_a.extra['property'] == 'trace' and m_b.extra['property'] == 'reference') or \
               (m_a.extra['property'] == 'reference' and m_b.extra['property'] == 'trace')


def test_leave_one_out_pairing(measurements, measurements_set2):
    meas = measurements[:1] + measurements_set2[:1]
    pairs = LeaveOneTwoOutPairing().transform(measurements=meas)
    assert len(pairs) == 1  # 1 same source pair
    meas = measurements[:2] + measurements_set2[:2]
    pairs = LeaveOneTwoOutPairing().transform(measurements=meas)
    assert len(pairs) == 4  # 4 diff source pairs


def test_max_m_per_source():
    # make sure there are multiple measurements per source id
    ids = np.concatenate([np.repeat([1, 2, 3], 3), np.array([4])])
    measurements = [Measurement(source=Source(id=id, extra={}),
                                sample=Sample(id=i), id=i, extra={}) for i, id in enumerate(ids)]
    dataset = Dataset(measurements=measurements)
    pairing_function = CartesianPairing()
    max_m_per_source = 2
    pairs = dataset.get_pairs(pairing_function=pairing_function, seed=0,
                              max_m_per_source=max_m_per_source)
    assert len(pairs) == 21

    measurement_ids_per_source = defaultdict(set)
    for mp in pairs:
        m_a, m_b = mp.measurements
        measurement_ids_per_source[m_a.source.id].add(m_a.id)
        measurement_ids_per_source[m_b.source.id].add(m_b.id)
    assert all([len(m_ids) <= max_m_per_source for m_ids in measurement_ids_per_source.values()])
    assert len(measurement_ids_per_source[4]) == 1
