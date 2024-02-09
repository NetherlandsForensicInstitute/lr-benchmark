from lrbenchmark.data.dataset import Dataset
from lrbenchmark.pairing import CartesianPairing, LeaveOneTwoOutPairing


def test_split_by_properties(measurements, measurements_set2):
    dataset = Dataset(measurements=measurements + measurements_set2)
    pairing_function = CartesianPairing()
    pairs = dataset.get_pairs(pairing_function=pairing_function, seed=0,
                              trace_reference_properties=({'property': 'trace'}, {'property': 'reference'}))
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
