from lrbenchmark.data.dataset import Dataset
from lrbenchmark.pairing import CartesianPairing, LeaveOneTwoOutPairing

# TODO: change/remove
def test_split_by_reference(measurements, measurements_set2):
    dataset = Dataset(measurements=measurements + measurements_set2)
    pairing_function = CartesianPairing()
    pairs = dataset.get_pairs(pairing_function=pairing_function, seed=0, filter_on_trace_reference_properties=True)
    for pair in pairs:
        assert (pair.measurement_a.is_like_reference and pair.measurement_b.is_like_trace) or (
                pair.measurement_a.is_like_trace and pair.measurement_b.is_like_reference)


def test_leave_one_out_pairing(measurements, measurements_set2):
    meas = measurements[:1] + measurements_set2[:1]
    pairs = LeaveOneTwoOutPairing().transform(measurements=meas)
    assert len(pairs) == 1 # 1 same source pair
    meas = measurements[:2] + measurements_set2[:2]
    pairs = LeaveOneTwoOutPairing().transform(measurements=meas)
    assert len(pairs) == 4 # 4 diff source pairs
