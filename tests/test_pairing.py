from lrbenchmark.data.dataset import Dataset
from lrbenchmark.pairing import CartesianPairing

# TODO: change/remove
def test_split_by_reference(measurements, measurements_set2):
    dataset = Dataset(measurements=measurements + measurements_set2)
    pairing_function = CartesianPairing()
    pairs = dataset.get_pairs(pairing_function=pairing_function, seed=0, filter_on_trace_reference_properties=True)
    for pair in pairs:
        assert (pair.measurement_a.is_like_reference and pair.measurement_b.is_like_trace) or (
                pair.measurement_a.is_like_trace and pair.measurement_b.is_like_reference)
