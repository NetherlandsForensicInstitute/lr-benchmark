from typing import List

import numpy as np
from tqdm import tqdm

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import MeasurementPair


def refnorm(score: float, scores_m_a: List[float], scores_m_b: List[float]) -> float:
    """
    Performs the reference normalization on the score, based on the mean and standard deviation of the scores of
    both individual measurements with the refnorm measurements.
    """
    norm_a = (score - (sum(scores_m_a) / len(scores_m_a))) / np.std(scores_m_a)
    norm_b = (score - (sum(scores_m_b) / len(scores_m_b))) / np.std(scores_m_b)
    return round((norm_a + norm_b) / 2, 6)


def perform_refnorm(train_scores: np.ndarray, train_pairs: List[MeasurementPair], refnorm_dataset: Dataset, scorer):
    """
    Transform the scores of the measurement pairs with reference normalization. For each measurement in the
    measurement pair, the appropriate refnorm measurement pairs are selected (i.e. all pairs of which one of the
    measurements is equal to the measurement that has to be normalized, and the other measurement has a source_id
    that is not in the `source_ids_to_exclude` list or equal to the source ids in the measurement pair).
    Once the refnorm pairs are selected, their scores are extracted and used for the transformation. The normalized
    score is replaced in the measurement pair.

    :param refnorm_dataset: the dataset from which to select measurement pairs to perform the refnorm transformation
    :param source_ids_to_exclude: list of source_ids which the complementary measurement is not allowed to have.
    """
    normalized_scores = []
    for (score, mp) in tqdm(zip(train_scores, train_pairs), desc="Performing reference normalization", position=0):
        refnorm_measurements = [measurement for measurement in refnorm_dataset.measurements if
                                measurement.source not in [mp.measurement_a.source, mp.measurement_b.source]]
        scores_m_a = scorer.transform([MeasurementPair(mp.measurement_a, other_measurement) for other_measurement in
                      refnorm_measurements])
        scores_m_b = scorer.transform([MeasurementPair(mp.measurement_b, other_measurement) for other_measurement in
                      refnorm_measurements])
        normalized_score = refnorm(score, scores_m_a, scores_m_b)
        normalized_scores.append(normalized_score)
    return np.array(normalized_scores)
