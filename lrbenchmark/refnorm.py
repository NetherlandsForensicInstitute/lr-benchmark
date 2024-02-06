from typing import List

import numpy as np

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import MeasurementPair
from lrbenchmark.transformers import BaseScorer


def refnorm(score: float, scores_m_a: List[float], scores_m_b: List[float]) -> float:
    """
    Performs the reference normalization on the score, based on the mean and standard deviation of the scores of
    both individual measurements with the refnorm measurements.
    """
    norm_a = (score - (sum(scores_m_a) / len(scores_m_a))) / np.std(scores_m_a, ddof=1)
    norm_b = (score - (sum(scores_m_b) / len(scores_m_b))) / np.std(scores_m_b, ddof=1)
    return round((norm_a + norm_b) / 2, 6)


def perform_refnorm(train_scores: np.ndarray, train_pairs: List[MeasurementPair], refnorm_dataset: Dataset,
                    scorer: BaseScorer) -> np.ndarray:
    """
    Transform the scores of the measurement pairs with reference normalization. For each measurement in the
    measurement pair, the appropriate refnorm measurement pairs are selected (i.e. all pairs of which one of the
    measurements is equal to the measurement that has to be normalized, and the other measurement has a source_id not
    equal to the source ids in the measurement pair).
    Once the refnorm pairs are selected, their scores are retrieved by applying a scorer and used for the
    transformation. The normalized scores are returned.

    :param train_scores: the scores of the training pairs that are to be normalized
    :param train_pairs: the training pairs containing necessary information on the sources.
    :param refnorm_dataset: the dataset from which to select measurement pairs to perform the refnorm transformation
    :param scorer: scorer class used to find the scores of the pairs consisting of one measurement from the refnorm
    dataset and one of the train pair
    """
    normalized_scores = []
    for (score, mp) in zip(train_scores, train_pairs):
        refnorm_measurements = [measurement for measurement in refnorm_dataset.measurements if
                                measurement.source not in [mp.measurement_a.source, mp.measurement_b.source]]
        scores_m_a = scorer.predict([MeasurementPair(mp.measurement_a, other_measurement) for other_measurement
                                     in refnorm_measurements])
        scores_m_b = scorer.predict([MeasurementPair(mp.measurement_b, other_measurement) for other_measurement
                                     in refnorm_measurements])
        normalized_score = refnorm(score, scores_m_a, scores_m_b)
        normalized_scores.append(normalized_score)
    return np.array(normalized_scores)
