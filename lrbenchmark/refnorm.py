import itertools
from typing import List

import numpy as np
from tqdm import tqdm

from lrbenchmark.data.dataset import Dataset
from lrbenchmark.data.models import MeasurementPair
from lrbenchmark.transformers import BaseScorer


def refnorm(score: float, scores_m_a: np.ndarray, scores_m_b: np.ndarray) -> float:
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

    # If the scorer has scores, source_indices and measurement_indices as attributes, the scores can be sliced directly
    # from the scores matrix
    if hasattr(scorer, 'scores') and hasattr(scorer, 'source_indices') and hasattr(scorer, 'measurement_indices'):
        # Take a set of indices to keep, based on the source ids in the refnorm dataset
        indices_to_keep = set(
            itertools.chain.from_iterable([scorer.source_indices[key] for key in refnorm_dataset.source_ids]))

        for (score, mp) in tqdm(zip(train_scores, train_pairs), desc='applying refnorm', total=len(train_scores),
                                position=0):
            # Remove indices from sources in pair
            indices_to_keep_pair = indices_to_keep.copy()
            indices_to_keep_pair.difference_update(
                np.concatenate(
                    (scorer.source_indices[mp.measurement_a.source.id],
                     scorer.source_indices[mp.measurement_b.source.id])))

            # Slice the measurement data to retrieve the scores
            scores_a = scorer.scores[scorer.measurement_indices[mp.measurement_a.extra['filename']],
                                     list(indices_to_keep_pair)]
            scores_b = scorer.scores[scorer.measurement_indices[mp.measurement_b.extra['filename']],
                                     list(indices_to_keep_pair)]

            # Perform refnorm
            normalized_score = refnorm(score, scores_a, scores_b)
            normalized_scores.append(normalized_score)

    else:  # If the scorer is of a different type, the scores should be calculated for each measurement pair
        for (score, mp) in tqdm(zip(train_scores, train_pairs), desc='applying refnorm', total=len(train_scores),
                                position=0):
            # Collect the measurements to use for the refnorm
            refnorm_measurements = [measurement for measurement in refnorm_dataset.measurements if
                                    measurement.source not in [mp.measurement_a.source, mp.measurement_b.source]]

            # Predict the score for each measurement pair
            scores_m_a = scorer.predict(
                [MeasurementPair(mp.measurement_a, other_measurement) for other_measurement in refnorm_measurements])
            scores_m_b = scorer.predict(
                [MeasurementPair(mp.measurement_b, other_measurement) for other_measurement in refnorm_measurements])

            # Perform refnorm
            normalized_score = refnorm(score, scores_m_a, scores_m_b)
            normalized_scores.append(normalized_score)
    return np.array(normalized_scores)
