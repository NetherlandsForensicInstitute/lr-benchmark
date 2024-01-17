from typing import List

import numpy as np


def refnorm(score: float, scores_m_a: List[float], scores_m_b: List[float]) -> float:
    """
    Performs the reference normalization on the score, based on the mean and standard deviation of the scores of
    both individual measurements with the refnorm measurements.
    """
    norm_a = (score - (sum(scores_m_a) / len(scores_m_a))) / np.std(scores_m_a)
    norm_b = (score - (sum(scores_m_b) / len(scores_m_b))) / np.std(scores_m_b)
    return round((norm_a + norm_b) / 2, 6)




