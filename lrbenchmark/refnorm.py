from typing import List

import numpy as np


def refnorm(score: float, scores_m_a: List[float], scores_m_b: List[float]) -> float:
    """
    Performs the reference normalization on the score, based on the mean and standard deviation of the scores of
    both individual measurements with the refnorm measurements.
    """
    norm1 = (score - (sum(scores_m_a) / len(scores_m_a))) / np.std(scores_m_a)
    norm2 = (score - (sum(scores_m_b) / len(scores_m_b))) / np.std(scores_m_b)
    return round((norm1 + norm2) / 2, 6)




