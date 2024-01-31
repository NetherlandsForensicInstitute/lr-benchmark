import os
from pathlib import Path

import confidence
import numpy as np

from lrbenchmark.data.dataset import ASRDataset
from lrbenchmark.pairing import CartesianPairing
from lrbenchmark.refnorm import perform_refnorm
from lrbenchmark.transformers import PrecalculatedScorerASR
from tests.conftest import TEST_DIR, ROOT_DIR


def test_refnorm():
    config = confidence.load_name(str(TEST_DIR / 'lrbenchmark_test'))
    asr_config = config.dataset_test.asr
    # read the file containing filenames of the refnorm cohort
    file = open(TEST_DIR / "test_resources/refnormcohort.txt", "r")
    filenames_refnorm = file.read().split('\n')
    filenames_refnorm = list(map(lambda x: x.replace("_30s", ""), filenames_refnorm))
    # create the refnorm dataset by filtering on filename
    dataset_refnorm = ASRDataset(scores_path=ROOT_DIR / asr_config.scores_path,
                                 meta_info_path=ROOT_DIR / asr_config.meta_info_path,
                                 source_filter={'filename': filenames_refnorm})
    # retrieve the raw scores that need to be normalized
    scorer_raw = PrecalculatedScorerASR(scores_path=ROOT_DIR / asr_config.scores_path)
    # retrieve the normalized scores from csv
    scorer_normalized = PrecalculatedScorerASR(TEST_DIR / "test_resources/unittestrefnorm_normalizedscores.csv")
    # create a dataset and pair all measurements
    dataset_asr = ASRDataset(scores_path=TEST_DIR / "test_resources/unittestrefnorm_rawscores.csv",
                             meta_info_path=ROOT_DIR / asr_config.meta_info_path)
    pairs = dataset_asr.get_pairs(pairing_function=CartesianPairing())
    # retrieve the raw and normalized scores for every pair
    raw_scores = scorer_raw.fit_predict(pairs)
    normalized_scores_expected = scorer_normalized.fit_predict(pairs)
    # perform reference normalization
    normalized_scores_actual = perform_refnorm(raw_scores, pairs, dataset_refnorm, scorer_raw)
    assert np.all(np.abs(normalized_scores_expected-normalized_scores_actual) < 0.09)
