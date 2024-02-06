from functools import partial
from pathlib import Path
from typing import Any, Union, Optional, Iterable, MutableMapping, Callable, Mapping

from lir import LogitCalibrator, KDECalibrator, ELUBbounder, DummyLogOddsCalibrator, IsotonicCalibrator
from lir.transformers import PercentileRankTransformer, AbsDiffTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from lrbenchmark.data.dataset import XTCDataset, GlassDataset, ASRDataset
from lrbenchmark.data.simulation import SynthesizedNormalDataset
from lrbenchmark.pairing import CartesianPairing, BalancedPairing
from lrbenchmark.transformers import DummyTransformer, PrecalculatedScorerASR, MeasurementPairScorer

PAIRING = {'cartesian': CartesianPairing, 'balanced': BalancedPairing}

SCORERS = {'precalculated_asr': PrecalculatedScorerASR,
           'lda': partial(MeasurementPairScorer, LDA),
           'qda': partial(MeasurementPairScorer, QDA),
           'gb': partial(MeasurementPairScorer, GradientBoostingClassifier),
           'rf': partial(MeasurementPairScorer,
                         lambda: RandomForestClassifier(
                             n_estimators=100,
                             class_weight='balanced')),
           'logit': partial(MeasurementPairScorer,
                            lambda: LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=500)),
           'xgb': partial(MeasurementPairScorer, lambda: XGBClassifier(eval_metric='error', use_label_encoder=False)),
           'rf_optim': partial(MeasurementPairScorer, lambda: RandomizedSearchCV(estimator=RandomForestClassifier(),
                                                                                 param_distributions={
                                                                                     'bootstrap': [True, False],
                                                                                     'max_depth': [10, 20, 30, 40, 50,
                                                                                                   60, 70, 80, 90, 100,
                                                                                                   None],
                                                                                     'max_features': ['auto', 'sqrt'],
                                                                                     'min_samples_leaf': [1, 2, 4],
                                                                                     'min_samples_split': [2, 5, 10],
                                                                                     'n_estimators': [5, 10, 20, 50,
                                                                                                      100]}, n_iter=100,
                                                                                 cv=3))}

CALIBRATORS = {'logit': LogitCalibrator,
               'elub_logit': partial(ELUBbounder, first_step_calibrator=LogitCalibrator()),
               'kde': KDECalibrator,
               'dummy': DummyLogOddsCalibrator,
               'isotonic': IsotonicCalibrator}

DATASETS = {'asr': ASRDataset,
            'xtc': XTCDataset,
            'glass': GlassDataset,
            'synthesized_normal': SynthesizedNormalDataset}

PREPROCESSORS = {'dummy': DummyTransformer,
                 'rank_transformer': PercentileRankTransformer,
                 'abs_diff': AbsDiffTransformer}



config_option_dicts = {'scorer': SCORERS,
                       'pairing': PAIRING,
                       'preprocessors': PREPROCESSORS,
                       'dataset': DATASETS,
                       'calibrator': CALIBRATORS}


def parse_config(config: Union[str, Path, Mapping[str, Any]],
                 parsers: Mapping[str, Mapping[Optional[str], Callable]]) -> MutableMapping[str, Any]:
    """
    Recursively parse a `config` mapping consisting of serialized values of
    built-in types (e.g. `str`, `int`, `list`, etc.) and deserialize them by
    applying the appropriate callbacks in `parsers`.

    This function iterates over each `(key, value)` pair in `config`. If the
    `key` matches a `key` in `parsers` and `value` is itself a `Mapping`, the
    corresponding callback in `parsers` (i.e. `parsers[key]`) is used to
    instantiate a Python object from the serialized items in `value`.

    :param config: Mapping[str, Any]
    :param parsers: Optional[Mapping[str, Factory]]
    :return: MutableMapping[str, Any]
    """

    def parse_item(key: str, value: Any) -> Any:
        if isinstance(value, Mapping):
            value = dict(value)
            name = value.pop('name', None)
            value = parse_config(value, parsers)
            if parsers and key in parsers and name in parsers[key]:
                return parsers[key][name](**value)
            return value

        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            return [parse_item(key, v) for v in value]

        return value

    return {k: parse_item(k, v) for k, v in config.items()}
