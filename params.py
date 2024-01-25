from collections.abc import Sequence, Mapping
from functools import partial
from typing import List, Any, Union, Optional

from lir import LogitCalibrator, KDECalibrator, ELUBbounder, DummyLogOddsCalibrator, IsotonicCalibrator
from lir.transformers import PercentileRankTransformer, AbsDiffTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from lrbenchmark.data.dataset import XTCDataset, GlassDataset, ASRDataset
from lrbenchmark.data.simulation import SynthesizedNormalDataset
from lrbenchmark.evaluation import DescribedValue
from lrbenchmark.pairing import CartesianPairing, BalancedPairing
from lrbenchmark.transformers import DummyTransformer, PrecalculatedScorerASR, MeasurementPairScorer


def resolve_parameter(param: Optional[Union[str, int, Sequence, Mapping]], possible_params: Mapping,
                      desc: Optional[str] = None) -> Optional[Any]:
    if param is None:
        return None
    elif isinstance(param, Sequence) and not isinstance(param, str):
        return [DescribedValue(resolve_parameter(p, possible_params), desc=p) for p in param]
    elif isinstance(param, Mapping):
        return {k: resolve_parameter(v, possible_params, k) for k, v in param.items()}
    elif param in possible_params:
        return possible_params[param]()
    else:
        return param


def get_parameters(param: Union[str, Sequence, Mapping], possible_params: Mapping) -> List[DescribedValue]:
    if isinstance(param, str):
        return [DescribedValue(possible_params[param](), desc=param)]
    elif isinstance(param, Sequence):
        return [DescribedValue(possible_params[item](), desc=item) for item in param]
    elif isinstance(param, Mapping):
        alternatives = []
        for key, value in param.items():
            if value is None:
                alternatives.append(possible_params[key]())
            elif isinstance(value, Sequence):
                alternatives.append(possible_params[key](*resolve_parameter(value, possible_params)))
            elif isinstance(value, Mapping):
                alternatives.append(possible_params[key](**resolve_parameter(value, possible_params)))

        return alternatives


PAIRING = {'cartesian': CartesianPairing,
           'balanced': BalancedPairing}

SCORERS = {'precalculated': PrecalculatedScorerASR,
           'lda': partial(MeasurementPairScorer, AbsDiffTransformer, LDA),
           'qda': partial(MeasurementPairScorer, AbsDiffTransformer, QDA),
           'gb': partial(MeasurementPairScorer, AbsDiffTransformer, GradientBoostingClassifier),
           'rf': partial(MeasurementPairScorer, AbsDiffTransformer,
                         lambda: RandomForestClassifier(n_estimators=100, class_weight='balanced')),
           'logit': partial(MeasurementPairScorer, AbsDiffTransformer,
                            lambda: LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=500)),
           'xgb': partial(MeasurementPairScorer, AbsDiffTransformer,
                          lambda: XGBClassifier(eval_metric='error', use_label_encoder=False)),
           'rf_optim': partial(MeasurementPairScorer, AbsDiffTransformer,
                               lambda: RandomizedSearchCV(estimator=RandomForestClassifier(),
                                                          param_distributions={'bootstrap': [True, False],
                                                                               'max_depth': [10, 20, 30, 40, 50, 60, 70,
                                                                                             80, 90, 100, None],
                                                                               'max_features': ['auto', 'sqrt'],
                                                                               'min_samples_leaf': [1, 2, 4],
                                                                               'min_samples_split': [2, 5, 10],
                                                                               'n_estimators': [5, 10, 20, 50, 100]},
                                                             n_iter=100, cv=3))}

CALIBRATORS = {'logit': LogitCalibrator,
               'elub_logit': partial(ELUBbounder, first_step_calibrator=LogitCalibrator()),
               'kde': KDECalibrator,
               'dummy': DummyLogOddsCalibrator,
               'isotonic': IsotonicCalibrator}

DATASETS = {'asr': ASRDataset,
            'drugs_xtc': XTCDataset,
            'glass': GlassDataset,
            'synthesized_normal': SynthesizedNormalDataset}


def create_pipeline(**params):
    return Pipeline([(k, v) for k, v in params.items()])


PREPROCESSORS = {'dummy': DummyTransformer,
                 'rank_transformer': PercentileRankTransformer,
                 'abs_diff': AbsDiffTransformer,
                 'pipeline': create_pipeline, }
