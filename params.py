from collections.abc import Sequence, Mapping
from typing import List

from lir.calibration import *
from lir.transformers import PercentileRankTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA, LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from lrbenchmark.dataset import XTCDataset, GlassDataset
from lrbenchmark.evaluation import DescribedValue
from lrbenchmark.transformers import DummyTransformer


def get_parameters(param: Union[str, Sequence, Mapping], possible_params: Mapping) -> List[DescribedValue]:
    if isinstance(param, str):
        return [DescribedValue(possible_params[param], desc=param)]
    elif isinstance(param, Sequence):
        return [DescribedValue(possible_params[item], desc=item) for item in param]
    elif isinstance(param, Mapping):
        return [DescribedValue(possible_params[key](**value)) for key, value in param.items()]


SCORERS = {
    'LDA': LDA(),
    'QDA': QDA(),
    'GB': GradientBoostingClassifier(),
    'RF': RandomForestClassifier(n_estimators=100),
    'RF_weighted': RandomForestClassifier(n_estimators=100, class_weight='balanced'),
    'LR': LogisticRegression(solver='liblinear', max_iter=500),
    'LR_weighted': LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=500),
    'XGB': XGBClassifier(eval_metric='error', use_label_encoder=False),
    'RF_optim': RandomizedSearchCV(estimator=RandomForestClassifier(),
                                   param_distributions={'bootstrap': [True, False],
                                                        'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
                                                                      None],
                                                        'max_features': ['auto', 'sqrt'],
                                                        'min_samples_leaf': [1, 2, 4],
                                                        'min_samples_split': [2, 5, 10],
                                                        'n_estimators': [5, 10, 20, 50, 100]}, n_iter=100, cv=3)
}

CALIBRATORS = {
    'logit': LogitCalibrator(),
    'logit_normalized': NormalizedCalibrator(LogitCalibrator()),
    'logit_unweighted': LogitCalibrator(),
    'KDE': KDECalibrator(),
    'elub_KDE': ELUBbounder(KDECalibrator()),
    'elub': ELUBbounder(DummyCalibrator()),
    'dummy': DummyCalibrator(),
    'fraction': FractionCalibrator(),
    'isotonic': IsotonicCalibrator()
}

DATASETS = {
    'drugs_xtc': XTCDataset,
    'glass': GlassDataset
}

PREPROCESSORS = {
    'dummy': DummyTransformer(),
    'rank_transformer': PercentileRankTransformer()
}
