"""Member 3 — classical ML: Logistic Regression, Random Forest, XGBoost.

Each builder returns an (imblearn) Pipeline: preprocessing -> [SMOTE] -> model,
so SMOTE is only ever fitted on training folds.
"""
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

from . import config as C
from .features import make_preprocessor
from .splits import group_cv

STRATEGIES = ["none", "class_weight", "smote"]


def _estimator(name: str, strategy: str, pos_weight: float):
    cw = "balanced" if strategy == "class_weight" else None
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw, C=1.0)
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced_subsample" if cw else None,
                                      random_state=C.SEED)
    if name == "xgboost":
        return XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.9,
                             colsample_bytree=0.8, tree_method="hist", n_jobs=2, eval_metric="aucpr",
                             scale_pos_weight=pos_weight if strategy == "class_weight" else 1.0,
                             random_state=C.SEED)
    raise ValueError(name)


def build(name: str, df, cols, strategy: str = "class_weight", pos_weight: float = 1.0):
    steps = [("prep", make_preprocessor(df, cols, scale=(name == "logreg")))]
    if strategy == "smote":
        steps.append(("smote", SMOTE(random_state=C.SEED)))
    steps.append(("model", _estimator(name, strategy, pos_weight)))
    return ImbPipeline(steps)


PARAM_GRIDS = {
    "logreg": {"model__C": np.logspace(-3, 2, 12)},
    "random_forest": {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 14, 20],
                      "model__min_samples_leaf": [1, 2, 5], "model__max_features": ["sqrt", 0.5]},
    "xgboost": {"model__max_depth": [3, 4, 5, 6, 8], "model__learning_rate": [0.03, 0.05, 0.1],
                "model__n_estimators": [200, 400, 700], "model__min_child_weight": [1, 3, 5],
                "model__subsample": [0.7, 0.9, 1.0], "model__colsample_bytree": [0.6, 0.8, 1.0]},
}


def tune(name, df, cols, X_train, y_train, groups_train, strategy="class_weight",
         pos_weight=1.0, n_iter=12):
    """Randomised search with user-grouped 5-fold CV, scored on PR-AUC."""
    pipe = build(name, df, cols, strategy, pos_weight)
    grid = PARAM_GRIDS[name]
    n = min(n_iter, int(np.prod([len(v) for v in grid.values()])))
    search = RandomizedSearchCV(pipe, grid, n_iter=n, scoring="average_precision",
                                cv=group_cv(5), random_state=C.SEED, n_jobs=1, refit=True)
    search.fit(X_train, y_train, groups=groups_train)
    return search.best_estimator_, search.best_params_, search.best_score_
