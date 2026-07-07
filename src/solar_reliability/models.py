from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from lightgbm import LGBMRegressor
from sklearn.impute import SimpleImputer


@dataclass
class QuantileModels:
    imputer: SimpleImputer
    ensemble: ExtraTreesRegressor
    alpha: float

    def predict(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed = self.imputer.transform(x)
        tree_predictions = np.vstack([tree.predict(transformed) for tree in self.ensemble.estimators_])
        lower = np.quantile(tree_predictions, self.alpha / 2, axis=0)
        median = np.quantile(tree_predictions, 0.5, axis=0)
        upper = np.quantile(tree_predictions, 1 - self.alpha / 2, axis=0)
        return lower, median, upper


def fit_quantile_models(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    alpha: float,
    params: dict,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> QuantileModels:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    transformed = imputer.fit_transform(x_train)
    ensemble = ExtraTreesRegressor(
        n_estimators=int(params.get("n_estimators", 80)),
        max_depth=int(params.get("max_depth", 18)),
        min_samples_leaf=int(params.get("min_samples_leaf", 5)),
        max_features=float(params.get("max_features", 0.8)),
        bootstrap=True,
        max_samples=float(params.get("max_samples", 0.8)),
        n_jobs=int(params.get("n_jobs", 2)),
        random_state=seed,
    )
    ensemble.fit(transformed, np.asarray(y_train, dtype=float), sample_weight=sample_weight)
    return QuantileModels(imputer=imputer, ensemble=ensemble, alpha=alpha)


@dataclass
class RampRiskModel:
    imputer: SimpleImputer
    classifier: ExtraTreesClassifier

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        transformed = self.imputer.transform(x)
        probabilities = self.classifier.predict_proba(transformed)
        if probabilities.shape[1] == 1:
            return np.zeros(len(x), dtype=float)
        return probabilities[:, 1]


def fit_ramp_risk_model(
    x_train: pd.DataFrame,
    ramp_labels: np.ndarray,
    params: dict,
    seed: int,
) -> RampRiskModel:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    transformed = imputer.fit_transform(x_train)
    classifier = ExtraTreesClassifier(
        n_estimators=int(params.get("risk_n_estimators", 80)),
        max_depth=int(params.get("risk_max_depth", 14)),
        min_samples_leaf=int(params.get("risk_min_samples_leaf", 8)),
        max_features=float(params.get("max_features", 0.8)),
        class_weight="balanced",
        bootstrap=True,
        max_samples=float(params.get("max_samples", 0.8)),
        n_jobs=int(params.get("n_jobs", 2)),
        random_state=seed,
    )
    classifier.fit(transformed, np.asarray(ramp_labels, dtype=int))
    return RampRiskModel(imputer=imputer, classifier=classifier)



@dataclass
class LightGBMQuantileModels:
    imputer: SimpleImputer
    lower_model: LGBMRegressor
    median_model: LGBMRegressor
    upper_model: LGBMRegressor

    def predict(self, x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        transformed = self.imputer.transform(x)
        lower = self.lower_model.predict(transformed)
        median = self.median_model.predict(transformed)
        upper = self.upper_model.predict(transformed)
        lo = np.minimum(lower, upper)
        hi = np.maximum(lower, upper)
        return lo, median, hi


def fit_hist_quantile_models(
    x_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    alpha: float,
    params: dict,
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> LightGBMQuantileModels:
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    transformed = imputer.fit_transform(x_train)
    common = dict(
        objective="quantile",
        n_estimators=int(params.get("lgbm_n_estimators", 120)),
        num_leaves=int(params.get("lgbm_num_leaves", 23)),
        learning_rate=float(params.get("lgbm_learning_rate", 0.06)),
        min_child_samples=int(params.get("lgbm_min_child_samples", 35)),
        reg_lambda=float(params.get("lgbm_reg_lambda", 1.0)),
        subsample=float(params.get("lgbm_subsample", 0.85)),
        colsample_bytree=float(params.get("lgbm_colsample_bytree", 0.85)),
        verbosity=-1,
        n_jobs=int(params.get("n_jobs", 2)),
        random_state=seed,
    )
    y_arr = np.asarray(y_train, dtype=float)
    lower_model = LGBMRegressor(alpha=alpha / 2, **common)
    median_model = LGBMRegressor(alpha=0.5, **common)
    upper_model = LGBMRegressor(alpha=1 - alpha / 2, **common)
    lower_model.fit(transformed, y_arr, sample_weight=sample_weight)
    median_model.fit(transformed, y_arr, sample_weight=sample_weight)
    upper_model.fit(transformed, y_arr, sample_weight=sample_weight)
    return LightGBMQuantileModels(imputer, lower_model, median_model, upper_model)
