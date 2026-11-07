
"""Funcoes para criacao dos modelos de classificacao e regressao."""

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingRegressor
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from . import config as C


def criar_classificador():
    """Cria um classificador utilizando Random Forest."""

    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("classificador", RandomForestClassifier(
            n_estimators=C.N_ESTIMATORS,
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])


def criar_baseline():
    """Cria um classificador de referencia."""

    return DummyClassifier(strategy="most_frequent")


def criar_baseline_probabilidades():
    """Cria um baseline que utiliza as frequencias das classes."""

    return DummyClassifier(strategy="prior")


def criar_regressor():
    """Cria um regressor utilizando Random Forest."""

    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=C.N_ESTIMATORS,
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])


def criar_baseline_regressao():
    """Cria um regressor de referencia que prevê a media."""

    return DummyRegressor(strategy="mean")


def criar_regressor_poisson():
    """Cria um regressor utilizando distribuicao de Poisson."""

    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("escalonamento", StandardScaler()),
        ("regressor", PoissonRegressor(
            alpha=0.01,
            max_iter=10000
        ))
    ])


def criar_regressor_hist_gradient():
    """Cria um regressor utilizando HistGradientBoosting."""

    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("regressor", HistGradientBoostingRegressor(
            max_iter=500,
            learning_rate=0.01,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=C.SEED
        ))
    ])


def criar_regressor_xgboost():
    """Cria um regressor utilizando XGBoost."""

    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("regressor", XGBRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])
