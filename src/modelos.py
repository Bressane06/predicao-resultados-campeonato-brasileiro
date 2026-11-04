# @Author: gabriel.bressane  # @Date: 2026-11-03 15:26:20  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-03 15:26:20 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from xgboost import XGBRegressor
from . import config as C


def criar_classificador():
    """Cria um pipeline de classificação utilizando Random Forest."""

    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("classificador", RandomForestClassifier(
            n_estimators=C.N_ESTIMATORS,
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])


def criar_baseline():
    """Cria um classificador de referência."""

    return DummyClassifier(strategy="most_frequent")


def criar_regressor():
    """Cria um pipeline de regressão utilizando Random Forest."""

    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(
            n_estimators=C.N_ESTIMATORS,
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])


def criar_baseline_regressao():
    """Cria um regressor de referência que prevê a média."""

    return DummyRegressor(strategy="mean")

def criar_baseline_probabilidades():
    return DummyClassifier(strategy="prior")

def criar_regressor():
    """
    Cria um pipeline de regressao utilizando Random Forest.

    O pipeline realiza:
    1. Preenchimento de valores ausentes pela mediana.
    2. Treinamento do Random Forest Regressor.
    """

    modelo = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),

        ("regressor", RandomForestRegressor(
            n_estimators=C.N_ESTIMATORS,
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])

    return modelo


def criar_baseline_regressao():
    """
    Cria um modelo de referencia que prevê
    a media dos valores do treinamento.
    """

    return DummyRegressor(strategy="mean")

def criar_regressor_poisson():
    return Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),
        ("escalonamento", StandardScaler()),
        ("regressor", PoissonRegressor(
            alpha=0.01,
            max_iter=5000
        ))
    ])

def criar_regressor_hist_gradient():
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