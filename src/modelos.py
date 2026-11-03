# @Author: gabriel.bressane  # @Date: 2026-11-03 15:26:20  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-03 15:26:20 
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier

from . import config as C


def criar_classificador():
    """
    Cria um pipeline de classificacao utilizando Random Forest.

    O pipeline realiza:
    1. Preenchimento de valores ausentes pela mediana.
    2. Treinamento do Random Forest.
    """

    modelo = Pipeline([
        ("imputacao", SimpleImputer(strategy="median")),

        ("classificador", RandomForestClassifier(
            n_estimators=C.N_ESTIMATORS,
            random_state=C.SEED,
            n_jobs=-1
        ))
    ])

    return modelo

def criar_baseline():
    """
    Cria um modelo de referencia que sempre prevê
    a classe mais frequente no conjunto de treinamento.
    """

    return DummyClassifier(strategy="most_frequent")