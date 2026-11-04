
"""Metricas de avaliacao dos modelos de classificacao."""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    log_loss,
    accuracy_score
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from . import config as CONFIGURACAO


def avaliar_classificacao(
    y_real,
    y_previsto,
    probabilidades=None,
    classes=None
):
    """
    Calcula as metricas gerais e individuais da classificacao.
    """

    # Metricas individuais por classe.
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_real,
        y_previsto,
        labels=CONFIGURACAO.CLASSES,
        zero_division=0
    )

    # Metricas gerais.
    precisao_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            y_real,
            y_previsto,
            labels=CONFIGURACAO.CLASSES,
            average="macro",
            zero_division=0
        )
    )

    metricas = {
        "Acuracia": accuracy_score(y_real, y_previsto),
        "Precisao (macro)": precisao_macro,
        "Recall (macro)": recall_macro,
        "F1-score (macro)": f1_macro
    }

    # Log Loss, quando as probabilidades estiverem disponiveis.
    if probabilidades is not None:
        metricas["Log Loss"] = log_loss(
            y_real,
            probabilidades,
            labels=classes
        )

    # Organiza as metricas por classe.
    por_classe = pd.DataFrame({
        "Classe": CONFIGURACAO.CLASSES,
        "Precisao": precisao,
        "Recall": recall,
        "F1-score": f1,
        "Suporte": suporte
    })

    # Matriz de confusao.
    matriz = confusion_matrix(
        y_real,
        y_previsto,
        labels=CONFIGURACAO.CLASSES
    )

    matriz = pd.DataFrame(
        matriz,
        index=[f"Real: {c}" for c in CONFIGURACAO.CLASSES],
        columns=[f"Previsto: {c}" for c in CONFIGURACAO.CLASSES]
    )

    return metricas, por_classe, matriz

def avaliar_regressao(y_real, y_previsto):
    """
    Calcula as metricas de avaliacao da regressao.
    """

    metricas = {
        "MAE": mean_absolute_error(y_real, y_previsto),
        "RMSE": mean_squared_error(y_real, y_previsto) ** 0.5,
        "R2": r2_score(y_real, y_previsto)
    }

    return metricas

def avaliar_probabilidades(y_real, probabilidades):
    """Avalia a qualidade das probabilidades de classificacao."""

    # Ordem original das colunas: M, E, V.
    classes = ["M", "E", "V"]

    y_binario = pd.get_dummies(y_real).reindex(
        columns=classes,
        fill_value=0
    ).to_numpy()

    previsoes = np.argmax(probabilidades, axis=1)

    # O sklearn exige a ordem lexicografica: E, M, V.
    probabilidades_log_loss = probabilidades[:, [1, 0, 2]]

    return {
        "Log Loss": log_loss(
            y_real,
            probabilidades_log_loss,
            labels=["E", "M", "V"]
        ),
        "Brier Score": np.mean(
            np.sum((probabilidades - y_binario) ** 2, axis=1)
        ),
        "Acuracia": accuracy_score(
            y_real,
            [classes[i] for i in previsoes]
        )
    }