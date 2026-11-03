
"""Metricas de avaliacao dos modelos de classificacao."""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    log_loss
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
