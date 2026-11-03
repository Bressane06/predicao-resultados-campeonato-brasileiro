# @Author: gabriel.bressane  # @Date: 2026-11-03 15:34:59  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-03 15:34:59 
"""Metricas de avaliacao dos modelos de classificacao."""

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support
)

from . import config as CONFIGURACAO


def avaliar_classificacao(y_real, y_previsto):
    """
    Calcula as metricas gerais e individuais da classificacao.

    Retorna:
        metricas: dicionario com as metricas gerais.
        por_classe: DataFrame com metricas de cada classe.
        matriz: matriz de confusao.
    """

    # Metricas individuais de cada classe.
    precisao, recall, f1, suporte = precision_recall_fscore_support(
        y_real,
        y_previsto,
        labels=CONFIGURACAO.CLASSES,
        zero_division=0
    )

    # Metricas gerais.
    metricas = {
        "Acuracia": accuracy_score(y_real, y_previsto),
        "Precisao (macro)": precision_recall_fscore_support(
            y_real, y_previsto,
            labels=CONFIGURACAO.CLASSES,
            average="macro",
            zero_division=0
        )[0],
        "Recall (macro)": precision_recall_fscore_support(
            y_real, y_previsto,
            labels=CONFIGURACAO.CLASSES,
            average="macro",
            zero_division=0
        )[1],
        "F1-score (macro)": precision_recall_fscore_support(
            y_real, y_previsto,
            labels=CONFIGURACAO.CLASSES,
            average="macro",
            zero_division=0
        )[2]
    }

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
