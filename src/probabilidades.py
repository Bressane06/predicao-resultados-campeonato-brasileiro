# @Author: gabriel.bressane  # @Date: 2026-11-14 01:08:03  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-14 01:08:03 
import numpy as np
import pandas as pd
from scipy.stats import poisson


def calcular_matriz_placares(
    gols_mandante,
    gols_visitante,
    max_gols=10,
    rho=0.0
):
    """
    Calcula a matriz de probabilidades dos placares
    utilizando Poisson com ajuste Dixon-Coles.
    """

    gols = np.arange(max_gols + 1)

    prob_mandante = poisson.pmf(
        gols,
        mu=gols_mandante
    )

    prob_visitante = poisson.pmf(
        gols,
        mu=gols_visitante
    )

    matriz = np.outer(
        prob_mandante,
        prob_visitante
    )

    # Ajuste Dixon-Coles nos placares baixos.
    matriz[0, 0] *= (
        1 - gols_mandante * gols_visitante * rho
    )

    matriz[0, 1] *= (
        1 + gols_mandante * rho
    )

    matriz[1, 0] *= (
        1 + gols_visitante * rho
    )

    matriz[1, 1] *= (
        1 - rho
    )

    # Evita probabilidades inválidas.
    if np.any(matriz < 0):
        raise ValueError(
            "O valor de rho produziu probabilidades negativas."
        )

    # Normaliza a matriz.
    matriz = matriz / matriz.sum()

    placares = pd.DataFrame(
        matriz,
        index=gols,
        columns=gols
    )

    placares.index.name = "GM"
    placares.columns.name = "GV"

    return placares

def calcular_probabilidades_resultado(placares):
    """
    Agrupa os placares em vitória, empate e derrota
    do mandante.
    """

    matriz = placares.to_numpy()

    vitoria_mandante = np.tril(matriz, k=-1).sum()
    empate = np.trace(matriz)
    vitoria_visitante = np.triu(matriz, k=1).sum()

    return {
        "M": vitoria_mandante,
        "E": empate,
        "V": vitoria_visitante
    }