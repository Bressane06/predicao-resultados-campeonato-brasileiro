# @Author: gabriel.bressane  # @Date: 2026-11-14 01:07:59  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-14 01:07:59 
"""Funcoes para salvar e carregar modelos."""

import joblib

from . import config as C


def salvar_modelo(modelo, caminho=None):
    """
    Salva um modelo treinado em disco.

    Args:
        modelo: Modelo treinado que sera persistido.
        caminho: Caminho do arquivo. Utiliza o caminho padrao
                 definido em config.py quando nao informado.
    """

    caminho = caminho or C.ARQ_MODELO

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(modelo, caminho)

    print(f"\nModelo salvo em: {caminho}")


def carregar_modelo(caminho=None):
    """
    Carrega um modelo previamente salvo.

    Args:
        caminho: Caminho do arquivo do modelo.
                 Utiliza o caminho padrao definido em config.py
                 quando nao informado.

    Returns:
        Modelo carregado.
    """

    caminho = caminho or C.ARQ_MODELO

    if not caminho.is_file():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {caminho}"
        )

    return joblib.load(caminho)
