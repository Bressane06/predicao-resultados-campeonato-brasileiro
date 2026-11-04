"""Funcoes para salvar e carregar modelos."""

import joblib

from . import config as C


def salvar_modelo(modelo, caminho=None):
    """Salva um modelo treinado em disco."""

    caminho = caminho or C.ARQ_MODELO

    caminho.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(modelo, caminho)

    print(f"\nModelo salvo em: {caminho}")


def carregar_modelo(caminho=None):
    """Carrega um modelo previamente salvo."""

    caminho = caminho or C.ARQ_MODELO

    if not caminho.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {caminho}"
        )

    return joblib.load(caminho)