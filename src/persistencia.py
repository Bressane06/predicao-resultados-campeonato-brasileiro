
"""Funcoes para salvar e carregar modelos."""

import joblib

from . import config as C


def salvar_modelo(modelo):
    """Salva o modelo treinado em disco."""

    C.ARQ_MODELO.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(modelo, C.ARQ_MODELO)

    print(f"\nModelo salvo em: {C.ARQ_MODELO}")


def carregar_modelo():
    """Carrega um modelo previamente salvo."""

    if not C.ARQ_MODELO.exists():
        raise FileNotFoundError(
            f"Modelo nao encontrado: {C.ARQ_MODELO}"
        )

    return joblib.load(C.ARQ_MODELO)
