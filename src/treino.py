
"""Treinamento e avaliacao inicial dos modelos."""

import pandas as pd

from . import config as C
from .modelos import criar_classificador, criar_baseline
from .avaliacao import avaliar_classificacao


def carregar_base():
    """Carrega a base de modelagem."""

    if not C.ARQ_BASE.exists():
        raise FileNotFoundError(
            f"Base de modelagem nao encontrada: {C.ARQ_BASE}"
        )

    return pd.read_csv(C.ARQ_BASE)


def preparar_dados(base):
    """Separa os atributos (X) e o resultado (y)."""

    base = base.dropna(subset=["resultado"]).copy()

    base["resultado"] = (
        base["resultado"].astype(str).str.strip().str.upper()
    )

    ignorar = {
        "ID",
        "id",
        "data",
        "gm",
        "gv",
        "resultado"
    }

    colunas = [
        coluna
        for coluna in base.columns
        if coluna not in ignorar
        and pd.api.types.is_numeric_dtype(base[coluna])
    ]

    if not colunas:
        raise ValueError("Nenhum atributo numerico encontrado.")

    X = base[colunas]
    y = base["resultado"]

    return X, y


def dividir_dados(X, y, base):
    """Realiza a divisao cronologica entre treino e teste."""

    datas = pd.to_datetime(base.loc[X.index, "data"], errors="coerce")

    if datas.isna().any():
        raise ValueError("Existem datas invalidas na base.")

    ordem = datas.sort_values().index

    X = X.loc[ordem]
    y = y.loc[ordem]

    limite = int(len(X) * C.PROPORCAO_TREINO)

    X_treino = X.iloc[:limite]
    X_teste = X.iloc[limite:]

    y_treino = y.iloc[:limite]
    y_teste = y.iloc[limite:]

    return X_treino, X_teste, y_treino, y_teste


def exibir_resultados(nome, metricas, por_classe, matriz):
    """Exibe as metricas e a matriz de confusao."""

    print(f"\n===== {nome} =====")

    print("\nMetricas gerais:")

    for nome_metrica, valor in metricas.items():
        print(f"{nome_metrica}: {valor:.4f}")

    print("\nMetricas por classe:")
    print(por_classe.to_string(index=False, float_format="%.4f"))

    print("\nMatriz de confusao:")
    print(matriz)


def main():

    # 1. Carregamento da base.
    base = carregar_base()

    print(f"Partidas carregadas: {len(base)}")

    # 2. Preparacao dos atributos e resultados.
    X, y = preparar_dados(base)

    print(f"Quantidade de atributos: {X.shape[1]}")

    # 3. Divisao cronologica.
    X_treino, X_teste, y_treino, y_teste = dividir_dados(
        X, y, base
    )

    print(f"Partidas para treino: {len(X_treino)}")
    print(f"Partidas para teste: {len(X_teste)}")

    print("\nDistribuicao das classes no treinamento:")
    print(y_treino.value_counts())

    # 4. Treinamento do Random Forest.
    modelo = criar_classificador()

    print("\nTreinando Random Forest...")
    modelo.fit(X_treino, y_treino)

    previsoes_rf = modelo.predict(X_teste)

    # 5. Treinamento do baseline.
    baseline = criar_baseline()

    print("Treinando baseline...")
    baseline.fit(X_treino, y_treino)

    previsoes_baseline = baseline.predict(X_teste)

    print(
        f"Classe prevista pelo baseline: "
        f"{baseline.classes_[baseline.class_prior_.argmax()]}"
    )

    # 6. Avaliacao do Random Forest.
    metricas_rf, classes_rf, matriz_rf = avaliar_classificacao(
        y_teste,
        previsoes_rf
    )

    exibir_resultados(
        "RANDOM FOREST",
        metricas_rf,
        classes_rf,
        matriz_rf
    )

    # 7. Avaliacao do baseline.
    metricas_base, classes_base, matriz_base = avaliar_classificacao(
        y_teste,
        previsoes_baseline
    )

    exibir_resultados(
        "BASELINE",
        metricas_base,
        classes_base,
        matriz_base
    )


if __name__ == "__main__":
    main()
