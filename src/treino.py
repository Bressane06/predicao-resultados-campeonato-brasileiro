
"""Treinamento e avaliacao inicial dos modelos."""

import pandas as pd

from .persistencia import salvar_modelo
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
    """Separa os dados cronologicamente em treino, validacao e teste."""

    datas = pd.to_datetime(
        base.loc[X.index, "data"],
        errors="coerce"
    )

    if datas.isna().any():
        raise ValueError("Existem datas invalidas na base.")

    # Ordena as partidas cronologicamente.
    ordem = datas.sort_values(kind="stable").index

    X = X.loc[ordem]
    y = y.loc[ordem]
    anos = datas.loc[ordem].dt.year

    # Identifica os indices de cada conjunto.
    indices_treino = anos[
        anos <= C.ANO_FIM_TREINO
    ].index

    indices_validacao = anos[
        anos == C.ANO_VALIDACAO
    ].index

    indices_teste = anos[
        anos >= C.ANO_INICIO_TESTE
    ].index

    # Verifica se todos os conjuntos possuem dados.
    if (
        len(indices_treino) == 0
        or len(indices_validacao) == 0
        or len(indices_teste) == 0
    ):
        raise ValueError(
            "Um dos conjuntos de dados esta vazio."
        )

    X_treino = X.loc[indices_treino]
    X_validacao = X.loc[indices_validacao]
    X_teste = X.loc[indices_teste]

    y_treino = y.loc[indices_treino]
    y_validacao = y.loc[indices_validacao]
    y_teste = y.loc[indices_teste]

    return (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste
    )


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

    datas = pd.to_datetime(base["data"], errors="coerce")

    print("Primeira partida:", datas.min())
    print("Última partida:", datas.max())

    print("\nQuantidade de partidas por ano:")
    print(datas.dt.year.value_counts().sort_index())

    print(f"Partidas carregadas: {len(base)}")

    # 2. Preparacao dos atributos e resultados.
    X, y = preparar_dados(base)

    print(f"Quantidade de atributos: {X.shape[1]}")

    # 3. Divisao cronologica.
    (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste
    ) = dividir_dados(X, y, base)

    print(f"Partidas para treino: {len(X_treino)}")
    print(f"Partidas para validacao: {len(X_validacao)}")
    print(f"Partidas para teste: {len(X_teste)}")

    print("\nDistribuicao das classes no treinamento:")
    print(y_treino.value_counts())

    modelo = criar_classificador()

    print("\nTreinando Random Forest...")
    modelo.fit(X_treino, y_treino)

    # Previsoes para validacao.
    previsoes_validacao = modelo.predict(X_validacao)

    # Previsoes para teste.
    previsoes_teste = modelo.predict(X_teste)

    # Baseline.
    baseline = criar_baseline()
    baseline.fit(X_treino, y_treino)
    salvar_modelo(modelo)

    previsoes_baseline_validacao = baseline.predict(X_validacao)
    previsoes_baseline_teste = baseline.predict(X_teste)

    # Probabilidades do Random Forest.
    probabilidades_validacao = modelo.predict_proba(X_validacao)
    probabilidades_teste = modelo.predict_proba(X_teste)

    # Probabilidades do baseline.
    probabilidades_baseline_validacao = baseline.predict_proba(
        X_validacao
    )

    probabilidades_baseline_teste = baseline.predict_proba(
        X_teste
    )

    # 6. Avaliacao dos modelos na validacao.

    metricas_rf, classes_rf, matriz_rf = avaliar_classificacao(
        y_validacao,
        previsoes_validacao,
        probabilidades_validacao,
        modelo.classes_
    )

    exibir_resultados(
        "RANDOM FOREST - VALIDACAO",
        metricas_rf,
        classes_rf,
        matriz_rf
    )

    metricas_base, classes_base, matriz_base = avaliar_classificacao(
        y_validacao,
        previsoes_baseline_validacao,
        probabilidades_baseline_validacao,
        baseline.classes_
    )

    exibir_resultados(
        "BASELINE - VALIDACAO",
        metricas_base,
        classes_base,
        matriz_base
    )

    # 7. Avaliacao dos modelos no teste.

    metricas_rf, classes_rf, matriz_rf = avaliar_classificacao(
        y_teste,
        previsoes_teste,
        probabilidades_teste,
        modelo.classes_
    )

    exibir_resultados(
        "RANDOM FOREST - TESTE",
        metricas_rf,
        classes_rf,
        matriz_rf
    )

    metricas_base, classes_base, matriz_base = avaliar_classificacao(
        y_teste,
        previsoes_baseline_teste,
        probabilidades_baseline_teste,
        baseline.classes_
    )

    exibir_resultados(
        "BASELINE - TESTE",
        metricas_base,
        classes_base,
        matriz_base
    )
    print("\nDistribuicao das classes:")

    print("\nTreinamento:")
    print(y_treino.value_counts())

    print("\nValidacao:")
    print(y_validacao.value_counts())

    print("\nTeste:")
    print(y_teste.value_counts())


if __name__ == "__main__":
    main()
