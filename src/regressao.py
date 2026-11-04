
"""Treinamento e avaliacao dos modelos de regressao."""

import pandas as pd

from .probabilidades import (
    calcular_matriz_placares,
    calcular_probabilidades_resultado,
)
from . import config as C
from .modelos import (
    criar_regressor,
    criar_baseline_regressao,
    criar_regressor_poisson,
    criar_regressor_hist_gradient,
    criar_regressor_xgboost,
    criar_baseline_probabilidades
)
from .persistencia import salvar_modelo
from .avaliacao import avaliar_regressao, avaliar_probabilidades
from .treino import carregar_base, dividir_dados

def avaliar_baseline_classificacao(base, probabilidades_poisson):
    """Compara o baseline de classificacao com as probabilidades de Poisson."""

    base = base.copy()
    base["resultado"] = base["resultado"].astype(str)

    base = base.dropna(subset=["resultado"])

    # Utiliza os mesmos atributos numericos dos modelos de regressao.
    ignorar = {
        "ID", "id", "data", "gm", "gv", "resultado"
    }

    colunas = [
        coluna
        for coluna in base.columns
        if coluna not in ignorar
        and pd.api.types.is_numeric_dtype(base[coluna])
    ]

    X = base[colunas]
    y = base["resultado"]

    # Separacao cronologica dos dados.
    (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste
    ) = dividir_dados(X, y, base)

    # Treinamento do baseline.
    modelo = criar_baseline_probabilidades()
    modelo.fit(X_treino, y_treino)

    # Probabilidades previstas na validacao.
    probabilidades = modelo.predict_proba(X_validacao)

    # Organiza as colunas na ordem M, E, V.
    probabilidades = pd.DataFrame(
        probabilidades,
        columns=modelo.classes_,
        index=X_validacao.index
    ).reindex(
        columns=C.CLASSES,
        fill_value=0
    )

    # Alinha os dois modelos nas mesmas partidas.
    probabilidades_poisson = probabilidades_poisson.set_index("ID")

    indices = probabilidades.index.intersection(
        probabilidades_poisson.index
    )

    # Avalia ambos com exatamente as mesmas partidas.
    reais = probabilidades_poisson.loc[
        indices, "resultado_real"
    ]

    metricas_baseline = avaliar_probabilidades(
        reais,
        probabilidades.loc[indices].to_numpy()
    )

    metricas_poisson = avaliar_probabilidades(
        reais,
        probabilidades_poisson.loc[
            indices, ["P_M", "P_E", "P_V"]
        ].to_numpy()
    )

    print("\n===== COMPARACAO: BASELINE VS POISSON =====")
    print(f"Partidas avaliadas: {len(indices)}")

    comparacao = pd.DataFrame({
        "Baseline": metricas_baseline,
        "Poisson": metricas_poisson
    })

    print(comparacao.round(4))

    return comparacao

def gerar_probabilidades_partidas(base, resultado_gm, resultado_gv):
    """Combina as previsoes de GM e GV para calcular probabilidades."""

    # Identifica partidas presentes nas duas validacoes.
    indices = resultado_gm["previsoes"].index.intersection(
        resultado_gv["previsoes"].index
    )

    previsoes_gm = resultado_gm["previsoes"].loc[indices]
    previsoes_gv = resultado_gv["previsoes"].loc[indices]

    partidas = []

    for indice in indices:

        # Medias previstas de gols.
        lambda_mandante = max(0, previsoes_gm.loc[indice])
        lambda_visitante = max(0, previsoes_gv.loc[indice])

        # Distribuicao de probabilidades dos placares.
        placares = calcular_matriz_placares(
            gols_mandante=lambda_mandante,
            gols_visitante=lambda_visitante,
            rho=-0.05
        )

        # Probabilidades de M, E e V.
        probabilidades = calcular_probabilidades_resultado(placares)

        # Placar individual mais provavel.
        melhor_placar = placares.stack().idxmax()

        partidas.append({
            "ID": indice,
            "GM_previsto": lambda_mandante,
            "GV_previsto": lambda_visitante,
            "placar_previsto": f"{melhor_placar[0]} x {melhor_placar[1]}",
            "P_M": probabilidades["M"],
            "P_E": probabilidades["E"],
            "P_V": probabilidades["V"],
            "resultado_real": base.loc[indice, "resultado"]
        })

    return pd.DataFrame(partidas)
def analisar_erros(y_real, y_previsto, nome):
    resultado = pd.DataFrame({
        "real": y_real,
        "previsto": y_previsto
    })

    resultado["erro_absoluto"] = (
        resultado["real"] - resultado["previsto"]
    ).abs()

    print(f"\nANALISE DE ERROS: {nome}")
    print(
        resultado.groupby("real")["erro_absoluto"]
        .agg(["count", "mean", "median"])
        .round(3)
    )

def analisar_vies(y_real, y_previsto, nome):
    resultado = pd.DataFrame({
        "real": y_real,
        "previsto": y_previsto
    })

    resultado["erro"] = (
        resultado["previsto"] - resultado["real"]
    )

    print(f"\nANALISE DE VIES: {nome}")

    print(
        resultado.groupby("real")["erro"]
        .agg(["count", "mean"])
        .round(3)
    )

def preparar_dados_regressao(base, alvo):
    """Separa os atributos (X) e o alvo numerico (y)."""

    if alvo not in ("gm", "gv"):
        raise ValueError("O alvo deve ser 'gm' ou 'gv'.")

    base = base.copy()
    base[alvo] = pd.to_numeric(base[alvo], errors="coerce")
    base = base.dropna(subset=[alvo])

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
    y = base[alvo]

    return X, y, base


def exibir_resultados(nome, metricas):
    """Exibe as metricas de avaliacao."""

    print(f"\n===== {nome} =====")

    for nome_metrica, valor in metricas.items():
        print(f"{nome_metrica}: {valor:.4f}")

def analisar_zeros_e_vitorias_visitantes(base, probabilidades):
    analise = probabilidades.merge(
        base[["ID", "gm", "gv"]],
        on="ID",
        how="left"
    )

    # Análise dos placares 0 a 0
    zeros = analise[
        (analise["gm"] == 0) & (analise["gv"] == 0)
    ]

    print("\n===== ANÁLISE DOS PLACARES 0 A 0 =====")
    print("Partidas reais:", len(zeros))
    print(
        zeros[
            ["ID", "GM_previsto", "GV_previsto",
             "placar_previsto"]
        ].to_string(index=False)
    )

    # Análise das vitórias visitantes
    visitantes = analise[
        analise["resultado_real"] == "V"
    ]

    acertos = visitantes[
        visitantes["P_V"] >= visitantes[["P_M", "P_E"]].max(axis=1)
    ]

    print("\n===== ANÁLISE DAS VITÓRIAS VISITANTES =====")
    print("Vitórias visitantes reais:", len(visitantes))
    print("Vitórias visitantes previstas:", len(acertos))
    print(
        "Taxa de acerto:",
        round(len(acertos) / len(visitantes), 4)
        if len(visitantes) else 0
    )

def treinar_modelo(base, alvo, caminho):
    """Treina, avalia e salva um modelo de regressao."""

    X, y, base_filtrada = preparar_dados_regressao(base, alvo)

    (
        X_treino,
        X_validacao,
        X_teste,
        y_treino,
        y_validacao,
        y_teste
    ) = dividir_dados(X, y, base_filtrada)

    print(f"\n{'=' * 50}")
    print(f"PREVISAO: {alvo.upper()}")
    print(f"{'=' * 50}")

    print(f"Partidas para treino: {len(X_treino)}")
    print(f"Partidas para validacao: {len(X_validacao)}")
    print(f"Partidas para teste: {len(X_teste)}")

    # Treinamento do Random Forest.
    modelo = criar_regressor()
    modelo.fit(X_treino, y_treino)

    # Treinamento do baseline.
    baseline = criar_baseline_regressao()
    baseline.fit(X_treino, y_treino)

    # Avaliacao na validacao.
    previsoes_validacao = modelo.predict(X_validacao)
    previsoes_baseline_validacao = baseline.predict(X_validacao)

    # Previsoes na validacao
    previstos_validacao = modelo.predict(X_validacao)

    # Analise dos erros
    analisar_erros(
        y_validacao,
        previstos_validacao,
        alvo.upper()
    )
    analisar_vies(
        y_validacao,
        previstos_validacao,
        alvo.upper()
    )

    modelo_poisson = criar_regressor_poisson()

    modelo_poisson.fit(X_treino, y_treino)

    # Validacao
    previstos_validacao_poisson = modelo_poisson.predict(
        X_validacao
    )

    metricas_validacao_poisson = avaliar_regressao(
        y_validacao,
        previstos_validacao_poisson
    )

    print(f"\n===== POISSON REGRESSOR - {alvo.upper()} - VALIDACAO =====")

    for metrica, valor in metricas_validacao_poisson.items():
        print(f"{metrica}: {valor:.4f}")

    # Teste
    previstos_teste_poisson = modelo_poisson.predict(X_teste)

    metricas_teste_poisson = avaliar_regressao(
        y_teste,
        previstos_teste_poisson
    )

    print(f"\n===== POISSON REGRESSOR - {alvo.upper()} - TESTE =====")

    for metrica, valor in metricas_teste_poisson.items():
        print(f"{metrica}: {valor:.4f}")

    metricas = avaliar_regressao(
        y_validacao,
        previsoes_validacao
    )

    # HistGradientBoosting
    modelo_hist = criar_regressor_hist_gradient()
    modelo_hist.fit(X_treino, y_treino)

    pred_val_hist = modelo_hist.predict(X_validacao)
    pred_teste_hist = modelo_hist.predict(X_teste)

    print("\nHistGradientBoosting")
    print("Validacao:", avaliar_regressao(y_validacao, pred_val_hist))
    print("Teste:", avaliar_regressao(y_teste, pred_teste_hist))

    # XGBoost
    modelo_xgb = criar_regressor_xgboost()
    modelo_xgb.fit(X_treino, y_treino)

    pred_val_xgb = modelo_xgb.predict(X_validacao)
    pred_teste_xgb = modelo_xgb.predict(X_teste)

    print("\n===== XGBOOST -", alvo.upper(), "- VALIDACAO =====")
    print(avaliar_regressao(y_validacao, pred_val_xgb))

    print("\n===== XGBOOST -", alvo.upper(), "- TESTE =====")
    print(avaliar_regressao(y_teste, pred_teste_xgb))

    exibir_resultados(
        f"RANDOM FOREST - {alvo.upper()} - VALIDACAO",
        metricas
    )

    metricas_baseline = avaliar_regressao(
        y_validacao,
        previsoes_baseline_validacao
    )

    exibir_resultados(
        f"BASELINE - {alvo.upper()} - VALIDACAO",
        metricas_baseline
    )

    # Avaliacao no teste.
    previsoes_teste = modelo.predict(X_teste)
    previsoes_baseline_teste = baseline.predict(X_teste)

    metricas = avaliar_regressao(
        y_teste,
        previsoes_teste
    )


    exibir_resultados(
        f"RANDOM FOREST - {alvo.upper()} - TESTE",
        metricas
    )

    metricas_baseline = avaliar_regressao(
        y_teste,
        previsoes_baseline_teste
    )

    exibir_resultados(
        f"BASELINE - {alvo.upper()} - TESTE",
        metricas_baseline
    )

    # Persistencia do modelo Random Forest.
    salvar_modelo(modelo, caminho)

    # Retorna o modelo Poisson e as previsoes da validacao.
    return {
        "modelo": modelo_poisson,
        "previsoes": pd.Series(
            previstos_validacao_poisson,
            index=X_validacao.index
        ),
        "reais": y_validacao
    }


def main():

    base = carregar_base()

    print(f"Partidas carregadas: {len(base)}")

    # Modelo para prever os gols do mandante.
    resultado_gm = treinar_modelo(
        base,
        "gm",
        C.ARQ_MODELO_MANDANTE
    )

    # Modelo para prever os gols do visitante.
    resultado_gv = treinar_modelo(
        base,
        "gv",
        C.ARQ_MODELO_VISITANTE
    )

    # Combina as previsoes para gerar probabilidades.
    probabilidades = gerar_probabilidades_partidas(
        base,
        resultado_gm,
        resultado_gv
    )
    

    print("\n===== PROBABILIDADES DAS PARTIDAS - VALIDACAO =====")

    print(
        probabilidades.head(20).round(4).to_string(index=False)
    )

    probabilidades_array = probabilidades[
        ["P_M", "P_E", "P_V"]
    ].to_numpy()

    metricas = avaliar_probabilidades(
        probabilidades["resultado_real"],
        probabilidades_array
    )

    print("\n===== AVALIACAO DAS PROBABILIDADES =====")

    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.4f}")


    avaliar_baseline_classificacao(
        base,
        probabilidades
    )
    analisar_zeros_e_vitorias_visitantes(base, probabilidades)

if __name__ == "__main__":
    main()
