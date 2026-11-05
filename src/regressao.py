
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
from .persistencia import salvar_modelo, carregar_modelo
from .avaliacao import avaliar_regressao, avaliar_probabilidades
from .treino import carregar_base, dividir_dados
from .data_loader import juntar_operacoes
from src.atributos import (
    gerar_atributos_partida_futura,
    obter_historico_partida,
    obter_atributos_numericos
)
def avaliar_baseline_classificacao(base, probabilidades_poisson):
    """Compara o baseline de classificacao com as probabilidades de Poisson."""

    base = base.copy()
    base["resultado"] = base["resultado"].astype(str)
    base = base.dropna(subset=["resultado"])

    # Utiliza os mesmos atributos numericos dos modelos de regressao.
    colunas = obter_atributos_numericos(base)

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

    # Exibe a comparacao entre os metodos.
    comparacao = pd.DataFrame({
        "Baseline": metricas_baseline,
        "Poisson": metricas_poisson
    })

    print("\n===== COMPARACAO: BASELINE VS POISSON =====")
    print(f"Partidas avaliadas: {len(indices)}")
    print(comparacao.round(4))

    return comparacao


def gerar_probabilidades_partidas(base, resultado_gm, resultado_gv, rho=0.0):
    """Combina as previsoes de GM e GV para calcular probabilidades."""

    indices = resultado_gm["previsoes"].index.intersection(
        resultado_gv["previsoes"].index
    )

    previsoes_gm = resultado_gm["previsoes"].loc[indices]
    previsoes_gv = resultado_gv["previsoes"].loc[indices]

    partidas = []

    for indice in indices:

        lambda_mandante = max(0, previsoes_gm.loc[indice])
        lambda_visitante = max(0, previsoes_gv.loc[indice])

        placares = calcular_matriz_placares(
            gols_mandante=lambda_mandante,
            gols_visitante=lambda_visitante,
            rho=rho
        )

        probabilidades = calcular_probabilidades_resultado(placares)

        top_placares = placares.stack().nlargest(4)
        melhor_placar = top_placares.index[0]

        partida = {
            "ID": indice,
            "GM_previsto": lambda_mandante,
            "GV_previsto": lambda_visitante,
            "placar_previsto": f"{melhor_placar[0]} x {melhor_placar[1]}",
            "P_00": placares.loc[0, 0],
            "P_M": probabilidades["M"],
            "P_E": probabilidades["E"],
            "P_V": probabilidades["V"],
            "resultado_real": base.loc[indice, "resultado"]
        }

        for posicao, (placar, probabilidade) in enumerate(
            top_placares.items(),
            start=1
        ):
            partida[f"placar_{posicao}"] = (
                f"{placar[0]} x {placar[1]}"
            )
            partida[f"prob_placar_{posicao}"] = probabilidade

        partidas.append(partida)

    return pd.DataFrame(partidas)


def prever_partida(atributos_futuros, rho=-0.05, modelos=None):
    """
    Gera a previsão de gols e as probabilidades de uma partida.

    Utiliza os modelos Poisson treinados para GM e GV.
    """

    # Carrega os modelos somente se não forem fornecidos.
    if modelos is None:
        modelos = carregar_modelos_treinados()

    pacote_gm, pacote_gv = modelos

    # Recupera os modelos e suas respectivas colunas.
    modelo_gm = pacote_gm["modelo"]
    modelo_gv = pacote_gv["modelo"]

    colunas_gm = pacote_gm["colunas_atributos"]
    colunas_gv = pacote_gv["colunas_atributos"]

    # Organiza os atributos conforme o treinamento.
    X_gm = atributos_futuros.reindex(columns=colunas_gm)
    X_gv = atributos_futuros.reindex(columns=colunas_gv)

    # Prevê as médias de gols.
    lambda_mandante = max(0, float(modelo_gm.predict(X_gm)[0]))
    lambda_visitante = max(0, float(modelo_gv.predict(X_gv)[0]))

    # Calcula a distribuição dos placares.
    placares = calcular_matriz_placares(
        gols_mandante=lambda_mandante,
        gols_visitante=lambda_visitante,
        rho=rho
    )

    # Calcula as probabilidades de M, E e V.
    probabilidades = calcular_probabilidades_resultado(placares)

    # Seleciona os quatro placares mais prováveis.
    top_placares = placares.stack().nlargest(4)

    melhores_placares = [
        {
            "placar": f"{gm} x {gv}",
            "probabilidade": float(probabilidade)
        }
        for (gm, gv), probabilidade in top_placares.items()
    ]

    return {
        "GM_previsto": lambda_mandante,
        "GV_previsto": lambda_visitante,
        "P_00": float(placares.loc[0, 0]),
        "P_M": probabilidades["M"],
        "P_E": probabilidades["E"],
        "P_V": probabilidades["V"],
        "placares_mais_provaveis": melhores_placares,
        "matriz_placares": placares
    }


def testar_previsao_futura(
    mandante,
    visitante,
    data,
    rodada,
    arena,
    rho=-0.05
):
    """Testa a previsao de uma partida futura."""

    # Carrega os dados historicos.
    partidas, estatisticas, gols, cartoes = juntar_operacoes(
        salvar=False
    )

    # Gera os atributos da partida futura.
    atributos = gerar_atributos_partida_futura(
        partidas=partidas,
        estatisticas=estatisticas,
        gols=gols,
        cartoes=cartoes,
        mandante=mandante,
        visitante=visitante,
        data=data,
        rodada=rodada,
        arena=arena
    )

    # Realiza a previsao.
    previsao = prever_partida(
        atributos,
        rho=rho
    )

    nome_modelo = (
        "POISSON" if rho == 0
        else "DIXON-COLES"
    )

    print(f"\n===== PREVISAO FUTURA - {nome_modelo} =====")
    print(f"{mandante} x {visitante}")
    print(f"Data: {data}")
    print(f"Estadio: {arena}")

    print(f"Gols esperados do mandante: {previsao['GM_previsto']:.2f}")
    print(f"Gols esperados do visitante: {previsao['GV_previsto']:.2f}")

    print(f"Probabilidade de 0 x 0: {previsao['P_00']:.2%}")
    print(f"Vitoria do mandante: {previsao['P_M']:.2%}")
    print(f"Empate: {previsao['P_E']:.2%}")
    print(f"Vitoria do visitante: {previsao['P_V']:.2%}")

    print("\n===== QUATRO PLACARES MAIS PROVAVEIS =====")

    for item in previsao["placares_mais_provaveis"]:
        print(
            f"{item['placar']}: "
            f"{item['probabilidade']:.2%}"
        )



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

    colunas = obter_atributos_numericos(base)

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
    """Retorna análises de placares 0 a 0 e vitórias visitantes."""

    analise = probabilidades.merge(
        base[["ID", "gm", "gv"]],
        on="ID",
        how="left"
    )

    # Partidas que terminaram em 0 a 0.
    zeros = analise[
        (analise["gm"] == 0) & (analise["gv"] == 0)
    ].copy()

    acertos_00 = zeros[
        zeros["placar_previsto"] == "0 x 0"
    ]

    # Partidas que terminaram com vitória visitante.
    visitantes = analise[
        analise["resultado_real"] == "V"
    ].copy()

    acertos_visitantes = visitantes[
        visitantes["P_V"] >= visitantes[["P_M", "P_E"]].max(axis=1)
    ]

    # Métricas.
    recall_visitantes = (
        len(acertos_visitantes) / len(visitantes)
        if len(visitantes) else 0
    )

    return {
        "zeros": zeros,
        "acertos_00": acertos_00,
        "visitantes": visitantes,
        "acertos_visitantes": acertos_visitantes,
        "total_zeros": len(zeros),
        "total_acertos_00": len(acertos_00),
        "total_visitantes": len(visitantes),
        "total_acertos_visitantes": len(acertos_visitantes),
        "recall_visitantes": recall_visitantes
    }


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

    # Analise dos erros.
    analisar_erros(
        y_validacao,
        previsoes_validacao,
        alvo.upper()
    )

    analisar_vies(
        y_validacao,
        previsoes_validacao,
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

    # Persistencia do Random Forest.
    salvar_modelo(modelo, caminho)

    # Define o caminho do modelo Poisson.
    caminho_poisson = caminho.with_name(
        caminho.stem + "_poisson" + caminho.suffix
    )

    # Salva o modelo Poisson junto com seus metadados.
    pacote_poisson = {
        "modelo": modelo_poisson,
        "colunas_atributos": X_treino.columns.tolist(),
        "alvo": alvo
    }

    salvar_modelo(pacote_poisson, caminho_poisson)
    # Retorna os modelos e as previsoes da validacao.
    return {
        "modelo": modelo,
        "modelo_poisson": modelo_poisson,
        "caminho": caminho,
        "caminho_poisson": caminho_poisson,
        "colunas_atributos": X_treino.columns.tolist(),
        "previsoes": pd.Series(
            previstos_validacao_poisson,
            index=X_validacao.index
        ),
        "reais": y_validacao
    }

def carregar_dados_e_prever_validacao():
    """Carrega os dados, treina os modelos e gera previsões de validação."""

    base = carregar_base()

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

    # Gera probabilidades usando Poisson tradicional.
    probabilidades_poisson = gerar_probabilidades_partidas(
        base,
        resultado_gm,
        resultado_gv,
        rho=0.0
    )

    return base, probabilidades_poisson

def carregar_dados_historicos():
    """Carrega as bases históricas necessárias para a aplicação."""
    return juntar_operacoes(salvar=False)


def prever_partida_interativa(
    partidas, estatisticas, gols, cartoes,
    mandante, visitante, data, rodada, arena=None,
    rho=0.0, modelos=None
):
    """Gera a previsão e recupera os históricos da partida."""

    # Gera os atributos da partida futura.
    atributos = gerar_atributos_partida_futura(
        partidas=partidas,
        estatisticas=estatisticas,
        gols=gols,
        cartoes=cartoes,
        mandante=mandante,
        visitante=visitante,
        data=data,
        rodada=rodada,
        arena=arena
    )

    # Realiza a previsão utilizando os modelos salvos.
    previsao = prever_partida(
        atributos,
        rho=rho,
        modelos=modelos
    )
    # Recupera os três históricos.
    historico_mandante, historico_visitante, historico_h2h = (
        obter_historico_partida(
            partidas=partidas,
            mandante=mandante,
            visitante=visitante,
            data=data
        )
    )

    return (
        previsao,
        historico_mandante,
        historico_visitante,
        historico_h2h
    )

def carregar_modelos_treinados():
    """Carrega os modelos de gols previamente treinados."""

    caminho_gm = C.ARQ_MODELO_MANDANTE.with_name(
        C.ARQ_MODELO_MANDANTE.stem
        + "_poisson"
        + C.ARQ_MODELO_MANDANTE.suffix
    )

    caminho_gv = C.ARQ_MODELO_VISITANTE.with_name(
        C.ARQ_MODELO_VISITANTE.stem
        + "_poisson"
        + C.ARQ_MODELO_VISITANTE.suffix
    )

    if not caminho_gm.exists() or not caminho_gv.exists():
        raise FileNotFoundError(
            "Modelos não encontrados. Execute primeiro: py -m src.regressao"
        )

    pacote_gm = carregar_modelo(caminho_gm)
    pacote_gv = carregar_modelo(caminho_gv)

    return pacote_gm, pacote_gv
def main():

    base, probabilidades_poisson = carregar_dados_e_prever_validacao()



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

    # Poisson tradicional.
    probabilidades_poisson = gerar_probabilidades_partidas(
        base,
        resultado_gm,
        resultado_gv,
        rho=0.0
    )

    # Poisson com correcao Dixon-Coles.
    probabilidades_dc = gerar_probabilidades_partidas(
        base,
        resultado_gm,
        resultado_gv,
        rho=-0.05
    )

    # Avalia e compara os dois metodos.
    resultados = {}

    for nome, probabilidades in [
        ("POISSON", probabilidades_poisson),
        ("DIXON-COLES", probabilidades_dc)
    ]:

        metricas = avaliar_probabilidades(
            probabilidades["resultado_real"],
            probabilidades[["P_M", "P_E", "P_V"]].to_numpy()
        )

        resultados[nome] = metricas

        print(f"\n===== AVALIACAO: {nome} =====")

        for metrica, valor in metricas.items():
            print(f"{metrica}: {valor:.4f}")

    # Exibe a comparacao entre os metodos.
    comparacao = pd.DataFrame(resultados)

    print("\n===== COMPARACAO: POISSON VS DIXON-COLES =====")
    print(comparacao.round(4))

    # Utiliza Dixon-Coles nas analises complementares.
    probabilidades = probabilidades_poisson

    avaliar_baseline_classificacao(
        base,
        probabilidades
    )
    print("\n===== PROBABILIDADES DAS PARTIDAS - VALIDACAO =====")

    print(
        probabilidades.head(20).round(4).to_string(index=False)
    )

    # Compara Dixon-Coles com o baseline.
    avaliar_baseline_classificacao(
        base,
        probabilidades
    )

    # Analisa os erros nos placares 0 x 0 e nas vitorias visitantes.
    analisar_zeros_e_vitorias_visitantes(
        base,
        probabilidades
    )

    # Dados da partida futura.
    dados_partida = {
        "mandante": "São Paulo",
        "visitante": "Corinthians",
        "data": "2026-11-04",
        "rodada": 34,
        "arena": "Morumbi"
    }

    # Testa Poisson tradicional.
    testar_previsao_futura(
        **dados_partida,
        rho=0.0
    )

    # Testa Poisson com correcao Dixon-Coles.
    testar_previsao_futura(
        **dados_partida,
        rho=-0.05
    )


if __name__ == "__main__":
    main()
