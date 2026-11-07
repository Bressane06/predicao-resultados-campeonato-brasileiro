
import streamlit as st
import pandas as pd
from datetime import date

from src import config as CONFIGURACAO
from src.atributos import calcular_elo_final_temporadas
from src.regressao import (
    carregar_modelos_treinados,
    carregar_dados_historicos,
    prever_partida_interativa
)


# -----------------------------------------------------------------------------
# Configuração da página

st.set_page_config(
    page_title="Previsão de Resultados - Brasileirão",
    layout="wide"
)


# -----------------------------------------------------------------------------
# Carregamento dos dados

@st.cache_resource(show_spinner="Carregando modelos treinados...")
def carregar_modelos():
    return carregar_modelos_treinados()


@st.cache_data(show_spinner="Carregando histórico de partidas...")
def carregar_historico():
    return carregar_dados_historicos()

def exibir_grafico_placares(previsao):
    """Exibe os quatro placares com maiores probabilidades."""

    st.subheader("Distribuição de probabilidade dos placares")

    dados = pd.DataFrame(previsao["placares_mais_provaveis"])

    dados = dados.sort_values("probabilidade",ascending=True)

    dados = dados.set_index("placar")

    st.bar_chart(
        dados["probabilidade"],
        horizontal=False,
        x_label="Probabilidade",
        y_label="Placar",
        color="#2196F3"
    )


def exibir_grafico_resultados(previsao, mandante, visitante):
    """Compara as probabilidades de vitória, empate e derrota."""

    st.subheader("Comparação das probabilidades de resultado")

    dados = pd.DataFrame({
        "Resultado": [
            f"Vitória {mandante}",
            "Empate",
            f"Vitória {visitante}"
        ],
        "Probabilidade": [
            previsao["P_M"],
            previsao["P_E"],
            previsao["P_V"]
        ]
    })

    dados = dados.set_index("Resultado")

    st.bar_chart(
        dados,
        horizontal=False,
        x_label="Probabilidade",
        y_label="Resultado",
        color="#4CAF50"
    )

def exibir_mapa_calor(previsao):
    """Exibe o mapa de calor das probabilidades dos placares."""

    import plotly.express as px

    st.subheader("Mapa de calor dos placares")

    matriz = previsao["matriz_placares"] * 100

    figura = px.imshow(
        matriz,
        labels={
            "x": "Gols do visitante",
            "y": "Gols do mandante",
            "color": "Probabilidade (%)"
        },
        text_auto=".2f",
        color_continuous_scale="Reds",
        aspect="auto",
        template="plotly_dark"
    )
    figura.update_layout(
        xaxis=dict(dtick=1),
        yaxis=dict(dtick=1),
        coloraxis_colorbar=dict(title="Probabilidade (%)")
    )

    st.plotly_chart(figura, use_container_width=True)

# -----------------------------------------------------------------------------
# Exibição dos resultados da previsão

def exibir_previsao(
    previsao,
    mandante,
    visitante,
    historico_mandante,
    historico_visitante,
    historico_h2h
):

    st.header("Resultado da previsão")

    st.subheader(f"{mandante} x {visitante}")

    col1, col2 = st.columns(2)

    col1.metric(
        f"Gols esperados - {mandante}",
        f'{previsao["GM_previsto"]:.2f}'
    )

    col2.metric(
        f"Gols esperados - {visitante}",
        f'{previsao["GV_previsto"]:.2f}'
    )

    st.subheader("Probabilidades do resultado")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Vitória mandante", f'{previsao["P_M"]:.2%}')
    col2.metric("Empate", f'{previsao["P_E"]:.2%}')
    col3.metric("Vitória visitante", f'{previsao["P_V"]:.2%}')
    col4.metric("Placar 0 x 0", f'{previsao["P_00"]:.2%}')

    st.subheader("Quatro placares mais prováveis")

    dados_placares = pd.DataFrame(
        previsao["placares_mais_provaveis"]
    )

    dados_placares["probabilidade"] = (
        dados_placares["probabilidade"].map(
            lambda valor: f"{valor:.2%}"
        )
    )

    dados_placares.columns = ["Placar", "Probabilidade"]

    st.dataframe(
        dados_placares,
        use_container_width=True,
        hide_index=True
    )

    # -------------------------------------------------------------------------
    # Histórico das equipes

    colunas = [
        "data", "mandante", "gm", "gv", "visitante", "resultado"
    ]

    nomes_colunas = [
        "Data", "Mandante", "Gols mandante",
        "Gols visitante", "Visitante", "Resultado"
    ]

    st.header("Histórico de partidas")

    st.subheader(f"Últimos cinco jogos do {mandante}")

    tabela_mandante = historico_mandante[colunas].copy()
    tabela_mandante.columns = nomes_colunas
    tabela_mandante["Data"] = pd.to_datetime(
        tabela_mandante["Data"]
    ).dt.strftime("%d/%m/%Y")

    st.dataframe(
        tabela_mandante,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(f"Últimos cinco jogos do {visitante}")

    tabela_visitante = historico_visitante[colunas].copy()
    tabela_visitante.columns = nomes_colunas
    tabela_visitante["Data"] = pd.to_datetime(
        tabela_visitante["Data"]
    ).dt.strftime("%d/%m/%Y")

    st.dataframe(
        tabela_visitante,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(f"Últimos cinco confrontos diretos: {mandante} x {visitante}")

    tabela_h2h = historico_h2h[colunas].copy()
    tabela_h2h.columns = nomes_colunas
    tabela_h2h["Data"] = pd.to_datetime(
        tabela_h2h["Data"]
    ).dt.strftime("%d/%m/%Y")

    st.dataframe(
        tabela_h2h,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        exibir_grafico_placares(previsao)

    with col2:
        exibir_grafico_resultados(
            previsao,
            mandante,
            visitante
        )

    st.divider()

    exibir_mapa_calor(previsao)


# -----------------------------------------------------------------------------
# Desempenho do modelo

@st.cache_data(show_spinner="Carregando métricas do modelo...")
def carregar_metricas():
    """Le as tabelas de metricas geradas por: python -m src.regressao."""

    regressao = pd.read_csv(
        CONFIGURACAO.ARQ_METRICAS_REGRESSAO, encoding="utf-8-sig"
    )
    probabilidades = pd.read_csv(
        CONFIGURACAO.ARQ_METRICAS_PROBABILIDADES, encoding="utf-8-sig"
    )

    return regressao, probabilidades


def exibir_desempenho_modelo():
    """Exibe as metricas de avaliacao do modelo na validacao e no teste."""

    st.header("Desempenho do modelo")

    if not (
        CONFIGURACAO.ARQ_METRICAS_REGRESSAO.exists()
        and CONFIGURACAO.ARQ_METRICAS_PROBABILIDADES.exists()
    ):
        st.info(
            "As tabelas de métricas ainda não foram geradas. "
            "Execute `python -m src.regressao` para criá-las."
        )
        return

    regressao, probabilidades = carregar_metricas()

    st.caption(
        f"Treino com jogos até {CONFIGURACAO.ANO_FIM_TREINO}, validação em "
        f"{CONFIGURACAO.ANO_VALIDACAO} e teste de "
        f"{CONFIGURACAO.ANO_INICIO_TESTE} em diante (divisão cronológica)."
    )

    aba_gols, aba_resultado = st.tabs(
        ["Previsão de gols (regressão)", "Probabilidades do resultado"]
    )

    with aba_gols:

        st.caption(
            "MAE, MSE e RMSE: quanto menores, melhor. R²: quanto mais "
            "próximo de 1, melhor. O baseline sempre prevê a média de gols."
        )

        st.dataframe(
            regressao,
            height=altura_tabela(regressao),
            hide_index=True,
            use_container_width=True
        )

    with aba_resultado:

        st.caption(
            "Log Loss e Brier Score: quanto menores, melhor. O baseline usa "
            "apenas a frequência histórica de vitória do mandante, empate "
            "e vitória do visitante."
        )

        st.dataframe(
            probabilidades,
            height=altura_tabela(probabilidades),
            hide_index=True,
            use_container_width=True
        )


# -----------------------------------------------------------------------------
# Classificação final por temporada

def altura_tabela(tabela):
    """Altura (px) para mostrar todas as linhas, sem barra de rolagem."""
    return (len(tabela) + 1) * 35 + 3


@st.cache_data(show_spinner="Calculando classificações finais...")
def calcular_classificacoes(partidas):
    """Calcula a tabela final de cada temporada a partir dos resultados.

    Retorna {ano_de_inicio: (tabela, jogos_na_base, jogos_esperados)}.
    """

    jogos = partidas.dropna(subset=["gm", "gv"]).copy()
    jogos["data"] = pd.to_datetime(jogos["data"])
    jogos = jogos.sort_values(["data", "ID"]).reset_index(drop=True)

    # Uma nova temporada começa após uma pausa longa entre jogos. Isso evita
    # misturar anos de calendário (a temporada 2020 terminou em fev/2021).
    jogos["temporada"] = (jogos["data"].diff().dt.days.fillna(0) > 60).cumsum()

    mandantes = pd.DataFrame({
        "temporada": jogos["temporada"],
        "Clube": jogos["mandante"],
        "GP": jogos["gm"],
        "GC": jogos["gv"],
        "V": (jogos["resultado"] == "M").astype(int),
        "E": (jogos["resultado"] == "E").astype(int),
        "D": (jogos["resultado"] == "V").astype(int),
    })

    visitantes = pd.DataFrame({
        "temporada": jogos["temporada"],
        "Clube": jogos["visitante"],
        "GP": jogos["gv"],
        "GC": jogos["gm"],
        "V": (jogos["resultado"] == "V").astype(int),
        "E": (jogos["resultado"] == "E").astype(int),
        "D": (jogos["resultado"] == "M").astype(int),
    })

    resultados = pd.concat([mandantes, visitantes], ignore_index=True)
    resultados["J"] = 1

    classificacoes = {}

    for temporada, dados in resultados.groupby("temporada"):

        tabela = (
            dados.groupby("Clube")[["J", "V", "E", "D", "GP", "GC"]]
            .sum()
            .reset_index()
        )

        tabela["P"] = tabela["V"] * 3 + tabela["E"]
        tabela["SG"] = tabela["GP"] - tabela["GC"]
        tabela["Aprov. (%)"] = (tabela["P"] / (tabela["J"] * 3) * 100).round(1)

        # Critérios: pontos, vitórias, saldo de gols e gols pró.
        tabela = tabela.sort_values(
            ["P", "V", "SG", "GP"],
            ascending=False
        ).reset_index(drop=True)

        tabela.insert(0, "Pos", tabela.index + 1)
        tabela = tabela[
            ["Pos", "Clube", "P", "J", "V", "E", "D",
             "GP", "GC", "SG", "Aprov. (%)"]
        ]

        ano = int(jogos.loc[jogos["temporada"] == temporada, "data"].min().year)
        n_clubes = len(tabela)
        classificacoes[ano] = (
            tabela,
            int((jogos["temporada"] == temporada).sum()),
            n_clubes * (n_clubes - 1),
        )

    return dict(sorted(classificacoes.items(), reverse=True))


@st.cache_data(show_spinner="Calculando ranking Elo...")
def calcular_ranking_elo(partidas):
    """Ranking Elo (mesmo sistema usado pelo modelo) no fim de cada temporada.

    Retorna {ano_de_inicio: tabela}.
    """

    elo = calcular_elo_final_temporadas(partidas)

    rankings = {}

    for ano, dados in elo.groupby("temporada"):

        tabela = dados.sort_values("elo_final", ascending=False)
        tabela = tabela.reset_index(drop=True)

        tabela["variacao"] = tabela["elo_final"] - tabela["elo_inicial"]
        tabela.insert(0, "Pos", tabela.index + 1)
        tabela = tabela.rename(columns={
            "clube": "Clube",
            "elo_final": "Elo final",
            "elo_inicial": "Elo inicial",
            "variacao": "Variação",
        })

        rankings[int(ano)] = tabela[
            ["Pos", "Clube", "Elo final", "Elo inicial", "Variação"]
        ].round(1)

    return rankings


@st.cache_data(show_spinner="Calculando ranking geral...")
def calcular_ranking_geral(partidas):
    """Ranking geral de todos os clubes da base.

    Retorna (ranking_elo, tabela_historica).
    """

    # Elo: resumo de cada clube ao longo de todas as temporadas.
    elo = calcular_elo_final_temporadas(partidas).sort_values("temporada")

    ranking_elo = elo.groupby("clube").agg(
        pico=("elo_final", "max"),
        medio=("elo_final", "mean"),
        ultima=("temporada", "max"),
        temporadas=("temporada", "size"),
    )

    linhas_pico = elo.loc[elo.groupby("clube")["elo_final"].idxmax()]
    ranking_elo["ano_pico"] = linhas_pico.set_index("clube")["temporada"]

    ranking_elo["atual"] = (
        elo.groupby("clube").tail(1).set_index("clube")["elo_final"]
    )

    ranking_elo = ranking_elo.sort_values("pico", ascending=False)
    ranking_elo = ranking_elo.reset_index()
    ranking_elo.insert(0, "Pos", ranking_elo.index + 1)
    ranking_elo = ranking_elo.rename(columns={
        "clube": "Clube",
        "pico": "Pico de Elo",
        "ano_pico": "Temporada do pico",
        "medio": "Elo médio",
        "atual": "Elo na última temporada",
        "ultima": "Última temporada",
        "temporadas": "Temporadas",
    })[[
        "Pos", "Clube", "Pico de Elo", "Temporada do pico", "Elo médio",
        "Elo na última temporada", "Última temporada", "Temporadas"
    ]]
    ranking_elo[["Pico de Elo", "Elo médio", "Elo na última temporada"]] = (
        ranking_elo[["Pico de Elo", "Elo médio", "Elo na última temporada"]]
        .round(1)
    )

    # Tabela histórica: soma das classificações finais de todas as temporadas.
    tabelas = []
    for tabela, _, _ in calcular_classificacoes(partidas).values():
        parcial = tabela.copy()
        parcial["Títulos"] = (parcial["Pos"] == 1).astype(int)
        parcial["Temporadas"] = 1
        tabelas.append(parcial)

    historica = (
        pd.concat(tabelas, ignore_index=True)
        .groupby("Clube")[
            ["P", "J", "V", "E", "D", "GP", "GC", "Títulos", "Temporadas"]
        ]
        .sum()
        .reset_index()
    )

    historica["SG"] = historica["GP"] - historica["GC"]
    historica["Aprov. (%)"] = (
        historica["P"] / (historica["J"] * 3) * 100
    ).round(1)

    historica = historica.sort_values(
        ["P", "V", "SG", "GP"],
        ascending=False
    ).reset_index(drop=True)

    historica.insert(0, "Pos", historica.index + 1)
    historica = historica[[
        "Pos", "Clube", "P", "J", "V", "E", "D", "GP", "GC", "SG",
        "Aprov. (%)", "Títulos", "Temporadas"
    ]]

    return ranking_elo, historica


def exibir_ranking_geral(partidas):
    """Exibe o ranking geral dos clubes, do melhor ao pior."""

    st.header("Ranking geral dos clubes")

    ranking_elo, historica = calcular_ranking_geral(partidas)

    aba_elo, aba_historica = st.tabs(
        ["Ranking Elo geral", "Tabela histórica (pontos)"]
    )

    with aba_elo:

        st.caption(
            "Ordenado pelo maior Elo que o clube alcançou ao fim de uma "
            "temporada. O Elo médio e a quantidade de temporadas ajudam a "
            "comparar clubes com históricos de tamanhos diferentes."
        )

        st.dataframe(
            ranking_elo,
            height=altura_tabela(ranking_elo),
            hide_index=True,
            use_container_width=True
        )

    with aba_historica:

        st.caption(
            "Soma dos resultados de todas as temporadas da base. "
            "Clubes com mais temporadas acumulam mais pontos; "
            "o aproveitamento permite comparar clubes de forma justa."
        )

        st.dataframe(
            historica,
            height=altura_tabela(historica),
            hide_index=True,
            use_container_width=True
        )


def exibir_classificacoes(partidas):
    """Exibe a classificação final de todas as temporadas da base."""

    st.header("Classificação final dos campeonatos")

    st.caption(
        "Calculada a partir dos resultados da base histórica. "
        "Critérios de desempate: vitórias, saldo de gols e gols pró."
    )

    classificacoes = calcular_classificacoes(partidas)
    rankings_elo = calcular_ranking_elo(partidas)

    for indice, (ano, (tabela, jogos, esperados)) in enumerate(
        classificacoes.items()
    ):

        campeao = tabela.loc[0, "Clube"]

        with st.expander(
            f"{ano} — Campeão: {campeao}",
            expanded=(indice == 0)
        ):

            aba_pontos, aba_elo = st.tabs(
                ["Classificação por pontos", "Ranking Elo"]
            )

            with aba_pontos:

                if jogos != esperados:
                    st.caption(
                        f"Atenção: a base tem {jogos} de {esperados} jogos "
                        "desta temporada."
                    )

                st.dataframe(
                    tabela,
                    height=altura_tabela(tabela),
                    hide_index=True,
                    use_container_width=True
                )

            with aba_elo:

                st.caption(
                    "Elo ao fim da temporada, calculado com a mesma regra "
                    "usada pelo modelo (clube novo começa em 1500)."
                )

                ranking = rankings_elo[ano].copy()
                posicao_pontos = dict(zip(tabela["Clube"], tabela["Pos"]))
                ranking["Pos. por pontos"] = ranking["Clube"].map(
                    posicao_pontos
                )

                st.dataframe(
                    ranking,
                    height=altura_tabela(ranking),
                    hide_index=True,
                    use_container_width=True
                )


# -----------------------------------------------------------------------------
# Interface principal

def main():

    st.title("Previsão de Resultados do Brasileirão")

    st.caption(
        "Selecione as equipes e informe os dados da partida "
        "para gerar uma previsão."
    )

    # Carrega os modelos previamente treinados.
    modelos = carregar_modelos()

    # Carrega os dados históricos.
    partidas, estatisticas, gols, cartoes = carregar_historico()

    # Lista de equipes disponíveis.
    equipes = sorted(
        set(partidas["mandante"].dropna()) |
        set(partidas["visitante"].dropna())
    )

    st.header("Dados da partida")

    with st.form("formulario_previsao"):

        col1, col2 = st.columns(2)

        mandante = col1.selectbox(
            "Time mandante",
            equipes
        )

        equipes_visitantes = [
            equipe for equipe in equipes
            if equipe != mandante
        ]

        visitante = col2.selectbox(
            "Time visitante",
            equipes_visitantes
        )

        col3, col4, col5 = st.columns(3)

        data_partida = col3.date_input(
            "Data da partida",
            value=date.today()
        )

        rodada = col4.number_input(
            "Rodada",
            min_value=1,
            max_value=38,
            value=1,
            step=1
        )


        enviar = st.form_submit_button(
            "Prever resultado",
            use_container_width=True
        )

    if enviar:

        if pd.Timestamp(data_partida) <= pd.to_datetime(
            partidas["data"]
        ).max():
            st.error(
                "A data da previsão deve ser posterior "
                "à última partida disponível na base histórica."
            )

        else:

            try:

                with st.spinner("Gerando previsão..."):

                    (
                        previsao,
                        historico_mandante,
                        historico_visitante,
                        historico_h2h
                    ) = prever_partida_interativa(
                        partidas=partidas,
                        estatisticas=estatisticas,
                        gols=gols,
                        cartoes=cartoes,
                        mandante=mandante,
                        visitante=visitante,
                        data=data_partida,
                        rodada=int(rodada),
                        rho=0.0,
                        modelos=modelos
                    )

                exibir_previsao(
                    previsao,
                    mandante,
                    visitante,
                    historico_mandante,
                    historico_visitante,
                    historico_h2h
                )

            except Exception as erro:
                st.error(f"Não foi possível gerar a previsão: {erro}")

    st.divider()

    exibir_desempenho_modelo()

    st.divider()

    exibir_ranking_geral(partidas)

    st.divider()

    exibir_classificacoes(partidas)


if __name__ == "__main__":
    main()
