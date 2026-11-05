
import streamlit as st
import pandas as pd
from datetime import date

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


if __name__ == "__main__":
    main()
