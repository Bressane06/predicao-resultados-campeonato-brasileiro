# @Author: gabriel.bressane  # @Date: 2026-11-03 14:37:47  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-03 14:37:47 

"""
Módulo de engenharia de atributos (Feature Engineering).

Desenvolvimento: Gabriel Bressane
Apoio no desenvolvimento: Claude (Anthropic).
Validação: código revisado e testado pelo autor.

A ferramenta de IA foi utilizada como apoio na estruturação,
refatoração e aprimoramento das funções. A implementação final
foi analisada e validada pelo autor.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config as CONFIGURACAO

# -----------------------------------------------------------------------------
# Utilitários

def calcular_media_exponencial_anterior(historico_clubes, coluna, meia_vida, grupos=("clube",), max_intervalo_dias=None):
    """Calcula a média exponencial dos jogos anteriores (shift(1)).
    Se max_intervalo_dias for informado, anula o valor quando a última observação
    válida for mais antiga que o intervalo definido."""
    grupos = list(grupos)
    media = historico_clubes.groupby(grupos)[coluna].transform(
        lambda serie: serie.shift(1).ewm(halflife=meia_vida, ignore_na=True, min_periods=1).mean())
    if max_intervalo_dias is not None:
        ultima_data_valida = historico_clubes["data"].where(historico_clubes[coluna].notna())
        ultima_data_anterior = ultima_data_valida.groupby([historico_clubes[grupo] for grupo in grupos]).transform(lambda serie: serie.shift(1).ffill())
        intervalo = (historico_clubes["data"] - ultima_data_anterior).dt.days
        media = media.where(~(intervalo > max_intervalo_dias))
    return media


# -----------------------------------------------------------------------------
# Tabela com uma linha por clube/partida

def criar_historico_clubes(partidas, estatisticas, gols, cartoes):
    """Cria uma linha por clube/partida, considerando a perspectiva de cada clube,
    e ordena os registros cronologicamente."""
    partidas_mandantes = pd.DataFrame({"ID": partidas.ID, "data": partidas.data, "rodada": partidas.rodada, "clube": partidas.mandante,
                         "adversario": partidas.visitante, "mando_campo": 1, "gols_marcados": partidas.gm, "gols_sofridos": partidas.gv})
    partidas_visitantes = pd.DataFrame({"ID": partidas.ID, "data": partidas.data, "rodada": partidas.rodada, "clube": partidas.visitante,
                         "adversario": partidas.mandante, "mando_campo": 0, "gols_marcados": partidas.gv, "gols_sofridos": partidas.gm})
    historico_clubes = pd.concat([partidas_mandantes, partidas_visitantes]).sort_values(["data", "ID", "mando_campo"], ascending=[True, True, False])
    historico_clubes = historico_clubes.merge(estatisticas, on=["ID", "clube"], how="left") \
         .merge(gols, on=["ID", "clube"], how="left") \
         .merge(cartoes, on=["ID", "clube"], how="left")
    historico_clubes = historico_clubes.sort_values(["data", "ID", "mando_campo"], ascending=[True, True, False]).reset_index(drop=True)
    historico_clubes["pontos"] = np.where(historico_clubes.gols_marcados.isna(), np.nan, np.where(historico_clubes.gols_marcados > historico_clubes.gols_sofridos, 3, np.where(historico_clubes.gols_marcados == historico_clubes.gols_sofridos, 1, 0)))
    historico_clubes["aproveitamento"] = historico_clubes.pontos / 3
    return historico_clubes


# -----------------------------------------------------------------------------
# A, D: Atributos de cada clube

COLUNAS_ESTATISTICAS = ["chutes", "chutes_no_alvo", "posse", "faltas", "escanteios", "cartoes_est"]
COLUNAS_GOLS = ["gols_1t", "gols_2t", "gols_finais", "marcou_primeiro"]
COLUNAS_CARTOES = ["amarelos", "vermelhos", "cartoes_2t"]


def calcular_atributos_clube(historico_clubes):
    """Calcula atributos históricos de desempenho para cada clube."""
    atributos_clubes = pd.DataFrame({"ID": historico_clubes.ID, "clube": historico_clubes.clube, "mando_campo": historico_clubes.mando_campo})
    # Forma recente (curta), nível de desempenho (longa) e desempenho no mando atual
    for nome_atributo, coluna_origem in [("aproveitamento", "aproveitamento"), ("gols_marcados", "gols_marcados"), ("gols_sofridos", "gols_sofridos")]:
        atributos_clubes[f"{nome_atributo}_curto"] = calcular_media_exponencial_anterior(historico_clubes, coluna_origem, CONFIGURACAO.HL_CURTO)
        atributos_clubes[f"{nome_atributo}_longo"] = calcular_media_exponencial_anterior(historico_clubes, coluna_origem, CONFIGURACAO.HL_LONGO)
        atributos_clubes[f"{nome_atributo}_local"] = calcular_media_exponencial_anterior(historico_clubes, coluna_origem, CONFIGURACAO.HL_LOCAL, grupos=("clube", "mando_campo"))
    # Estatísticas das partidas, gols por tempo e cartões, considerando a validade temporal dos dados
    for coluna in COLUNAS_ESTATISTICAS + COLUNAS_GOLS + COLUNAS_CARTOES:
        atributos_clubes[f"{coluna}_ew"] = calcular_media_exponencial_anterior(historico_clubes, coluna, CONFIGURACAO.HL_STATS, max_intervalo_dias=CONFIGURACAO.MAX_GAP_DIAS)
    # C: descanso (dias desde a partida anterior) e experiência histórica
    atributos_clubes["dias_descanso"] = historico_clubes.groupby("clube")["data"].diff().dt.days.clip(upper=30)
    atributos_clubes["quantidade_jogos_historicos"] = historico_clubes.groupby("clube").cumcount()
    return atributos_clubes


# -----------------------------------------------------------------------------
# Elo

def calcular_classificacao_elo(partidas):
    """Calcula a classificação Elo pré-jogo dos mandantes e visitantes.
    Regride os ratings em direção à média a cada nova temporada."""

    classificacao_elo = defaultdict(
        lambda: CONFIGURACAO.ELO_INICIAL
    )

    registros_elo, ano_atual = [], None

    for partida in partidas.sort_values(["data", "ID"]).itertuples():
        if partida.ano != ano_atual:
            if ano_atual is not None:
                for clube in list(classificacao_elo):
                    classificacao_elo[clube] = (
                        CONFIGURACAO.ELO_INICIAL
                        + (1 - CONFIGURACAO.ELO_REGRESSAO_ANUAL)
                        * (classificacao_elo[clube] - CONFIGURACAO.ELO_INICIAL)
                    )

            ano_atual = partida.ano

        elo_mandante = classificacao_elo[partida.mandante]
        elo_visitante = classificacao_elo[partida.visitante]

        registros_elo.append(
            (partida.ID, elo_mandante, elo_visitante)
        )

        if pd.notna(partida.gm):
            probabilidade_esperada = 1 / (
                1 + 10 ** (
                    -(elo_mandante + CONFIGURACAO.ELO_VANTAGEM_MANDANTE
                      - elo_visitante) / 400
                )
            )

            resultado_obtido = (
                1.0 if partida.gm > partida.gv
                else (0.5 if partida.gm == partida.gv else 0.0)
            )

            variacao_elo = (
                CONFIGURACAO.ELO_K
                * (np.log(abs(partida.gm - partida.gv) + 1) + 1)
                * (resultado_obtido - probabilidade_esperada)
            )

            classificacao_elo[partida.mandante] += variacao_elo
            classificacao_elo[partida.visitante] -= variacao_elo

    return pd.DataFrame(
        registros_elo,
        columns=["ID", "elo_mandante", "elo_visitante"]
    )


# -----------------------------------------------------------------------------
# B: Confrontos diretos (H2H)

def calcular_confrontos_diretos(partidas, cartoes_por_partida):
    """
    Calcula atributos históricos dos confrontos diretos (Head-to-Head - H2H)
    entre os clubes de cada partida, considerando os encontros anteriores,
    independentemente de qual equipe atuou como mandante.

    Para cada partida, a função consulta o histórico de confrontos entre os
    dois clubes e calcula indicadores sob a perspectiva do mandante atual.
    Dessa forma, mesmo que os clubes tenham se enfrentado anteriormente com
    mandos de campo diferentes, os resultados são reorganizados para manter
    uma perspectiva consistente entre mandante e visitante.

    São considerados os seguintes atributos:
        - h2h_quantidade_jogos: quantidade de confrontos anteriores considerados.
        - h2h_vitorias_mandante: vitórias anteriores do clube mandante atual.
        - h2h_empates: quantidade de empates nos confrontos anteriores.
        - h2h_vitorias_visitante: vitórias anteriores do clube visitante atual.
        - h2h_media_gols_total: média de gols totais por confronto.
        - h2h_media_gols_mandante: média de gols marcados pelo mandante atual.
        - h2h_media_gols_visitante: média de gols marcados pelo visitante atual.
        - h2h_aproveitamento_ponderado: aproveitamento do mandante atual,
          considerando pesos maiores para confrontos mais recentes.
        - h2h_vitorias_mandante_em_casa: vitórias do mandante atual nos
          confrontos em que também atuou como mandante.
        - h2h_jogos_mandante_em_casa: quantidade de confrontos em que o
          mandante atual atuou em seu próprio estádio.
        - h2h_media_cartoes: média de cartões registrados nos confrontos
          anteriores considerados.

    O histórico é armazenado em um dicionário indexado por um conjunto
    imutável (frozenset) formado pelos dois clubes. Isso permite localizar
    os confrontos entre as mesmas equipes independentemente da ordem em
    que aparecem como mandante e visitante.

    A quantidade de confrontos considerados é limitada pela constante
    CONFIGURACAO.H2H_N. Além disso, é aplicado um decaimento exponencial
    definido por CONFIGURACAO.H2H_DECAY, atribuindo maior peso aos jogos
    mais recentes no cálculo do aproveitamento ponderado.

    Para evitar vazamento de informações (data leakage), os atributos de
    cada partida são calculados utilizando exclusivamente os confrontos
    registrados anteriormente. O resultado da partida atual só é adicionado
    ao histórico depois que seus atributos são calculados.

    Parâmetros:
        partidas (pd.DataFrame): conjunto de partidas, contendo identificação,
            data, mandante, visitante e gols marcados por cada equipe.
        cartoes_por_partida (dict): dicionário que associa o ID de cada partida
            à quantidade total de cartões registrados no confronto.

    Retorno:
        pd.DataFrame: tabela com uma linha por partida e os atributos
            históricos dos confrontos diretos entre os clubes.
    """
    historico_confrontos = defaultdict(list)  # frozenset({clube_a, clube_b}) -> [(mandante, gm, gv, cartoes_total)]
    registros_confrontos = []
    for partida in partidas.sort_values(["data", "ID"]).itertuples():
        clube_mandante, chave_confronto = partida.mandante, frozenset((partida.mandante, partida.visitante))
        confrontos_anteriores = historico_confrontos[chave_confronto][-CONFIGURACAO.H2H_N:][::-1]  # Mais recentes primeiro
        atributos_confronto = {"ID": partida.ID, "h2h_quantidade_jogos": len(confrontos_anteriores)}
        if confrontos_anteriores:
            pesos = CONFIGURACAO.H2H_DECAY ** np.arange(len(confrontos_anteriores))
            gols_mandante_atual = np.array([jogo[1] if jogo[0] == clube_mandante else jogo[2] for jogo in confrontos_anteriores], float)
            gols_visitante_atual = np.array([jogo[2] if jogo[0] == clube_mandante else jogo[1] for jogo in confrontos_anteriores], float)
            mandante_jogou_em_casa = np.array([jogo[0] == clube_mandante for jogo in confrontos_anteriores])
            pontos_mandante_atual = np.where(gols_mandante_atual > gols_visitante_atual, 3, np.where(gols_mandante_atual == gols_visitante_atual, 1, 0))
            atributos_confronto.update(
                h2h_vitorias_mandante=int((gols_mandante_atual > gols_visitante_atual).sum()), h2h_empates=int((gols_mandante_atual == gols_visitante_atual).sum()),
                h2h_vitorias_visitante=int((gols_mandante_atual < gols_visitante_atual).sum()),
                h2h_media_gols_total=float((gols_mandante_atual + gols_visitante_atual).mean()),
                h2h_media_gols_mandante=float(gols_mandante_atual.mean()), h2h_media_gols_visitante=float(gols_visitante_atual.mean()),
                h2h_aproveitamento_ponderado=float((pesos * pontos_mandante_atual).sum() / (3 * pesos.sum())),
                h2h_vitorias_mandante_em_casa=int(((gols_mandante_atual > gols_visitante_atual) & mandante_jogou_em_casa).sum()),
                h2h_jogos_mandante_em_casa=int(mandante_jogou_em_casa.sum()),
                h2h_media_cartoes=float(np.nanmean([jogo[3] for jogo in confrontos_anteriores])) if any(
                    not np.isnan(jogo[3]) for jogo in confrontos_anteriores) else np.nan)
        registros_confrontos.append(atributos_confronto)
        if pd.notna(partida.gm):  # A partida só entra no histórico depois de seus atributos serem calculados
            historico_confrontos[chave_confronto].append((partida.mandante, partida.gm, partida.gv, cartoes_por_partida.get(partida.ID, np.nan)))
    return pd.DataFrame(registros_confrontos)


# -----------------------------------------------------------------------------
# Base final

def construir_base(partidas, estatisticas, gols, cartoes):
    """Retorna uma linha por partida, contendo identificação, alvos (gm, gv, resultado) e atributos."""
    historico_clubes = criar_historico_clubes(partidas, estatisticas, gols, cartoes)
    atributos_clubes = calcular_atributos_clube(historico_clubes)
    colunas_atributos = [coluna for coluna in atributos_clubes.columns if coluna not in ("ID", "clube", "mando_campo")]
    atributos_mandante = atributos_clubes[atributos_clubes.mando_campo == 1].set_index("ID")[colunas_atributos].add_prefix("mandante_")
    atributos_visitante = atributos_clubes[atributos_clubes.mando_campo == 0].set_index("ID")[colunas_atributos].add_prefix("visitante_")
    base = partidas.merge(atributos_mandante, on="ID").merge(atributos_visitante, on="ID")

    base = base.merge(calcular_classificacao_elo(partidas), on="ID")
    cartoes_totais = (cartoes.groupby("ID")[["amarelos", "vermelhos"]].sum().sum(axis=1)).to_dict()
    base = base.merge(calcular_confrontos_diretos(partidas, cartoes_totais), on="ID", how="left")

    # Diferenças diretas entre mandante e visitante
    base["diferenca_elo"] = base.elo_mandante - base.elo_visitante
    base["diferenca_aproveitamento"] = base.mandante_aproveitamento_longo - base.visitante_aproveitamento_longo
    base["diferenca_saldo_gols"] = (base.mandante_gols_marcados_longo - base.mandante_gols_sofridos_longo) - (base.visitante_gols_marcados_longo - base.visitante_gols_sofridos_longo)
    return base.sort_values(["data", "ID"]).reset_index(drop=True)


COLUNAS_IDENTIFICACAO = ["ID", "data", "ano", "arena", "gm", "gv", "resultado"]
ATRIBUTOS_CATEGORICOS = ["mandante", "visitante"]


def obter_atributos_numericos(base):
    """Retorna os nomes das colunas numéricas que podem ser utilizadas como atributos."""
    colunas_excluidas = set(COLUNAS_IDENTIFICACAO) | set(ATRIBUTOS_CATEGORICOS)
    return [coluna for coluna in base.columns if coluna not in colunas_excluidas]


def adicionar_partida_futura(partidas, mandante, visitante, data, rodada=None, arena=None):
    """Adiciona uma partida sem placar ao histórico para gerar atributos de uma partida a prever."""
    data = pd.Timestamp(data)
    nova_partida = pd.DataFrame([{
        "ID": int(partidas.ID.max()) + 1, "rodada": rodada if rodada is not None else np.nan, "data": data,
        "ano": data.year, "mes": data.month, "dia_semana": data.dayofweek,
        "mandante": mandante, "visitante": visitante, "arena": arena,
        "gm": np.nan, "gv": np.nan, "resultado": np.nan}])
    return pd.concat([partidas, nova_partida], ignore_index=True)


def gerar_base(salvar=True):
    from .data_loader import juntar_operacoes
    partidas, estatisticas, gols, cartoes = juntar_operacoes(salvar)
    base = construir_base(partidas, estatisticas, gols, cartoes)
    if salvar:
        base.to_csv(CONFIGURACAO.ARQ_BASE, index=False)
    return base


if __name__ == "__main__":
    base = gerar_base()
    print(base.shape)
    print("Atributos numéricos:", len(obter_atributos_numericos(base)))
    print(base.isna().mean().sort_values(ascending=False).head(8).round(2))

    print("Vantagem do mandante:", CONFIGURACAO.ELO_VANTAGEM_MANDANTE)

    print(
        base[
            ["elo_mandante", "elo_visitante", "diferenca_elo"]
        ].describe()
    )