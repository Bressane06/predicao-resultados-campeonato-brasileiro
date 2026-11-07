# @Author: gabriel.bressane  # @Date: 2026-11-02 23:43:09  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-02 23:43:09 
# vou usar o pandas, pois já tenho mais familiaridade para manipular arquivos CSV
import numpy as np
import pandas as pd

from . import config as CONFIGURACAO

def arrumar_minutos(serie: pd.DataFrame) -> pd.DataFrame:

    """'45+2' -> base=45, extra=2. Minuto de ordenacao: 45+2 vem antes de 46."""
    s = serie.astype(str).str.strip()
    partes = s.str.split("+", n=1, expand=True)
    base = pd.to_numeric(partes[0], errors="coerce")
    extra = pd.to_numeric(partes[1], errors="coerce").fillna(0) if partes.shape[1] > 1 else 0
    return pd.DataFrame({"base": base, "ordem": base + extra / 100.0})


# limpando arquivo de partidas (campeonato-brasileiro-full.csv)
def limpar_partidas() -> pd.DataFrame:

    df = pd.read_csv(CONFIGURACAO.ARQ_FULL)

    # ajustando nomes de colunas
    df = df.rename(columns={"rodata": "rodada", 
                            "mandante_Placar": "gm", 
                            "visitante_Placar": "gv"})
    # dd/mm/yyyy para yyyy-mm-dd, fica melhor para ordenar depois
    df["data"] = pd.to_datetime(df["data"], dayfirst=True)

    df = df.drop_duplicates(subset=["ID"])
    df = df.sort_values(by=["data", "ID"])
    df = df.reset_index(drop=True)

    df["ano"] = df.data.dt.year
    df["mes"] = df.data.dt.month
    df["dia_semana"] = df.data.dt.dayofweek

    # se o ganhador for mandante coloca, M,
    # se for igual, empate, se não visitante ganha
    df["resultado"] = np.where(df.gm > df.gv, "M", 
                               np.where(df.gm == df.gv, "E", "V"))

    return df[["ID", "rodada", "data", "ano", "mes", "dia_semana", "mandante", "visitante",
            "arena", "gm", "gv", "resultado"]]

def limpar_estatisticas() -> pd.DataFrame:

    df = pd.read_csv(CONFIGURACAO.ARQ_ESTAT)
    df = df.drop_duplicates(["partida_id", "clube"])

    # os dados estão com o %, vou tirar isso, deixar apenas o numero
    for col in ["posse_de_bola", "precisao_passes"]:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace("%", "", regex=False),
            errors="coerce"
        )

    df["cartoes_est"] = df.cartao_amarelo + df.cartao_vermelho

    # linha valida = algum indicador de jogo registrado; senao e "sem dado" (zeros falsos)
    valida = (df.chutes > 0) | (df.faltas > 0) | (df.escanteios > 0) | (df.passes > 0)
    cols = ["chutes", "chutes_no_alvo", "posse_de_bola", "faltas", "escanteios", "cartoes_est"]
    # nas linhas consideradas inválidas, substituímos os zeros por NaN.
    # o ~ indica isso
    df.loc[~valida, cols] = np.nan
    df = df.rename(columns={"posse_de_bola": "posse", "partida_id": "ID"})
    return df[["ID", "clube", "chutes", "chutes_no_alvo", "posse", "faltas", "escanteios", "cartoes_est"]]

def validar_gols(partidas: pd.DataFrame) -> pd.DataFrame:

    df = pd.read_csv(CONFIGURACAO.ARQ_GOLS)
    # chamar a func que arruma os minutos dos gols
    df = pd.concat([df, arrumar_minutos(df["minuto"])], axis=1).rename(columns={"partida_id": "ID"})

    df["gol_1t"] = (df.base <= 45).astype(int)
    df["gol_2t"] = ((df.base > 45) & (df.base < 90)).astype(int)
    df["gol_final"] = (df.base >= 90).astype(int)
    por_clube = df.groupby(["ID", "clube"])[["gol_1t", "gol_2t", "gol_final"]].sum()

    # quem marcou primeiro
    gols = df.sort_values(["ID", "ordem"])
    primeiro = gols.groupby("ID").head(1).set_index("ID")
    primeiro_clube = primeiro["clube"]

    # isso aqui é feito, pois os jogos só foram cobrir gols após 2014
    cobertos = set(df.ID) | set(partidas.loc[(partidas.gm + partidas.gv == 0) & (partidas.ano >= 2014), "ID"])

    linhas = []

    # isso foi feito para juntar 
    for partida in partidas[partidas.ID.isin(cobertos)].itertuples():
        for clube in (partida.mandante, partida.visitante):
            if (partida.ID, clube) in por_clube.index:
                estatisticas = por_clube.loc[(partida.ID, clube)]
                gols_primeiro_tempo = estatisticas.gol_1t
                gols_segundo_tempo = estatisticas.gol_2t
                gols_minutos_finais = estatisticas.gol_final
            else:
                gols_primeiro_tempo = 0
                gols_segundo_tempo = 0
                gols_minutos_finais = 0

            clube_primeiro_gol = primeiro_clube.get(partida.ID, np.nan)

            marcou_primeiro = (
                np.nan
                if pd.isna(clube_primeiro_gol)
                else float(clube_primeiro_gol == clube)
            )

            linhas.append((
                partida.ID,
                clube,
                gols_primeiro_tempo,
                gols_segundo_tempo,
                gols_minutos_finais,
                marcou_primeiro
            ))
            
    return pd.DataFrame(linhas, columns=["ID", "clube", "gols_1t", "gols_2t", "gols_finais", "marcou_primeiro"])

def validar_cartoes(partidas: pd.DataFrame) -> pd.DataFrame:

    df = pd.read_csv(CONFIGURACAO.ARQ_CARTOES)
    df = df.rename(columns={"partida_id": "ID"})
    df = pd.concat([df, arrumar_minutos(df["minuto"])], axis=1)
    
    df["amarelo"] = (df.cartao == "Amarelo").astype(int)
    df["vermelho"] = (df.cartao == "Vermelho").astype(int)
    df["c2t"] = (df.base > 45).fillna(False).astype(int)

    por = df.groupby(["ID", "clube"])[["amarelo", "vermelho", "c2t"]].sum()
    cobertos = set(df.ID)

    linhas = []

    for partida in partidas[partidas.ID.isin(cobertos)].itertuples():
        for clube in (partida.mandante, partida.visitante):

            if(partida.ID, clube) in por.index:
                estatisticas_cartoes = por.loc[(partida.ID, clube)]
                quantidades_cartoes = estatisticas_cartoes.values
            else:
                quantidades_cartoes = (0, 0, 0)

            linhas.append((
                partida.ID,
                clube,
                *quantidades_cartoes
            ))

    return pd.DataFrame(linhas, columns=["ID", "clube", "amarelos", "vermelhos", "cartoes_2t"])

def juntar_operacoes(salvar: bool = True):
    partidas = limpar_partidas()
    estatisticas = limpar_estatisticas()
    gols = validar_gols(partidas)
    cartoes = validar_cartoes(partidas)
    if salvar:
        CONFIGURACAO.PROCESSED.mkdir(parents=True, exist_ok=True)
        partidas.to_csv(CONFIGURACAO.PROCESSED / "partidas_limpas.csv", index=False)
        estatisticas.to_csv(CONFIGURACAO.PROCESSED / "estatisticas_limpas.csv", index=False)
        gols.to_csv(CONFIGURACAO.PROCESSED / "gols_limpos.csv", index=False)
        cartoes.to_csv(CONFIGURACAO.PROCESSED / "cartoes_limpos.csv", index=False)

    return partidas, estatisticas, gols, cartoes

def carregar_dados_limpos():
    """Le os CSVs limpos gerados por juntar_operacoes(salvar=True)."""

    arquivos = {
        "partidas": "partidas_limpas.csv",
        "estatisticas": "estatisticas_limpas.csv",
        "gols": "gols_limpos.csv",
        "cartoes": "cartoes_limpos.csv",
    }

    for arquivo in arquivos.values():
        if not (CONFIGURACAO.PROCESSED / arquivo).is_file():
            raise FileNotFoundError(
                f"Arquivo limpo nao encontrado: {CONFIGURACAO.PROCESSED / arquivo}. "
                "Execute primeiro: python -m src.data_loader"
            )

    partidas = pd.read_csv(
        CONFIGURACAO.PROCESSED / arquivos["partidas"], parse_dates=["data"]
    )
    estatisticas = pd.read_csv(CONFIGURACAO.PROCESSED / arquivos["estatisticas"])
    gols = pd.read_csv(CONFIGURACAO.PROCESSED / arquivos["gols"])
    cartoes = pd.read_csv(CONFIGURACAO.PROCESSED / arquivos["cartoes"])

    return partidas, estatisticas, gols, cartoes

# juntando tudo
if __name__ == "__main__":
    p, e, g, c = juntar_operacoes()

