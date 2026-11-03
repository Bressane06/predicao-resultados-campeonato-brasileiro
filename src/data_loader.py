# @Author: gabriel.bressane  # @Date: 2026-11-02 23:43:09  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-02 23:43:09 
# vou usar o pandas, pois já tenho mais familiaridade para manipular arquivos CSV
import numpy as np
import pandas as pd

from . import config as CONFIGURACAO

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

def juntar_operacoes(salvar: bool = True):
    partidas = limpar_partidas()
    estatisticas = limpar_estatisticas()
    if salvar:
        CONFIGURACAO.PROCESSED.mkdir(parents=True, exist_ok=True)
        partidas.to_csv(CONFIGURACAO.PROCESSED / "partidas_limpas.csv", index=False)
        estatisticas.to_csv(CONFIGURACAO.PROCESSED / "estatisticas_limpas.csv", index=False)

    return partidas, estatisticas

# juntando tudo
if __name__ == "__main__":
    p, e = juntar_operacoes()

