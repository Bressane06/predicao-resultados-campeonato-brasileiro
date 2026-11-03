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

def juntar_operacoes(salvar: bool = True):
    partidas = limpar_partidas()
    if salvar:
        CONFIGURACAO.PROCESSED.mkdir(parents=True, exist_ok=True)
        partidas.to_csv(CONFIGURACAO.PROCESSED / "partidas_limpas.csv", index=False)
    return partidas

# juntando tudo
if __name__ == "__main__":
    p = juntar_operacoes()
    print("partidas", p.shape)
