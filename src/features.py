# @Author: gabriel.bressane  # @Date: 2026-11-03 12:35:17  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-03 12:35:17 
# esse arquivo será responsável por 


import pandas as pd
import numpy as np

from . import config as CONFIGURACAO


def criar_base_clubes(salvar: bool = True) -> pd.DataFrame:

    # carregando os arquivos processados
    partidas = pd.read_csv(CONFIGURACAO.PROCESSED / "partidas_limpas.csv",parse_dates=["data"])
    estatisticas = pd.read_csv(CONFIGURACAO.PROCESSED / "estatisticas_limpas.csv")
    gols = pd.read_csv(CONFIGURACAO.PROCESSED / "gols_limpos.csv")
    cartoes = pd.read_csv(CONFIGURACAO.PROCESSED / "cartoes_limpos.csv")

    # criando uma linha para o mandante de cada partida
    mandantes = partidas[[
        "ID", "data", "rodada", "ano", "mes", "dia_semana",
        "mandante", "visitante", "gm", "gv", "arena"
    ]].copy()

    mandantes = mandantes.rename(columns={
        "mandante": "clube",
        "visitante": "adversario",
        "gm": "gols_marcados",
        "gv": "gols_sofridos"
    })

    mandantes["mandante"] = 1

    # criando uma linha para o visitante de cada partida
    visitantes = partidas[[
        "ID", "data", "rodada", "ano", "mes", "dia_semana",
        "mandante", "visitante", "gm", "gv", "arena"
    ]].copy()

    visitantes = visitantes.rename(columns={
        "visitante": "clube",
        "mandante": "adversario",
        "gv": "gols_marcados",
        "gm": "gols_sofridos"
    })

    visitantes["mandante"] = 0

    # unindo mandantes e visitantes
    base = pd.concat(
        [mandantes, visitantes],
        ignore_index=True
    )

    # resultado da partida na perspectiva de cada clube
    base["resultado_clube"] = np.where(
        base.gols_marcados > base.gols_sofridos, "V",np.where(
            base.gols_marcados == base.gols_sofridos, "E", "D"))

    # pontuação obtida pelo clube na partida
    base["pontos"] = np.select([base.resultado_clube == "V",
                                base.resultado_clube == "E"],
                                [3, 1],default=0)

    # estatísticas de cada clube
    base = base.merge(
        estatisticas,
        on=["ID", "clube"],
        how="left",
        validate="one_to_one"
    )

    # informações dos gols
    base = base.merge(
        gols,
        on=["ID", "clube"],
        how="left",
        validate="one_to_one"
    )

    # informações dos cartões
    base = base.merge(
        cartoes,
        on=["ID", "clube"],
        how="left",
        validate="one_to_one"
    )

    base = base.sort_values(["clube", "data", "ID"]).reset_index(drop=True)

    if salvar:
        CONFIGURACAO.PROCESSED.mkdir(parents=True,exist_ok=True)
        base.to_csv(CONFIGURACAO.PROCESSED / "desempenho_clubes.csv",index=False)

    return base


if __name__ == "__main__":
    df = criar_base_clubes()
