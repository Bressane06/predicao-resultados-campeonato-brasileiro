# configuracao de caminhos: 

from pathlib import Path


RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"

ARQ_FULL = RAW / "campeonato-brasileiro-full.csv"
ARQ_ESTAT = RAW / "campeonato-brasileiro-estatisticas-full.csv"
ARQ_GOLS = RAW / "campeonato-brasileiro-gols.csv"
ARQ_CARTOES = RAW / "campeonato-brasileiro-cartoes.csv"