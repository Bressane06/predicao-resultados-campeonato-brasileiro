# @Author: gabriel.bressane  # @Date: 2026-11-03 00:41:13  # @Last Modified by:   gabriel.bressane  # @Last Modified time: 2026-11-03 00:41:13 
# configuracao de caminhos: 

from pathlib import Path


RAIZ = Path(__file__).resolve().parent.parent
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
MODELOS = RAIZ / "modelos"
ARQ_MODELO = MODELOS / "random_forest.joblib"
ARQ_MODELO_MANDANTE = MODELOS / "random_forest_mandante.joblib"
ARQ_MODELO_VISITANTE = MODELOS / "random_forest_visitante.joblib"

ARQ_FULL = RAW / "campeonato-brasileiro-full.csv"
ARQ_ESTAT = RAW / "campeonato-brasileiro-estatisticas-full.csv"
ARQ_GOLS = RAW / "campeonato-brasileiro-gols.csv"
ARQ_CARTOES = RAW / "campeonato-brasileiro-cartoes.csv"
ARQ_BASE = PROCESSED / "base_modelagem.csv"


# Meia-vida da forma recente, em quantidade de observações.
HL_CURTO = 5
HL_LONGO = 15
HL_LOCAL = 10
HL_STATS = 5
MAX_GAP_DIAS = 365

ELO_K = 20.0
ELO_VANTAGEM_MANDANTE = 60.0
ELO_REGRESSAO_ANUAL = 0.25
ELO_INICIAL = 1500.0

H2H_N = 5
H2H_DECAY = 0.80

SEED = 42
# PROPORCAO_TREINO = 0.80

CLASSES = ["M", "E", "V"]

N_ESTIMATORS = 100

ANO_FIM_TREINO = 2022
ANO_VALIDACAO = 2023
ANO_INICIO_TESTE = 2024