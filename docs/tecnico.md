# Documentação técnica

Previsão de resultados do Campeonato Brasileiro de Futebol (Série A).

## 1. Ideia do projeto

O objetivo é estimar, para uma partida futura, **quantos gols cada time deve marcar** e, a partir disso, a **probabilidade de vitória do mandante (M), empate (E) e vitória do visitante (V)**, além dos placares mais prováveis.

Em vez de classificar diretamente o resultado (M/E/V), o projeto segue a abordagem clássica da modelagem de futebol:

1. Um modelo prevê a **média de gols do mandante** (λ_M).
2. Outro modelo prevê a **média de gols do visitante** (λ_V).
3. Os gols de cada time são tratados como variáveis de **Poisson** com essas médias, formando uma matriz de probabilidade para cada placar possível (0x0, 1x0, 0x1, ...).
4. Somando as células da matriz obtém-se P(M), P(E) e P(V).

Essa abordagem tem duas vantagens sobre um classificador direto: entrega a distribuição completa de placares (útil para 0x0, over/under etc.) e produz probabilidades coerentes entre si.

O projeto também mantém um classificador Random Forest (M/E/V) e modelos alternativos de regressão (Random Forest, HistGradientBoosting, XGBoost) apenas como **comparação**. O app usa somente os modelos Poisson.

## 2. Dados

Fonte: [Kaggle, Campeonato Brasileiro de Futebol](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol). Os CSVs ficam em `data/raw/`:

| Arquivo | Conteúdo |
|---|---|
| `campeonato-brasileiro-full.csv` | Partidas: data, rodada, times, arena, placar |
| `campeonato-brasileiro-estatisticas-full.csv` | Estatísticas por clube/partida (chutes, posse, faltas, escanteios, cartões) |
| `campeonato-brasileiro-gols.csv` | Gols com minuto e clube |
| `campeonato-brasileiro-cartoes.csv` | Cartões com minuto e clube |

A base de modelagem final (`data/processed/base_modelagem.csv`) tem cerca de **9.165 partidas (2003 a 2025)** e **68 atributos numéricos**. A distribuição dos resultados é aproximadamente 49,6% M, 26,4% E e 23,9% V.

## 3. Estrutura do repositório

```
app.py                  Interface Streamlit
src/
  config.py             Caminhos e hiperparâmetros (meias-vidas, Elo, H2H, anos de split)
  data_loader.py        Limpeza e validação dos CSVs brutos
  atributos.py          Engenharia de atributos e geração de atributos de partida futura
  modelos.py            Fábricas de modelos (pipelines scikit-learn / XGBoost)
  treino.py             Pipeline do classificador M/E/V e divisão temporal dos dados
  regressao.py          Treino dos regressores de gols, previsão e avaliação
  probabilidades.py     Matriz de placares (Poisson + Dixon-Coles) e agregação em M/E/V
  avaliacao.py          Métricas de classificação, regressão e probabilidades
  persistencia.py       Salvar/carregar modelos (joblib)
modelos/                Modelos treinados (.joblib)
docs/pesos.md           Explicação dos parâmetros de EWM, Elo e H2H
```

## 4. Implementação

### 4.1 Limpeza dos dados (`data_loader.py`)

- **Partidas:** renomeia colunas, converte datas (`dd/mm/aaaa`), remove duplicatas por `ID`, cria `ano`, `mes`, `dia_semana` e o rótulo `resultado`.
- **Estatísticas:** remove o `%` de posse e precisão de passes e cria `cartoes_est`. Linhas sem nenhum indicador de jogo (chutes, faltas, escanteios, passes todos zerados) são "sem dado" e têm os valores trocados por `NaN`, para não confundir ausência de registro com zero real.
- **Gols:** o minuto (`45+2`) é normalizado. Cada gol é classificado em 1º tempo, 2º tempo ou minutos finais, e identifica-se quem marcou primeiro. Só partidas com cobertura de gols entram (o registro detalhado só existe a partir de 2014, e 0x0 a partir de 2014 também são consideradas cobertas).
- **Cartões:** conta amarelos, vermelhos e cartões no 2º tempo por clube/partida.

### 4.2 Engenharia de atributos (`atributos.py`)

Todos os atributos usam **somente informação anterior à partida**, para evitar vazamento de dados (data leakage).

1. **Histórico por clube.** A tabela tem uma linha por clube/partida, ordenada cronologicamente, com gols marcados/sofridos, pontos e aproveitamento.
2. **Médias exponenciais (EWM).** Para cada clube, calcula-se com `shift(1)` e `ewm(halflife=...)`:
   - aproveitamento, gols marcados e gols sofridos em três janelas: curta (`HL_CURTO=5`), longa (`HL_LONGO=15`) e específica do mando (`HL_LOCAL=10`);
   - estatísticas de jogo, gols por tempo e cartões (`HL_STATS=5`). O valor é anulado quando a última observação válida tem mais de `MAX_GAP_DIAS=365` dias.
3. **Contexto do clube.** Dias de descanso (limitado a 30) e quantidade de jogos históricos.
4. **Elo.** Rating pré-jogo de cada time, com K = 20, vantagem de mando de 60 pontos, ponderação pela margem de gols (`log(|saldo|+1)+1`) e regressão de 25% à média a cada nova temporada.
5. **Confronto direto (H2H).** Últimos 5 confrontos entre os dois clubes, independentemente do mando, com vitórias, empates, médias de gols, aproveitamento ponderado por decaimento (`H2H_DECAY=0.80`), desempenho do mandante em casa e média de cartões.
6. **Diferenças diretas.** `diferenca_elo`, `diferenca_aproveitamento` e `diferenca_saldo_gols`.

Os atributos de cada clube viram colunas com prefixo `mandante_` e `visitante_`. Os parâmetros são explicados em `docs/pesos.md`.

### 4.3 Divisão dos dados (`treino.dividir_dados`)

A divisão é **cronológica**, nunca aleatória:

| Conjunto | Período |
|---|---|
| Treino | até 2022 |
| Validação | 2023 |
| Teste | 2024 em diante |

### 4.4 Modelos (`modelos.py`, `regressao.py`)

Todos os modelos são `Pipeline` com imputação por mediana (valores ausentes são comuns nas estatísticas).

- **Modelo de produção:** `PoissonRegressor` (alpha=0,01) com `StandardScaler`. Um modelo para `gm` e outro para `gv`.
- **Comparação:** Random Forest, HistGradientBoosting e XGBoost, além de um `DummyRegressor` (média) como baseline.
- **Classificador de referência:** Random Forest para M/E/V (`treino.py`), com baseline pela classe mais frequente.

Os modelos Poisson são salvos como pacote `{modelo, colunas_atributos, alvo}` em `modelos/random_forest_mandante_poisson.joblib` e `modelos/random_forest_visitante_poisson.joblib`. Guardar as colunas garante que, na previsão, os atributos sejam reordenados exatamente como no treino.

### 4.5 Distribuição de placares (`probabilidades.py`)

Dados λ_M e λ_V:

1. Calcula-se `Poisson.pmf` para 0 a 10 gols de cada time e o produto externo forma a matriz de placares.
2. Aplica-se o **ajuste de Dixon-Coles** (parâmetro `rho`) nos placares baixos (0x0, 0x1, 1x0, 1x1), onde o Poisson independente costuma errar.
3. A matriz é normalizada.
4. `P(M)` é a soma do triângulo inferior, `P(E)` a diagonal e `P(V)` o triângulo superior.

Com `rho = 0` o resultado é o Poisson puro. O app usa `rho = 0`.

### 4.6 Previsão de uma partida futura

O fluxo, executado pelo app, é:

1. `adicionar_partida_futura` insere a partida sem placar no histórico.
2. `construir_base` reconstrói todos os atributos. Como o jogo futuro não tem resultado, ele não entra nos históricos, e seus atributos refletem só o passado.
3. A linha da partida é extraída e reordenada conforme as colunas do treino.
4. Os dois modelos Poisson preveem λ_M e λ_V.
5. `calcular_matriz_placares` e `calcular_probabilidades_resultado` produzem as probabilidades e os 4 placares mais prováveis.
6. `obter_historico_partida` recupera os últimos 5 jogos de cada time e os últimos 5 confrontos diretos para exibição.

### 4.7 Avaliação (`avaliacao.py`)

- **Gols:** MAE, RMSE e R².
- **Probabilidades M/E/V:** Log Loss, Brier Score e acurácia, comparados com um baseline que usa a frequência histórica das classes.
- **Classificação:** acurácia, precisão, recall e F1 (macro e por classe) e matriz de confusão.
- **Análises complementares:** acerto de 0x0 e recall das vitórias do visitante.

### 4.8 Interface (`app.py`)

App Streamlit com formulário (mandante, visitante, data, rodada, estádio). Exibe gols esperados, probabilidades M/E/V, P(0x0), os 4 placares mais prováveis, gráficos de barras, mapa de calor da matriz de placares (Plotly) e as tabelas de histórico. A data precisa ser posterior ao último jogo da base. Modelos e dados são cacheados com `st.cache_resource` e `st.cache_data`.

## 5. Como executar

```bash
pip install -r requirements.txt

# 1. Limpar data/raw/ e gerar os CSVs limpos em data/processed/
python -m src.data_loader

# 2. Ler os CSVs limpos e gerar a base de modelagem
python -m src.atributos

# 3. Treinar os modelos de gols (gera os .joblib em modelos/)
python -m src.regressao

# 4. (Opcional) classificador M/E/V de referência
python -m src.treino

# 5. Abrir o app
streamlit run app.py
```

Os arquivos `.csv` em `data/` e os `.joblib` estão no `.gitignore`: é preciso baixar os dados do Kaggle e rodar os passos 1 e 2 após clonar o repositório.

## 6. Limitações e pontos de atenção

- Futebol tem muito ruído: mesmo bons modelos acertam pouco acima do baseline, e a métrica relevante é a qualidade das probabilidades (Log Loss, Brier), não só a acurácia.
- Estatísticas detalhadas (chutes, posse, gols por minuto, cartões) só existem a partir de certo ano; antes disso os atributos ficam ausentes e são imputados.
- O Poisson assume independência entre os gols dos dois times; o ajuste Dixon-Coles corrige só parcialmente.
- O modelo não conhece escalações, lesões, suspensões, transferências ou calendário externo (copas).
- A rodada e o ano entram como atributos numéricos, e a arena é descartada.
- O estádio é pedido no app, mas não influencia a previsão.
- A base histórica termina em dezembro de 2025 (última partida: 07/12/2025). Previsões para datas posteriores usam atributos (Elo, EWM, H2H) congelados nessa data, sem nenhum resultado de 2026. Para atualizar, substitua os CSVs em `data/raw/` e refaça o pipeline (`data_loader`, `atributos`, `regressao`).

Desenvolvido por Gabriel Bressane, com apoio do Claude (Anthropic) na estruturação e refatoração do código.