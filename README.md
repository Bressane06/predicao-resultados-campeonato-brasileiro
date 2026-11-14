# Previsão de Resultados do Campeonato Brasileiro

## Fonte de dados

Os dados utilizados neste projeto foram obtidos no Kaggle:

[Campeonato Brasileiro de Futebol](https://www.kaggle.com/datasets/adaoduque/campeonato-brasileiro-de-futebol)

Os arquivos CSV originais já estão em `data/raw/`. Para atualizar a base, baixe-os novamente do Kaggle e substitua-os.

---

Aplicação que prevê, para uma partida do Brasileirão, os **gols esperados de cada time**, as **probabilidades de vitória do mandante, empate e vitória do visitante** e os **placares mais prováveis**.

A previsão combina engenharia de atributos sobre o histórico de partidas, um modelo de árvores (XGBoost com perda de Poisson) e uma matriz de probabilidade de placares. Os detalhes de implementação estão em [docs/tecnico.md](docs/tecnico.md).

## Como executar

```bash
pip install -r requirements.txt

python -m src.data_loader  # limpa data/raw/ e gera os CSVs limpos em data/processed/
python -m src.atributos     # le os CSVs limpos e gera data/processed/base_modelagem.csv
python -m src.regressao     # treina os modelos de gols, salva em modelos/ e gera as tabelas de métricas (CSV)
streamlit run app.py        # abre a aplicação
```

O repositório já inclui os CSVs brutos (`data/raw/`), os modelos XGBoost usados pelo app e as tabelas de métricas (`modelos/metricas_regressao.csv` e `modelos/metricas_probabilidades.csv`). Por isso, após `pip install -r requirements.txt`, basta executar `streamlit run app.py`. Os três primeiros comandos refazem todo o pipeline do zero; `data/processed/` é gerado por eles e não é versionado. O treino salva em `modelos/` apenas os modelos usados pelo app.

---

## Fluxo da aplicação

Do clique em "Prever resultado" até a tela final:

```
┌──────────────────────────────────────────────────────────────┐
│ 1. Formulário (app.py)                                       │
│    mandante, visitante, data, rodada, estádio                │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 2. Validações                                                │
│    • estádio preenchido                                      │
│    • data posterior à última partida da base                 │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 3. Atributos da partida futura (atributos.py)                │
│    a. adiciona a partida sem placar ao histórico             │
│    b. reconstrói a base: EWM, Elo, H2H, descanso             │
│    c. extrai a linha da partida e remove colunas de ID       │
│    d. reordena as colunas como no treino                     │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 4. Modelos de gols (regressao.py)                            │
│    XGBoost GM  → λ_M  (gols esperados do mandante)           │
│    XGBoost GV  → λ_V  (gols esperados do visitante)          │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 5. Matriz de placares (probabilidades.py)                    │
│    P(i x j) = Poisson(i; λ_M) · Poisson(j; λ_V)              │
│    (ajuste Dixon-Coles opcional)                             │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 6. Agregação                                                 │
│    P(M), P(E), P(V), P(0x0) e os 4 placares mais prováveis   │
└──────────────────────────────┬───────────────────────────────┘
                               ▼
┌──────────────────────────────────────────────────────────────┐
│ 7. Exibição (app.py)                                         │
│    métricas, gráficos, mapa de calor e últimos 5 jogos       │
│    de cada time e últimos 5 confrontos diretos               │
└──────────────────────────────────────────────────────────────┘
```

### Passo a passo

1. **Seleção dos times.** O usuário escolhe o mandante; a lista de visitantes exclui o mandante escolhido. Informa também data, rodada e estádio.
2. **Validação.** A previsão é bloqueada se o estádio estiver vazio ou se a data não for posterior ao último jogo da base. O motivo é que todo atributo é calculado só com o passado.
3. **Geração de atributos.** `gerar_atributos_partida_futura` insere o jogo como uma linha sem placar e roda o mesmo pipeline do treino. Como não há resultado, o jogo não entra nos históricos, e seus atributos usam apenas partidas anteriores. O estádio é descartado nessa etapa.
4. **Previsão dos gols.** Os dois modelos XGBoost recebem os mesmos atributos e devolvem λ_M e λ_V. Valores negativos são truncados em zero.
5. **Distribuição dos placares.** Os λ geram uma matriz de probabilidades de 0 a 10 gols por time.
6. **Probabilidades finais.** As células da matriz são somadas por tipo de resultado.
7. **Apresentação.** O app mostra os gols esperados, as probabilidades, os 4 placares mais prováveis, os gráficos, o mapa de calor da matriz e o histórico recente.

---

## Parte matemática

### 1. Pontos e aproveitamento

Para cada clube em cada jogo:

$$
\text{pontos} = \begin{cases} 3 & \text{vitória} \\ 1 & \text{empate} \\ 0 & \text{derrota} \end{cases}
\qquad
\text{aproveitamento} = \frac{\text{pontos}}{3}
$$

### 2. Média exponencial (EWM)

A forma de um clube é uma média em que jogos recentes pesam mais. Para uma meia-vida `h` (em jogos):

$$
\alpha = 1 - 2^{-1/h}
\qquad
\bar{x}_t = \frac{\sum_{i=0}^{t-1} (1-\alpha)^i \, x_{t-1-i}}{\sum_{i=0}^{t-1} (1-\alpha)^i}
$$

O índice `t-1-i` (com `shift(1)`) garante que o jogo atual nunca entra no próprio atributo. Um jogo com `h` partidas de idade pesa metade do jogo mais recente.

| Janela                    | Meia-vida h | α       |
| ------------------------- | ----------- | ------- |
| Curta (`HL_CURTO`)        | 5           | ≈ 0,129 |
| Longa (`HL_LONGO`)        | 15          | ≈ 0,045 |
| Mesmo mando (`HL_LOCAL`)  | 10          | ≈ 0,067 |
| Estatísticas (`HL_STATS`) | 5           | ≈ 0,129 |

Nas estatísticas de jogo, o valor é descartado (`NaN`) se a última observação válida for mais antiga que 365 dias.

### 3. Rating Elo

Probabilidade esperada de vitória do mandante, com bônus de 60 pontos pelo mando:

$$
E = \frac{1}{1 + 10^{-(R_M + 60 - R_V)/400}}
$$

Após o jogo, com `S = 1`, `0,5` ou `0` (vitória, empate, derrota do mandante) e `K = 20`:

$$
\Delta = K \cdot \big(\ln(|g_M - g_V| + 1) + 1\big) \cdot (S - E)
\qquad
R_M \leftarrow R_M + \Delta,\quad R_V \leftarrow R_V - \Delta
$$

O fator logarítmico dá mais peso a goleadas. A cada nova temporada, os ratings regridem 25% à média inicial de 1500:

$$
R \leftarrow 1500 + 0{,}75\,(R - 1500)
$$

### 4. Confronto direto (H2H)

Considera os últimos 5 jogos entre os dois clubes, em qualquer mando. Com `i = 0` sendo o mais recente e `p_i` os pontos do mandante atual naquele jogo:

$$
w_i = 0{,}8^{\,i}
\qquad
\text{h2h\_aproveitamento} = \frac{\sum_i w_i \, p_i}{3 \sum_i w_i}
$$

### 5. Modelo de gols: gradient boosting com perda de Poisson

Cada time faz `G` gols, com `G ~ Poisson(λ)`. O modelo de produção é um **XGBoost** (árvores de decisão em sequência) que prevê λ. A previsão passa por uma exponencial, o que mantém λ sempre positivo:

$$
\lambda = \exp\!\big(F(\mathbf{x})\big)
\qquad
F(\mathbf{x}) = \sum_{m=1}^{500} \eta \, h_m(\mathbf{x})
$$

em que `h_m` é a m-ésima árvore (profundidade máxima 3) e `η = 0,03` é a taxa de aprendizado. Cada árvore nova é ajustada para reduzir o erro que as anteriores deixaram, minimizando o negativo da log-verossimilhança de Poisson (`objective = "count:poisson"`), com regularização L2 nas folhas:

$$
\min \; \sum_{k=1}^{n} \big(\mu_k - y_k \ln \mu_k\big) \;+\; \sum_{m} \Omega(h_m)
\qquad \mu_k = \exp\!\big(F(\mathbf{x}_k)\big)
$$

São treinados dois modelos independentes: um para os gols do mandante (λ_M) e outro para os do visitante (λ_V). O Poisson linear (GLM), a rede neural (MLP), o Random Forest e o HistGradientBoosting são treinados apenas para comparação.

### 6. Matriz de placares

Probabilidade de um time marcar `k` gols:

$$
P(G = k) = \frac{e^{-\lambda}\,\lambda^{k}}{k!}
$$

Assumindo independência entre os dois times, a probabilidade do placar `i x j`, de 0 a 10 gols, é:

$$
P(i, j) = P(G_M = i;\ \lambda_M)\cdot P(G_V = j;\ \lambda_V)
$$

### 7. Ajuste de Dixon-Coles

O Poisson independente costuma errar nos placares baixos. O ajuste com parâmetro `ρ` multiplica quatro células e depois renormaliza a matriz para somar 1:

| Placar | Fator               |
| ------ | ------------------- |
| 0 x 0  | `1 − λ_M · λ_V · ρ` |
| 0 x 1  | `1 + λ_M · ρ`       |
| 1 x 0  | `1 + λ_V · ρ`       |
| 1 x 1  | `1 − ρ`             |

Com `ρ = 0` o resultado é o Poisson puro, que é o valor usado pelo app.

### 8. Probabilidades do resultado

Somando as células da matriz:

$$
P(M) = \sum_{i > j} P(i,j) \qquad
P(E) = \sum_{i = j} P(i,j) \qquad
P(V) = \sum_{i < j} P(i,j)
$$

Os 4 placares mais prováveis são as 4 maiores células da matriz.

### 9. Métricas de avaliação

Para `N` partidas, com `p_k` a probabilidade prevista da classe `k` e `o_k` igual a 1 para a classe real e 0 nas demais:

$$
\text{Log Loss} = -\frac{1}{N}\sum_{n} \ln p_{n,\,y_n}
\qquad
\text{Brier} = \frac{1}{N}\sum_{n}\sum_{k \in \{M,E,V\}} (p_{n,k} - o_{n,k})^2
$$

Para os gols esperados usa-se MAE, MSE, RMSE e R². Quanto menores Log Loss e Brier, melhor.

---

## Exemplo numérico

Suponha que os modelos devolvam **λ_M = 1,6** e **λ_V = 0,9**, com `ρ = 0`:

- P(0 x 0) = e^(−1,6) · e^(−0,9) = e^(−2,5) ≈ **8,2%**
- Placares mais prováveis: 1 x 0 (13,1%), 1 x 1 (11,8%), 2 x 0 (10,5%) e 2 x 1 (9,5%)
- **P(M) ≈ 53,8%**, **P(E) ≈ 25,0%**, **P(V) ≈ 21,2%**

## Limitações

- A previsão é probabilística. Mesmo um bom modelo erra muitos jogos, e a qualidade deve ser lida por Log Loss e Brier, não só por acurácia.
- O Poisson assume gols independentes entre os times.
- O modelo não conhece escalações, lesões, suspensões ou calendário de outras competições.
- A base histórica termina em dezembro de 2025 (última partida: 07/12/2025). Previsões para datas posteriores usam o Elo, a forma recente e o histórico dos confrontos congelados nessa data: nenhum resultado de 2026 é considerado. Para atualizar, substitua os CSVs em `data/raw/` por uma versão mais recente do Kaggle e repita os passos de "Como executar".
