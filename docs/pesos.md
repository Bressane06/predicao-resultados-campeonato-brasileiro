# Explicação dos Valores Iniciais e da Função de Cada Variável

Este documento apresenta os parâmetros iniciais utilizados na engenharia de atributos (_Feature Engineering_) para o projeto de previsão de resultados do Campeonato Brasileiro.

As configurações determinam como o histórico de partidas é transformado em atributos que auxiliam os modelos de _Machine Learning_ na previsão do vencedor e da quantidade de gols por equipe.

Os parâmetros dividem-se em três grupos principais:

1. **EWM (_Exponential Weighted Moving_):** mede o desempenho recente dos clubes atribuindo maior peso a jogos mais recentes.
2. **Elo:** estima a força relativa dos clubes ao longo das temporadas.
3. **H2H (_Head-to-Head_):** extrai informações exclusivas dos confrontos diretos entre os clubes.

---

## 1. Médias Exponenciais (EWM)

A EWM (_Exponentially Weighted Moving Average_) calcula uma média ponderada exponencialmente. Diferentemente da média aritmética simples, ela dá mais relevância às partidas recentes do que às antigas.

Considere o histórico de pontos de um clube nas últimas cinco partidas:

| Partida               | Pontos |
| --------------------- | -----: |
| Jogo 1 (mais antigo)  |      0 |
| Jogo 2                |      3 |
| Jogo 3                |      1 |
| Jogo 4                |      3 |
| Jogo 5 (mais recente) |      3 |

Uma média simples resultaria em:

\[
\frac{0+3+1+3+3}{5}=2{,}0
\]

A EWM considera o mesmo histórico, mas atribui peso progressivamente maior ao Jogo 5 e menor ao Jogo 1.

O parâmetro `halflife` (meia-vida) define o ritmo com que a influência das partidas antigas diminui. Uma meia-vida de 5 significa que o peso relativo de uma observação reduz-se pela metade a cada 5 partidas.

### `HL_CURTO = 5`

Representa a meia-vida para medir a **forma recente do clube** (últimos 5 jogos).

Utilizado nos seguintes cálculos:

- `aproveitamento_curto`: aproveitamento recente em pontos.
- `gols_marcados_curto`: média ponderada de gols marcados.
- `gols_sofridos_curto`: média ponderada de gols sofridos.

**Exemplo:** se um clube iniciou o campeonato mal, mas venceu as últimas partidas, essa variável destaca a reação recente sem desconsiderar todo o histórico anterior.

### `HL_LONGO = 15`

Representa a meia-vida para medir o **desempenho histórico geral do clube** (últimos 15 jogos).

Calcula os mesmos indicadores da configuração anterior, mas com maior inércia temporal:

- `aproveitamento_longo`
- `gols_marcados_longo`
- `gols_sofridos_longo`

**Exemplo:** um clube em crise temporária pode manter um desempenho sólido ao longo de toda a temporada. Essa variável preserva essa estabilidade no modelo.

| Configuração | Meia-vida | Objetivo                                     |
| ------------ | --------: | -------------------------------------------- |
| `HL_CURTO`   |   5 jogos | Capturar oscilações e momento recente.       |
| `HL_LONGO`   |  15 jogos | Avaliar a força estrutural e a estabilidade. |

### `HL_LOCAL = 10`

Controla a meia-vida dos indicadores filtrados por mando de campo.

O histórico é dividido em dois contextos:

- Partidas como mandante.
- Partidas como visitante.

Isso permite quantificar o rendimento específico da equipe dentro ou fora de casa.

**Exemplo:** uma equipe pode ter aproveitamento alto no próprio estádio, mas fraco como visitante. O modelo utiliza esses atributos separadamente para projetar a partida.

### `HL_STATS = 5`

Controla a meia-vida das estatísticas detalhadas de cada jogo.

Aplica-se aos seguintes grupos de variáveis:

| Grupo        | Variáveis                                                                      |
| ------------ | ------------------------------------------------------------------------------ |
| Estatísticas | Chutes, chutes no alvo, posse de bola, faltas, escanteios e cartões estimados. |
| Gols         | Gols no 1º tempo, 2º tempo, placar final e primeiro a marcar.                  |
| Cartões      | Amarelos, vermelhos e cartões no 2º tempo.                                     |

**Exemplo:** se o volume de finalizações de uma equipe subiu de 8 para 16 chutes por jogo nas últimas rodadas, a EWM priorizará a média mais recente, capturando a mudança tática.

### `MAX_GAP_DIAS = 365`

Define o limite máximo de dias para considerar o histórico de estatísticas válido.

Esse parâmetro previne ruídos causados por lacunas na coleta de dados entre temporadas.

Se a última informação de cartões disponível de um clube tiver mais de 365 dias em relação à data do jogo previsto, o sistema desconsiderará o dado antigo e retornará `NaN`.

_Nota:_ esta restrição é aplicada especificamente às estatísticas calculadas via EWM com `HL_STATS`.

---

## 2. Sistema de Classificação Elo

O sistema Elo calcula a **força relativa dos clubes** atualizada a cada resultado, ajustando os pontos com base na expectativa de vitória pré-jogo.

- Vitória gera ganho de pontos; derrota gera perda.
- O impacto nos pontos depende do nível do adversário: vencer um favorito rende mais pontos do que vencer um azarão.

### `ELO_K = 20.0`

Determina a sensibilidade de atualização do Elo após cada partida.

Funciona como um fator multiplicador entre o resultado real e o esperado.

- **`ELO_K` alto:** a pontuação reage rapidamente aos últimos resultados.
- **`ELO_K` baixo:** a pontuação varia de forma mais conservadora e suave.

### `ELO_VANTAGEM_MANDANTE = 65.0`

Adiciona uma bonificação de pontos à equipe da casa no cálculo da probabilidade esperada de vitória.

Considerando duas equipes com a mesma pontuação base:

| Clube               | Elo Base | Elo Efetivo para a Partida |
| ------------------- | -------: | -------------------------: |
| Clube A (mandante)  |     1500 |                   **1565** |
| Clube B (visitante) |     1500 |                   **1500** |

A vantagem de 65 pontos eleva matematicamente a expectativa de vitória do mandante antes de a bola rolar.

### `ELO_REGRESSAO_ANUAL = 0.25`

Aplica uma compressão da pontuação Elo em direção à média (1500) na virada de cada temporada.

Com regressão de 25%, cada clube retém 75% do desvio acumulado em relação à média de 1500 pontos.

**Cálculo de Exemplo:**

Para um clube que encerrou a temporada com Elo 1700:

\[
1500 + (1700 - 1500) \times (1 - 0{,}25) = 1650
\]

O clube reinicia o campeonato seguinte com 1650 pontos. Isso evita a propagação indefinida de discrepâncias de anos anteriores.

### `ELO_INICIAL = 1500.0`

Pontuação padrão atribuída a qualquer equipe sem histórico prévio no banco de dados.

_Nota de implementação:_ na versão atual da função `calcular_classificacao_elo()`, certifique-se de substituir o valor _hardcoded_ `1500` pela constante `CONFIGURACAO.ELO_INICIAL`.

---

## 3. Confrontos Diretos (H2H)

Enquanto o Elo avalia a força global da equipe, o módulo H2H (_Head-to-Head_) quantifica especificamente o retrospecto histórico entre os dois adversários do confronto.

### `H2H_N = 5`

Limita a análise aos últimos 5 confrontos diretos disputados entre os dois clubes, independentemente de quem teve o mando de campo.

Atributos extraídos:

- Número de vitórias do mandante, empates e vitórias do visitante.
- Média de gols marcados por equipe e média total da partida.
- Aproveitamento ponderado pelo tempo.
- Vitórias do mandante atual quando jogou em casa contra este mesmo rival.
- Média de cartões nos confrontos.

Se houver menos de 5 jogos registrados no histórico, o cálculo utilizará apenas as partidas existentes.

### `H2H_DECAY = 0.80`

Fator de decaimento exponencial que aplica pesos regressivos às partidas passadas.

Com `H2H_DECAY = 0.80`, os pesos atribuídos aos 5 jogos mais recentes são:

| Confronto         |  Peso |
| ----------------- | ----: |
| 1º (mais recente) | 1,000 |
| 2º                | 0,800 |
| 3º                | 0,640 |
| 4º                | 0,512 |
| 5º (mais antigo)  | 0,410 |

**Fórmula do Aproveitamento Ponderado:**

Para um mandante com retrospecto de **Vitória (3 pts)**, **Derrota (0 pts)** e **Empate (1 pt)** nas últimas 3 partidas contra o adversário:

\[
\text{Aproveitamento} = \frac{(1{,}0 \times 3) + (0{,}8 \times 0) + (0{,}64 \times 1)}{3 \times (1{,}0 + 0{,}8 + 0{,}64)} = \frac{3{,}64}{7{,}32} \approx 49{,}7\%
\]

---

## 4. Integração dos Atributos no Modelo

Para prever um jogo como **Corinthians vs. São Paulo**, o algoritmo recebe um vetor contendo:

```
[EWM: Forma recente, saldo de gols e mando]
                   +
[Elo: Força relativa atual das equipes]
                   +
[H2H: Retrospecto histórico do clássico]
```

O modelo combina esses fatores para estimar a probabilidade de vitória, empate ou derrota, além da expectativa de gols.

---

## 5. Resumo dos Parâmetros

| Variável                | Valor Inicial | Aplicação / Objetivo                                            |
| ----------------------- | ------------: | --------------------------------------------------------------- |
| `HL_CURTO`              |             5 | Sensibilidade a momentos e forma recente.                       |
| `HL_LONGO`              |            15 | Estabilidade e desempenho histórico geral.                      |
| `HL_LOCAL`              |            10 | Desempenho filtrado por casa/fora.                              |
| `HL_STATS`              |             5 | Ponderação de estatísticas secundárias (chutes, cartões, etc.). |
| `MAX_GAP_DIAS`          |           365 | Janela temporal máxima de validade dos dados.                   |
| `ELO_K`                 |          20.0 | Taxa de atualização do ranking Elo.                             |
| `ELO_VANTAGEM_MANDANTE` |          65.0 | Bônus de pontos Elo para a equipe mandante.                     |
| `ELO_REGRESSAO_ANUAL`   |          0.25 | Compressão da pontuação Elo na virada de temporada.             |
| `ELO_INICIAL`           |        1500.0 | Pontuação base para novos clubes.                               |
| `H2H_N`                 |             5 | Quantidade de confrontos diretos recentes considerados.         |
| `H2H_DECAY`             |          0.80 | Taxa de decaimento temporal para jogos do H2H.                  |

---

## 6. Próximos Passos e Otimização de Hiperparâmetros

Estes valores são pontos de partida razoáveis. Para otimizar o desempenho do modelo, deve-se realizar uma busca de parâmetros (_Grid Search_ ou _Bayesian Optimization_) avaliada exclusivamente via **validação cruzada temporal** (_Time Series Split_):

| Parâmetro      | Intervalo Sugerido para Teste |
| -------------- | ----------------------------- |
| `HL_CURTO`     | 3, 5, 7                       |
| `HL_LONGO`     | 10, 15, 20                    |
| `HL_LOCAL`     | 5, 10, 15                     |
| `HL_STATS`     | 3, 5, 10                      |
| `MAX_GAP_DIAS` | 180, 365, 540                 |
| `ELO_K`        | 10, 20, 30, 40                |
| `H2H_N`        | 3, 5, 7, 10                   |
| `H2H_DECAY`    | 0.60, 0.70, 0.80, 0.90        |

> **Aviso:** Nunca valide a alteração desses hiperparâmetros no conjunto de teste final para evitar vazamento temporal (_data leakage_) e _overfitting_.

---

_Documento de referência dos parâmetros iniciais da engenharia de atributos do projeto._
