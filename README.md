# Previsão multi-horizonte de vazão com LSTM — Bacia Tietê-Jacaré

Extensão de Zampieri (2025) para horizontes de 1, 7 e 30 dias nos postos IVR, GAP e FSB.

## Ambiente
- Dados: `.venv` (Python 3.14). Modelos: `.venv-ml` (Python 3.12 + TensorFlow 2.21 com GPU), `requirements-ml.txt`.
- WSL: o TensorFlow não acha sozinho as bibliotecas NVIDIA instaladas via pip. Correção (acrescentada ao fim de `.venv-ml/bin/activate`):
  `export LD_LIBRARY_PATH=$(find <site-packages>/nvidia -maxdepth 2 -type d -name lib | paste -sd:):$LD_LIBRARY_PATH`

## Parte 1 — Reprodução de Zampieri (2025)

Reproduzimos o modelo "somente dados" de Zampieri (2025) — LSTM(64)×3 + Dense(1), janela de 10
dias, 17 variáveis meteorológicas do INMET, horizonte de 1 dia — nos três postos (IVR, GAP, FSB),
usando os dados consolidados de 2022–2024 do próprio autor (`src/repro/zampieri.py`). **Atenção:**
a coluna "Vazao Observada" desses dados é, muito provavelmente, **cota em cm** (não vazão em
m³/s) — ver `docs/decisoes.md`; todas as métricas abaixo estão na unidade original da série, não
em m³/s.

Três versões do mesmo modelo, 3 sementes cada (42, 43, 44):

- **A (fiel ao original):** reproduz exatamente as escolhas do código original —
  `MinMaxScaler` ajustado no conjunto inteiro antes da divisão, janelas que atravessam dias sem
  dado, divisão **aleatória** treino/validação/teste (80/10/10) e *data augmentation* com ruído
  gaussiano (inclusive no alvo). O vazamento aqui é intencional: é o que se quer medir.
- **B (cronológica):** mesma arquitetura e hiperparâmetros, mas sem vazamento — divisão por
  **tempo** (80% treino, 10% validação, 10% finais teste), scaler ajustado só no treino, e janelas
  que não atravessam dias ausentes no calendário.
- **C (B + cota):** igual a B, acrescentando a cota dos 10 dias anteriores como entrada (o dia
  previsto nunca entra — sem vazamento).

Em todas as versões, a **persistência** (valor de ontem) é calculada nos mesmos dias de teste,
como referência trivial — ausente do trabalho original.

### Resultados (NSE e RMSE, média ± desvio entre as 3 sementes; persistência não depende da semente)

| Posto | Versão | Modelo | NSE | RMSE (cm) |
|---|---|---|---|---|
| IVR | A | LSTM | 0,86 ± 0,08 | 12,4 ± 2,7 |
| IVR | A | persistência | 0,93 ± 0,03 | 9,1 ± 2,6 |
| IVR | B | LSTM | -6,39 ± 3,22 | 13,3 ± 3,0 |
| IVR | B | persistência | 0,65 | 2,9 |
| IVR | C | LSTM | 0,37 ± 0,28 | 3,9 ± 0,9 |
| IVR | C | persistência | 0,65 | 2,9 |
| GAP | A | LSTM | 0,85 ± 0,02 | 14,0 ± 1,5 |
| GAP | A | persistência | 0,85 ± 0,07 | 13,7 ± 3,8 |
| GAP | B | LSTM | -4,62 ± 1,13 | 19,0 ± 1,9 |
| GAP | B | persistência | -0,45 | 9,7 |
| GAP | C | LSTM | -0,94 ± 0,16 | 11,2 ± 0,5 |
| GAP | C | persistência | -0,45 | 9,7 |
| FSB | A | LSTM | 0,88 ± 0,02 | 8,4 ± 1,9 |
| FSB | A | persistência | 0,93 ± 0,04 | 6,1 ± 2,1 |
| FSB | B | LSTM | -11,54 ± 1,79 | 15,9 ± 1,1 |
| FSB | B | persistência | 0,78 | 2,1 |
| FSB | C | LSTM | -1,33 ± 1,15 | 6,7 ± 1,6 |
| FSB | C | persistência | 0,78 | 2,1 |

Tabela completa (KGE, PBIAS, R², épocas, período de teste): `results/tabelas/repro_zampieri.csv`.

### Figura

`results/figuras/07_repro_previsto_observado.png` — à esquerda, previsto × observado da versão A
(LSTM e persistência, ambos ajustados à diagonal); à direita, série temporal da versão C
(observado, LSTM e persistência). A figura confirma visualmente que a persistência acompanha ou
supera a LSTM em GAP e FSB (a LSTM suaviza os picos que a persistência captura); em IVR a diferença
é mais sutil visualmente, mas presente na tabela.

### Conclusão

- **Vazamento confirmado:** o NSE da versão A (0,85–0,93) despenca para valores fortemente
  negativos na versão B (-4,6 a -11,5) ao remover apenas o vazamento metodológico (divisão
  aleatória, scaler ajustado no conjunto todo, janelas atravessando lacunas). O desempenho "bom"
  do artigo original é um artefato da metodologia, não evidência de que o modelo aprendeu a
  dinâmica da bacia.
- **Ausência de referência trivial:** o trabalho original não compara com nenhuma baseline. Aqui,
  a persistência supera a LSTM em praticamente todas as configurações sem vazamento (B e C, nos
  três postos) — inclusive quando a LSTM recebe a própria cota passada como entrada (C). Isso
  indica que, nesses dados, nem o acréscimo da cota foi suficiente para o LSTM superar um modelo
  trivial de um parâmetro.
- Escopo da Parte 1 encerrado conforme `docs/decisoes.md` (sem SMAP, sem busca de
  hiperparâmetros, sem outras arquiteturas nesta etapa).
