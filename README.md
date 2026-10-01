# Previsão de vazão com LSTM — Bacia Tietê-Jacaré

Prever a vazão diária de três rios, de 1 a 30 dias à frente, com uma LSTM. A pergunta: depois
de corrigir os erros metodológicos comuns nesse tipo de trabalho — vazamento de dados, falta
de uma referência trivial — a LSTM ainda bate modelos simples? Resposta: depende do posto e do
horizonte, mas sim, até um ponto que está na tabela abaixo. Projeto de pesquisa independente de
Felipe Inserti, em duas partes: reprodução crítica de Zampieri (2025), medindo o efeito
isolado de vazamento de dados; e um pipeline próprio de LSTM multi-horizonte, com avaliação
metodologicamente mais rigorosa.

Texto completo com métodos, resultados e discussão:
[docs/conclusoes.md](docs/conclusoes.md).

## Parte 2 — Resultado principal: LSTM multi-horizonte

Teste final (2015-2019, trancado até a avaliação, rodado uma única vez), 3 sementes, NSE:

| posto | lead | NSE LSTM | NSE persistência | NSE climatologia |
|---|---|---|---|---|
| FSB | 1 | 0.937 | 0.914 | 0.14 |
| FSB | 7 | 0.264 | -0.093 | 0.143 |
| FSB | 30 | 0.138 | -0.778 | 0.164 |
| GAP | 1 | 0.968 | 0.95 | 0.265 |
| GAP | 7 | 0.526 | 0.303 | 0.267 |
| GAP | 30 | 0.281 | -0.366 | 0.293 |
| IVR | 1 | 0.934 | 0.966 | -0.459 |
| IVR | 7 | 0.601 | 0.593 | -0.42 |
| IVR | 30 | 0.412 | 0.342 | -0.039 |

![Skill score por lead, teste final](results/figuras/11_lstm_teste_skill_score.png)

A LSTM supera as duas referências (persistência e climatologia, skill score > 0) até o lead
13 no FSB, até o 22 no GAP (e também no 24) e quase o horizonte todo no IVR (6-30):

| Posto | Leads em que a LSTM supera as duas referências | Total |
|---|---|---|
| IVR | 6-30 | 25/30 |
| GAP | 1-22, 24 | 23/30 |
| FSB | 1-13 | 13/30 |

A faixa grande do IVR também reflete uma referência fraca ali (a climatologia do IVR quase
não tem sinal) — não leia como "o modelo generaliza melhor no IVR". Depois dessas faixas, a
climatologia (GAP/FSB) ou a persistência (IVR) retomam a dianteira.

Tabelas completas: `results/tabelas/lstm_teste.csv` e `lstm_teste_skill_score.csv`.

## Parte 1 — Reprodução de Zampieri (2025)

Zampieri (2025) treinou uma LSTM "só com dados" pra prever cota, nos mesmos três postos.
Reproduzi o código original (versão A), depois removi o vazamento de dados (versão B) e
acrescentei a cota passada como entrada (versão C) — mesma arquitetura nas três.

NSE médio entre 3 sementes (série em cota, cm — não é vazão):

| posto | versao | NSE LSTM | NSE persistência |
|---|---|---|---|
| FSB | A | 0.876 | 0.931 |
| FSB | B | -11.535 | 0.781 |
| FSB | C | -1.328 | 0.781 |
| GAP | A | 0.848 | 0.852 |
| GAP | B | -4.616 | -0.448 |
| GAP | C | -0.935 | -0.448 |
| IVR | A | 0.863 | 0.932 |
| IVR | B | -6.393 | 0.648 |
| IVR | C | 0.368 | 0.648 |

![Previsto x observado, reprodução de Zampieri](results/figuras/07_repro_previsto_observado.png)

O NSE despenca de ~0,86 (A, com vazamento) para muito negativo (B, sem vazamento) nos três
postos: o desempenho relatado no trabalho original é um artefato da divisão aleatória dos
dados, não evidência de que a rede aprendeu a dinâmica da bacia. E a persistência bate a LSTM
em quase toda versão sem vazamento (B e C) — o trabalho original não comparava com nenhuma
referência trivial.

## Dados

| Posto | Código ANA | Rio | Área (km²) | Dado de | até |
|---|---|---|---|---|---|
| IVR | 62752000 | Jacaré-Pepira | 1800 | 1999 | 2019 |
| GAP | 62776800 | Jacaré-Guaçu | 2430 | 1981 | 2019 |
| FSB | 62800000 | Ribeirão dos Porcos | 2710 | 1974 | 2019 |

Vazão: HidroWeb/ANA, série consistida até 2019. Chuva, Tmax e Tmin: grade de Xavier (BR-DWGD).
Trechos de medição com problema (salto de cota sem chuva que explique, cota abaixo da faixa
da curva de descarga) viram falta — critério e datas em
[`configs/periodos_invalidos.yaml`](configs/periodos_invalidos.yaml).

## Método

- **Divisão**: validação walk-forward 2005-2014 (10 dobras anuais, janela expansiva; as que
  caem em períodos de medição removidos são puladas — 6 a 8 por posto); teste 2015-2019,
  trancado até a avaliação final e rodado uma única vez.
- **Modelo**: uma camada LSTM(64) + Dense(30) — prevê os 30 dias de uma vez, não um modelo por
  horizonte. Janela de entrada de 60 dias.
- **Alvo**: variação do log da vazão em relação ao dia da previsão —
  log1p(Q[t+h]) − log1p(Q[t]) — não o nível absoluto. Escolhido comparando as duas opções
  antes da validação completa.
- **Entradas**: vazão, chuva, Tmax, Tmin, seno/cosseno do dia do ano — tudo só até o dia da
  previsão, nunca dado futuro.
- **Referências**: persistência (valor de ontem) e climatologia (média histórica do dia do
  ano), nos mesmos dias que a LSTM.
- **Métricas**: NSE, KGE, PBIAS, RMSE, R², sempre em m³/s. 3 sementes por configuração.
- Decisões de método (alvo, janela, arquitetura, teste único) registradas **antes** de ver o
  resultado — histórico completo e datado em [`docs/decisoes.md`](docs/decisoes.md).

## Posto de controle: IVR

IVR tem medição comprovadamente mais inconsistente que GAP e FSB (balanço chuva-vazão
independente da régua). Hipótese pré-registrada, em duas partes: (1) desempenho do LSTM pior
que GAP e FSB; (2) pior ainda nos dias com medição sinalizada como suspeita. No teste final: a
parte 1 depende da métrica (NSE e skill score discordam); a parte 2 foi **refutada** pelas
duas métricas — dias suspeitos não saíram piores. Números e critério completos em
`docs/decisoes.md`.

## Limitações

- Dado vai só até 2019 — não captura o regime hidrológico mais recente da bacia.
- Uma arquitetura só foi testada (LSTM(64) + Dense), sem busca de hiperparâmetros.
- Sem previsão meteorológica futura nas entradas — só o observado até o dia da previsão; um
  uso operacional exigiria isso.
- Trechos de medição removidos (`configs/periodos_invalidos.yaml`) encurtam a série e deixam
  alguns anos de validação sem amostra suficiente.
- NSE depende da variância de cada subconjunto — comparar grupos com variância diferente
  (ex.: suspeito x não-suspeito) usa skill score, não NSE puro.

## Como reproduzir

Dois ambientes, propositalmente separados:

| venv | Python | Para quê |
|---|---|---|
| `.venv` | 3.14 | dados, QA, baselines, agregação, testes |
| `.venv-ml` | 3.12 + TensorFlow | treino da LSTM |

**Sempre** `source .venv-ml/bin/activate` antes de qualquer comando com TensorFlow — nunca
chamar `.venv-ml/bin/python` direto (sem isso o TensorFlow não acha a GPU no WSL).

Dados brutos não são versionados (`.gitignore`) — baixe antes de rodar:
- Vazão: HidroWeb (ANA), estações 62752000 (IVR), 62776800 (GAP), 62800000 (FSB), arquivo
  `*_Vazoes.csv`, em `data/raw/hidroweb/<posto>/`.
- Chuva, Tmax, Tmin: grade de Xavier (BR-DWGD), baixada à parte e recortada com
  `python src/ingest/xavier_recorte.py <pasta_dos_nc>`.

Depois:

```bash
scripts/reproduzir.sh --sem-treino   # minutos — usa as previsões já salvas em results/
scripts/reproduzir.sh --completo     # inclui os treinos: ~4h no total, com GPU
```

## Estrutura

```
src/ingest/     parsers dos dados brutos (HidroWeb, Xavier, Zampieri)
src/prep/       limpeza e montagem da tabela diária por posto
src/qa/         checagem de consistência (chuva x vazão)
src/eda/        figuras e tabela exploratória
src/features/   janelas, divisão treino/validação/teste, escala
src/models/     baselines e arquitetura da LSTM
src/repro/      reprodução de Zampieri (2025)
src/relatorio/  tabelas deste README, geradas a partir dos resultados
configs/        postos e períodos de medição inválida
docs/           registro de decisões, cronológico e datado
results/        tabelas, figuras e previsões (versionados)
```
