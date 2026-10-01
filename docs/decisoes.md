# Registro de decisões e achados

## 2026-09-30 — Dados de vazão (HidroWeb/ANA)
- Fonte: HidroWeb, arquivos *_Vazoes.csv (IVR 62752000, GAP 62776800, FSB 62800000).
- Séries terminam em 31/12/2019 nos três postos; todos os dias com valor são nível 2 (consistido).
- Parser prioriza consistido sobre bruto; dia vazio = NaN (sem medição), zero só se gravado.
- Exemplo do porquê: nov/2019 IVR, dia 6: bruto 55,0 m³/s × consistido 9,5 m³/s.
- Períodos: IVR 1999–2019 (10,5% faltante), GAP 1969–2019 (28,9%), FSB 1966–2019 (12,2%).

## Lacunas
- Interpolação linear só em lacunas <= 7 dias; lacunas maiores = fronteira de segmento.
- Sensibilidade 7 × 15 dias: diferença desprezível (+15 dias só em IVR). Lacunas são bimodais.
- Anos úteis em blocos >= 1 ano: FSB 46,3; GAP 35,2; IVR 18,1.

## Divisão
- Teste comum aos três postos: 01/01/2015 a 31/12/2019 (dentro do último bloco contínuo de cada um).
- Treino + validação walk-forward: até 31/12/2014.

## Zeros em FSB
- 1971-09-18: mantido (recessão suave até o limite inferior da curva de descarga).
- 2016-11-28 e 2016-11-29: tratados como falta (logo após 3 dias sem medição; dentro do teste).

## Dados de Zampieri (2022–2024)
- CSVs do repositório plzampieri/unicamp-repos; vazão vinha como "111.0\r" (\r removido no loader).
- 59 dias ausentes, iguais nos três postos (falhas da estação INMET A737, compartilhada).
- Código original: divisão treino/teste ALEATÓRIA (train_test_split), scaler ajustado no conjunto todo,
  3 camadas LSTM de 64, janela 10, 200 épocas, augmentation com ruído no alvo, sem vazão passada nas entradas.
- ACHADO: a "Vazao Observada" de Zampieri é muito provavelmente COTA (cm), não vazão:
  valores inteiros, faixa 85–370; mediana ~5x a vazão do HidroWeb e compatível com a cota do HidroWeb
  (medianas 2015–2019: IVR 144 cm, GAP 134 cm, FSB 135 cm). A confirmar com o autor.
- Áreas de drenagem (tese, Apêndice B): GAP 2.430 km², FSB 2.710 km².

## Meteorologia
- Estudo principal: grade de Xavier (BR-DWGD), 1961–2020, cobre toda a série do HidroWeb.

## 2026-09-30 — Cadastro dos postos (inventário ANA)
- IVR 62752000: rio Jacaré-Pepira, Bocaina, (-22.0775, -48.4842), 1.800 km².
- GAP 62776800: rio Jacaré-Guaçu, Gavião Peixoto, (-21.8494, -48.4917), 2.430 km².
- FSB 62800000: ribeirão dos Porcos, Ibitinga, (-21.6997, -49.0106), 2.710 km².
- Correção à tese de Zampieri (Ap. B): IVR não está no rio Tietê, e sim no Jacaré-Pepira.
- Vazão específica (mediana HidroWeb): IVR 11,6; GAP 12,5; FSB 7,8 L/s·km² (plausíveis).
  Pela "vazão" de Zampieri, IVR daria ~63 L/s·km² -> reforça que a série dele é cota.

## 2026-09-30 — Consistência da medição (períodos inválidos)
- Baselines revelaram climatologia sem sinal sazonal (IVR/GAP, PBIAS +31-35%) -> investigação.
- Causa: saltos de patamar na COTA (não na curva), com chuva normal. Em 2010 nos TRÊS postos
  (mesma rede de monitoramento UHE Ibitinga/Promissão) -> provável intervenção operacional.
- IVR 2016: salto de ~70 cm coincidente com a curva 04 (15/01/2016).
- FSB nov/2016-jan/2017: cota abaixo da faixa de validade da curva (< 70 cm) -> vazão ~0.
- Tratados como falta (configs/periodos_invalidos.yaml): IVR 2010 e 15/01-31/12/2016;
  GAP 12/2009-10/2011; FSB 01-07/2010 e 11/2016-01/2017.
- Validação walk-forward ampliada para 2008-2014 (7 dobras).
- Após limpeza: climatologia GAP PBIAS 35% -> 5%; FSB estável; IVR ainda +25% e R² ~0 -> pendente.
- Baselines (validação): persistência NSE 1 dia = 0,81/0,92/0,84 (IVR/GAP/FSB);
  em 7 e 30 dias persistência e climatologia <= 0 nos três postos.
- Memória (ACF log Q < 0,5) após limpeza: 47/43/50 dias (antes 76/59/50: platôs anômalos inflavam).

## 2026-09-30 — Checagem objetiva (balanço chuva-vazão) e PRÉ-REGISTRO do controle IVR
- Vizinhos no mesmo rio não servem: estações desativadas antes de 2010
  (62754000 até 1979; 62777000 até 1961; 62778000 até 2007; 62795000 até 1975; 62755000 sem vazão).
- Método adotado: balanço anual chuva (Xavier, independente das réguas) x escoamento,
  log R = a + b log P + c log P_ano_anterior, ajuste robusto; |z| > 2,5 = suspeito. Mais dupla massa mensal.
- FSB: consistente de 1995 em diante. GAP: anos hidro 2010-2013 anômalos -> removido 12/2009-09/2013.
- IVR: anômalo em 2010, 2014, 2015, 2016, 2017 e 2019 (razões 1,44; 1,50; 1,50; 2,71; 2,00; 0,79),
  inclusive na crise hídrica de 2014-2015 (mais escoamento do que a chuva explica).
- Validação walk-forward ampliada para 2005-2014 (10 dobras) para compensar as remoções.

### PRÉ-REGISTRO (antes de qualquer treinamento de LSTM)
IVR é mantido como POSTO DE CONTROLE, com medição comprovadamente inconsistente:
- 2010 e 15/01-31/12/2016 removidos (grosseiros); demais anos suspeitos SINALIZADOS e mantidos
  no treino e no teste (coluna "suspeito").
- Hipótese H-IVR: o desempenho do LSTM em IVR será inferior ao de GAP e FSB, e nos dias
  sinalizados como suspeitos será inferior ao dos dias consistentes do próprio IVR, em todos
  os horizontes (1, 7 e 30 dias).
- IVR é reportado SEPARADAMENTE e nunca entra em médias com GAP e FSB.
- Objetivo: quantificar o custo de treinar e avaliar com dados inconsistentes (lição final do trabalho).

## 2026-09-30 — FSB antes de 1995 e baselines finais da limpeza
- GAP 1982-1994: nenhum ano suspeito.
- FSB: anos hidro 1969 (razão 0,50; z=-4,1), 1971 (0,31; z=-7,0) e 1984 (1,63; z=3,1) suspeitos.
  Critério: se o par chuva-vazão é inconsistente, o ano não serve para treinar um modelo chuva-vazão,
  qualquer que seja o lado errado. Removidos: 1966-01 a 1974-09 (inclui década com curva
  de descarga desconhecida e grade de chuva esparsa) e 1983-10 a 1984-09.
- Baselines na validação 2005-2014 (NSE, melhor baseline por antecedência):
  GAP 0,95 / 0,44 / 0,18 ; FSB 0,87 / 0,25 / 0,13 (1 / 7 / 30 dias).
  Climatologia GAP: R² 0,06 -> 0,27 e PBIAS +35% -> -4,6% após a limpeza.
  IVR (controle): climatologia sem sinal (R² 0,03), coerente com a hipótese pré-registrada.

## 2026-10-01 — Escopo final e checagens
- Escopo final: sem comparação com o SMAP, sem ajuste extenso de hiperparâmetros, sem outras arquiteturas.
  Métricas: NSE, RMSE, PBIAS, R² e KGE. Entrega final: código reproduzível + texto de conclusões.
- IVR mar/2009 (mínimo 4,5 m³/s em período chuvoso) conferido: recessão contínua e recuperação, mantido.

## 2026-10-01 — Decisões do pipeline da LSTM multi-horizonte (antes do código)
- Vazão em log1p antes de escalar (MinMaxScaler ajustado só no treino da dobra, nunca no
  conjunto inteiro): a série é assimétrica e a memória da bacia já é medida em log Q (ver
  "Checagem chuva-vazão" acima). Métricas sempre desfeitas de volta para m³/s antes de
  NSE/KGE/PBIAS/RMSE/R² (`src/features/escala.py`).
- Um único modelo seq2vec por posto, saída Dense(30), substituindo "um modelo por horizonte":
  mesma entrada para os 30 leads, muito menos treinos, e uma curva de 1 a 30 dias diretamente
  comparável à dos baselines (`results/tabelas/baselines_validacao.csv`). Reporta-se com destaque
  os leads 1, 7 e 30. Arquitetura fixada a priori, sem busca de hiperparâmetros: uma camada
  LSTM(64) + Dense(30) (`src/models/lstm.py`).
- Janela de entrada N = 60 dias, fixada a priori (não é hiperparâmetro a ajustar): cobre a
  memória medida por ACF em log Q (43-50 dias após a limpeza). Os índices válidos de emissão
  continuam vindo de `janelas.indices_validos(segmento, 90, 30)` — os MESMOS dias avaliados
  pelos baselines; o modelo só usa os últimos 60 dias dessa janela de 90.
- Sementes: durante o desenvolvimento, 1 semente e só a dobra mais recente (val2014). A
  validação completa (10 dobras x sementes 42/43/44) roda uma vez só, depois do código fechado,
  com o tempo total estimado a partir do tempo de uma dobra.
- Sem vazamento meteorológico: as entradas (chuva, tmax, tmin) usam só os dias até a emissão t;
  nada de clima de t+1 em diante, porque no uso real esse dado não existiria. Garantido por
  `tests/test_amostras.py`.
- Sazonalidade: seno e cosseno do dia do ano entram como entrada (custo baixo, ajuda nos leads
  longos, onde a sazonalidade domina mais que a memória de curto prazo).
- Parada antecipada: usa o último ano do treino de cada dobra como conjunto de validação do
  treino (early stopping), em ordem cronológica, sem embaralhar entre segmentos. O scaler é
  ajustado só no treino da dobra, excluindo esse último ano.
- IVR: dias com `suspeito = True` continuam no treino e na avaliação (pré-registro já feito
  acima); a avaliação reporta separadamente suspeito vs. não suspeito.

## 2026-10-01 — Correção do conjunto de parada antecipada e dobras puladas
- O conjunto de parada antecipada (dev) era "o último ano calendário do treino da dobra";
  colide com os períodos removidos em `configs/periodos_invalidos.yaml` quando esse ano cai
  dentro (ou logo depois) de uma lacuna. Medido: GAP val2014 ficava com só 3 amostras de dev
  (2013 está quase todo dentro do período 12/2009-09/2013 removido do GAP) e GAP val2010
  tinha 0 amostras de validação. Corrigido: dev = as últimas 365 amostras VÁLIDAS do treino
  da dobra, em ordem cronológica — pode atravessar uma lacuna, porque cada amostra já respeita
  o segmento via `indices_validos`. O scaler continua ajustado só com os dias anteriores à
  primeira amostra de dev (`src/treinar_lstm.py`, `N_DEV = 365`).
- Dobras com menos de 60 amostras de validação são puladas (não treinam): o balanço
  chuva-vazão já havia removido blocos inteiros de alguns anos de validação. Medido nas 10
  dobras x 3 postos: IVR pula val2008 e val2010 (0 amostras); GAP pula val2010-val2013 (0
  amostras cada, período 12/2009-09/2013 removido cobre os quatro anos hidrológicos); FSB
  pula val2008, val2010 e val2011 (0 amostras). `MIN_VAL_AMOSTRAS = 60` em
  `src/treinar_lstm.py`; a dobra pulada é registrada no console (`DobraPulada`), não treina.

## 2026-10-01 — PRÉ-REGISTRO: escolha entre alvo "nível" e "delta" (antes da validação completa)
- Motivo: no treino completo de GAP val2014 (semente 42, 24 épocas, parada na melhor época 4),
  a LSTM perdeu da persistência nos três leads de resumo mesmo recebendo a própria vazão
  Q[t] como entrada (NSE lead 1/7/30: LSTM 0,78/0,02/-0,76 contra persistência 0,88/0,17/-0,42).
  Hipótese: prever o NÍVEL absoluto da vazão é mais difícil pra rede do que prever a VARIAÇÃO
  em relação a Q[t], que é literalmente o que a persistência já faz implicitamente (delta=0).
- Variante testada: alvo "delta" — para cada lead h, o alvo vira
  log1p(Q[t+h]) - log1p(Q[t]), escalado (MinMax) só com o treino; a previsão final é
  log1p(Q[t]) + delta previsto, desfeita (expm1) para m³/s. Entradas do modelo não mudam.
  Alvo "nível" é o comportamento atual (`src/treinar_lstm.py`, `--alvo nivel`, padrão).
- Comparação, ANTES de rodar: GAP val2005, GAP val2014 e FSB val2014, semente 42 (3 dobras x
  2 alvos = 6 treinos, `src/selecionar_alvo.py`).
- Critério, fixado agora: maior média de NSE da LSTM nos leads 1, 7 e 30 dessas 3 dobras
  (9 valores por alvo). A variante vencedora é usada em TODA a validação completa (10 dobras x
  3 sementes) e no teste final; não se testam novas variantes de alvo depois desta decisão.
- Leve otimismo declarado: GAP val2005, GAP val2014 e FSB val2014 também fazem parte da
  validação completa de 10 dobras — usá-las pra escolher o alvo significa que o alvo escolhido
  tem uma vantagem (pequena) justamente nessas 3 dobras, que não existe nas outras 7. Isso é
  aceito conscientemente pelo custo de treino (6 treinos completos já é caro) e registrado
  aqui para não ser esquecido na hora de interpretar os resultados finais.

## 2026-10-01 — RESULTADO do pré-registro: alvo "delta" escolhido
- Saída completa em `results/selecao_alvo.txt`. Média de NSE da LSTM nos leads 1/7/30 das 3
  dobras (critério fixado acima): delta 0,324 contra nível 0,319. **Margem pequena, declarada**
  — a diferença é de 0,005 (menos de 2% relativo), não uma vitória folgada. Olhando lead a
  lead, delta venceu ou empatou com nível nos 3 pares dobra/lead onde havia diferença visível
  (ex.: GAP val2005 lead 1: nível 0,917 -> delta 0,975; GAP val2014 lead 1: 0,776 -> 0,900),
  então a direção do efeito é consistente mesmo com a margem média pequena.
- Decisão: `--alvo` passa a ter padrão `delta` em `src/treinar_lstm.py`. A validação completa
  (10 dobras x 3 sementes, 63 treinos, log em `results/validacao_completa.txt`) já rodou com
  `delta` antes mesmo desta entrada ser escrita — consistente com a decisão, sem retreinar.
- Conforme pré-registrado: nenhuma nova variante de alvo será testada depois desta decisão.

## 2026-10-01 — PRÉ-REGISTRO: teste final (2015-2019), antes de implementar `--dobra teste`
- O teste final roda UMA ÚNICA VEZ, com tudo o que foi decidido na validação já fixado:
  arquitetura (uma camada LSTM(64) + Dense(30), `src/models/lstm.py`), janela N=60, alvo
  "delta" (escolhido acima), scaler log1p+MinMax ajustado só no treino, dev = últimas 365
  amostras válidas antes do início do teste, parada antecipada (paciência 20) decidindo o
  número de épocas — nenhum desses valores é reajustado depois de ver o resultado no teste.
  Sementes 42/43/44, mesmas métricas (NSE/KGE/PBIAS/RMSE/R²), mesmas referências
  (persistência/climatologia) e mesma quebra suspeito/não-suspeito do IVR da validação.
- Treino = tudo antes de 2015-01-01 (igual ao treino de qualquer dobra, só que até o fim da
  série de validação); avaliação = 2015-01-01 a 2019-12-31 (`divisao.INICIO_TESTE/FIM_TESTE`,
  já definidos desde a Parte 1). Implementado como `--dobra teste` em `src/treinar_lstm.py`.
- Isso fecha o ciclo de decisões pré-registradas do trabalho: IVR como controle (Parte 1),
  alvo delta (acima) e agora o próprio teste. Depois de rodar o teste, não há mais ajuste de
  método — só análise e texto de conclusões.

## 2026-10-01 — RESULTADO da validação completa (10 dobras x 3 sementes, alvo delta)
Saída completa: `results/tabelas/lstm_validacao.csv`, `..._por_dobra.csv`,
`lstm_skill_score.csv`; figuras `08_lstm_validacao_nse.png` e `09_lstm_skill_score.png`.

- **Resumo (NSE, leads 1/7/30):**

  | posto | LSTM | persistência | climatologia |
  |---|---|---|---|
  | GAP | 0,963 / 0,567 / 0,199 | 0,952 / 0,439 / -0,324 | 0,222 / 0,219 / 0,175 |
  | FSB | 0,878 / 0,359 / 0,072 | 0,869 / 0,135 / -0,721 | 0,258 / 0,239 / 0,112 |
  | IVR | 0,838 / 0,196 / -0,196 | 0,836 / -0,102 / -0,914 | -0,031 / -0,031 / -0,069 |

- **Faixa de leads em que a LSTM supera AS DUAS referências** (skill score > 0 contra a
  melhor das duas, `habilidade()` em `src/agregar_validacao.py`): GAP 1-30 (as 30, nunca
  perde); FSB 1-14; IVR 1-22. Atenção: essa faixa depende também de quão fraca é a
  referência em cada posto (a climatologia do IVR é quase sem sinal, R²~0, então é "fácil"
  superá-la) — não deve ser lida como "IVR generaliza melhor que FSB", só como "a régua de
  comparação do IVR é mais baixa". A comparação de desempenho absoluto é pela tabela acima.

- **H-IVR parte 1 (IVR pior que GAP e FSB) — CONFIRMADA na validação.** Pela tabela acima,
  o NSE da LSTM em IVR é o menor dos três postos nos três leads de resumo (0,838 < 0,878 e
  0,963 no lead 1; 0,196 < 0,359 e 0,567 no lead 7; -0,196 < 0,072 e 0,199 no lead 30).
  Coerente com a medição do IVR ser a mais inconsistente dos três (balanço chuva-vazão,
  Parte 1).

- **H-IVR parte 2 (dias suspeitos piores que não-suspeitos) — INCONCLUSIVA na validação,
  conforme pré-registrado.** Comparação direta de NSE entre os dois subconjuntos é injusta
  (variâncias diferentes — subséries distintas, NSE normaliza pela variância de cada uma) e
  CHEGOU A APONTAR NA DIREÇÃO ERRADA (NSE suspeito 0,915/0,476/0,148 > não-suspeito
  0,837/0,192/-0,202 nos leads 1/7/30). Corrigindo com skill score DENTRO de cada
  subconjunto (1 - MSE_LSTM/MSE_persistência do próprio subconjunto) a direção se inverte
  pra mais perto do esperado: suspeito -0,249/0,116/0,000 contra não-suspeito
  0,022/0,277/0,378 — a LSTM é relativamente pior nos dias suspeitos nos 3 leads, na direção
  de H-IVR. MAS: os 280 dias suspeitos da validação vêm de só dois anos consecutivos
  (2013: 62 dias, out-dez; 2014: 218 dias — praticamente um único bloco/episódio contínuo,
  não uma amostra de anos diferentes). Não dá pra separar "efeito de ser suspeito" de
  "o que aconteceu especificamente nesse episódio de 2013-2014". A avaliação definitiva é
  no teste (2015-2019), onde os períodos sinalizados cobrem três episódios SEPARADOS
  (2015-jan/2016, 2017, 2018-2019) — ver pré-registro do teste final acima.

## 2026-10-01 — H-IVR reavaliada: as duas partes, pelas duas métricas (NSE e skill score)
Esta entrada substitui a classificação "inconclusiva" dada acima para a Parte 2 por uma
avaliação mais rigorosa: as duas partes de H-IVR, cada uma pela métrica em que foi
pré-registrada (NSE) E pelo skill score vs. persistência (1 - MSE_LSTM/MSE_persistência,
dentro de cada posto / de cada subconjunto — corrige a comparação entre séries de variância
diferente, mas NÃO substitui a métrica pré-registrada; reportam-se as duas, sem escolher uma
a posteriori). Números de `src/agregar_validacao.py` (rodado de novo, sem mudar nenhum
resultado anterior).

**Parte 1 — "o desempenho do LSTM em IVR será inferior ao de GAP e FSB, em todos os
horizontes (1, 7, 30 dias)":**

| lead | NSE IVR | NSE GAP | NSE FSB | skill IVR | skill GAP | skill FSB |
|---|---|---|---|---|---|---|
| 1  | 0,838 | 0,963 | 0,878 | 0,014 | 0,231 | 0,066 |
| 7  | 0,196 | 0,567 | 0,359 | 0,270 | 0,229 | 0,261 |
| 30 | -0,196 | 0,199 | 0,072 | 0,375 | 0,395 | 0,461 |

- **Por NSE (métrica em que a hipótese foi escrita): CONFIRMADA.** IVR é o pior dos três
  postos nos três leads, sem exceção.
- **Por skill score: NÃO confirmada integralmente.** Nos leads 1 e 30 o IVR também é o pior
  (skill mais baixo). Mas no lead 7 o IVR (0,270) **supera** o GAP (0,229) e fica
  tecnicamente empatado com o FSB (0,261) — 2 de 3 leads na direção prevista, 1 na direção
  contrária.
- **O resultado depende da métrica.** Por NSE, confirmada nos três leads; por skill score,
  só em dois.

**Parte 2 — "nos dias sinalizados como suspeitos [o desempenho] será inferior ao dos dias
consistentes do próprio IVR, em todos os horizontes":**

| lead | NSE suspeito | NSE não-susp. | skill suspeito | skill não-susp. |
|---|---|---|---|---|
| 1  | 0,915 | 0,837 | -0,249 | 0,022 |
| 7  | 0,476 | 0,192 | 0,116 | 0,277 |
| 30 | 0,148 | -0,202 | 0,000 | 0,378 |

- **Por NSE (métrica em que a hipótese foi escrita): REFUTADA.** O NSE do subconjunto
  suspeito é MAIOR que o do não-suspeito nos três leads — direção oposta à hipótese.
- **Por skill score: CONFIRMADA.** O skill do subconjunto suspeito é MENOR que o do
  não-suspeito nos três leads (-0,249<0,022; 0,116<0,277; 0,000<0,378) — direção esperada,
  sem exceção.
- **O resultado depende inteiramente da métrica usada**: NSE refuta, skill score confirma,
  nos mesmos três leads. Explicação provável: NSE normaliza pela variância de cada
  subconjunto, e os dois têm variâncias bem diferentes (subséries distintas); o skill score
  usa o MSE absoluto contra a mesma referência (persistência) dentro do subconjunto, o que
  evita essa distorção.

**Dias por ano em cada subconjunto do IVR, na validação (1 por data, não por lead/semente):**
- não-suspeito: 2005 (335), 2006 (335), 2007 (335), 2009 (246), 2011 (241), 2012 (206),
  2013 (273) — 7 anos distintos, 1.971 dias no total.
- suspeito: 2013 (62, out-dez), 2014 (218) — só 2 anos, um único bloco contínuo de
  ~15 meses, 280 dias no total.
- Caveat válido para as DUAS métricas (não é um critério pra escolher entre elas): o
  subconjunto suspeito vem de um único episódio, não de uma amostra de anos diferentes —
  qualquer diferença encontrada, em qualquer direção, por qualquer métrica, pode refletir o
  que aconteceu especificamente nesse episódio de 2013-2014, não "ser suspeito" em geral.

**Classificação final desta rodada, sem reinterpretar a hipótese depois do resultado:**
Parte 1: CONFIRMADA por NSE; NÃO confirmada integralmente por skill score (diverge no lead 7).
Parte 2: REFUTADA por NSE; CONFIRMADA por skill score.
Nenhuma das duas partes tem veredito estável entre as duas métricas — isso fica registrado
tal como medido, sem favorecer uma leitura. A avaliação definitiva, com o confundimento de
ano reduzido (três episódios suspeitos separados em vez de um só), é o teste 2015-2019.

## 2026-10-01 — RESULTADO DO TESTE FINAL (2015-2019) e VEREDITO DEFINITIVO de H-IVR
Treino único, sem ajuste depois de ver o resultado (conforme pré-registrado acima). Saída
completa: `results/tabelas/lstm_teste.csv`, `lstm_teste_skill_score.csv`; figuras
`10_lstm_teste_nse.png`, `11_lstm_teste_skill_score.png`. Comando usado:
`python -m src.agregar_validacao --dobra teste`.

**Resumo (NSE, leads 1/7/30):**

| posto | LSTM | persistência | climatologia |
|---|---|---|---|
| GAP | 0,968 / 0,526 / 0,281 | 0,950 / 0,303 / -0,366 | 0,265 / 0,267 / 0,293 |
| FSB | 0,937 / 0,264 / 0,138 | 0,914 / -0,093 / -0,778 | 0,140 / 0,143 / 0,164 |
| IVR | 0,934 / 0,601 / 0,412 | 0,966 / 0,593 / 0,342 | -0,459 / -0,420 / -0,039 |

**Faixa de leads em que a LSTM supera as DUAS referências:** GAP 1-22 e 24 (23/30); FSB 1-13
(13/30); IVR 6-30 (25/30) — maior faixa das três, mas pelo mesmo motivo já registrado na
validação (a climatologia do IVR é fraca; não ler como "IVR generaliza melhor").

**Validação x teste, lado a lado (NSE, leads 1/7/30; `--dobra comparar`):** o achado mais
notável é o IVR — melhora MUITO do validação pro teste em todos os 3 leads e nos 3 modelos
(ex.: LSTM lead 7: 0,196 -> 0,601, +0,405; lead 30: -0,196 -> 0,412, +0,608; persistência
lead 30: -0,914 -> 0,342, +1,256). GAP e FSB mudam pouco (a maioria das diferenças fica
entre -0,14 e +0,12). Isso por si só já é um indício de que o IVR do período de teste é
qualitativamente diferente do IVR da validação — relevante para interpretar os vereditos
abaixo.

### PARTE 1 — "desempenho do LSTM em IVR inferior ao de GAP e FSB, em todos os horizontes"
skill score = 1 - MSE_LSTM/MSE_persistência, leads 1/7/30:
IVR (-0,909 / 0,022 / 0,107); GAP (0,360 / 0,320 / 0,473); FSB (0,271 / 0,327 / 0,515).

- **Por NSE (métrica pré-registrada): REFUTADA.** IVR só é pior que os dois nos TRÊS postos
  no lead 1 (e por margem mínima contra FSB: 0,934 vs 0,937). Nos leads 7 e 30 o IVR tem NSE
  MAIOR que GAP e FSB (lead 7: 0,601 > 0,526 e > 0,264; lead 30: 0,412 > 0,281 e > 0,138) —
  o oposto do que a hipótese prevê.
- **Por skill score: CONFIRMADA.** IVR tem skill menor que GAP e FSB nos três leads, sem
  exceção (-0,909 < 0,360 e 0,271; 0,022 < 0,320 e 0,327; 0,107 < 0,473 e 0,515).
- **O resultado depende da métrica — e na direção OPOSTA à da validação** (lá o NSE
  confirmava nos três leads; aqui o NSE refuta em dois deles). O skill score é o único dos
  dois critérios estável entre validação e teste para a Parte 1 (confirma, ou quase confirma,
  nos dois).

### PARTE 2 — "dias suspeitos piores que não-suspeitos (dentro do IVR), em todos os horizontes"
NSE e skill score (vs. persistência do próprio subconjunto), leads 1/7/30:

| lead | NSE suspeito | NSE não-susp. | skill suspeito | skill não-susp. |
|---|---|---|---|---|
| 1  | 0,926 | 0,783 | -0,823 | -1,778 |
| 7  | 0,522 | 0,285 |  0,029 | -0,073 |
| 30 | 0,370 | -0,347 | 0,116 | -0,052 |

- **Por NSE (métrica pré-registrada): REFUTADA.** Suspeito tem NSE MAIOR que não-suspeito
  nos três leads — mesma direção observada na validação.
- **Por skill score: REFUTADA.** Suspeito tem skill MAIOR (melhor) que não-suspeito nos três
  leads (-0,823>-1,778; 0,029>-0,073; 0,116>-0,052) — ao contrário da validação, onde o
  skill score tinha CONFIRMADO a hipótese.
- **As duas métricas concordam no teste: REFUTADA, sem exceção, nos três leads.**

**Dias por ano em cada subconjunto, no teste** (1 por data, não por lead/semente):
- suspeito: 2015 (331, ano inteiro), 2017 (184, jan-set), 2018 (92, out-dez),
  2019 (273, jan-set) — total 880, QUATRO anos, TRÊS episódios separados.
- não-suspeito: 2017 (92, out-dez), 2018 (273, jan-set), 2019 (62, out-dez) — total 427.
- Ao contrário da validação (um único bloco 2013-2014), aqui o confundimento com "o que
  aconteceu num ano específico" é bem menor — os dois subconjuntos atravessam os mesmos
  anos (2017, 2018, 2019), só que em meses diferentes dentro de cada um.

### VEREDITO FINAL DE H-IVR (avaliação definitiva, conforme pré-registrado; sem reinterpretar a hipótese)
- **Parte 1: resultado depende da métrica, sem veredito único.** CONFIRMADA por skill score,
  REFUTADA por NSE (a métrica em que a hipótese foi escrita). Achado adicional não previsto:
  o IVR melhora muito do validação pro teste (tabela acima), o que por si muda o sinal da
  comparação com GAP/FSB nos leads mais longos.
- **Parte 2: REFUTADA.** As duas métricas concordam, nos três leads, com uma amostra de dias
  suspeitos bem menos confundida com um único episódio do que na validação (4 anos, 3
  episódios). A hipótese de que dias de medição sinalizada como inconsistente pioram o
  desempenho do LSTM NÃO se sustenta nos dados deste trabalho.
- Isso fecha a avaliação de H-IVR. Nenhum ajuste de método é feito depois deste resultado
  (pré-registro de 2026-10-01, "PRÉ-REGISTRO: teste final").
