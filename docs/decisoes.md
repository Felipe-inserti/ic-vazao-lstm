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
