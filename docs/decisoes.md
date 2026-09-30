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
