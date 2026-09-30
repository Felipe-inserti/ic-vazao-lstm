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
