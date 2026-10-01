#!/usr/bin/env bash
# Reproduz o pipeline do projeto, na ordem certa. Ver README.md para o que cada etapa faz.
#
# Pré-requisito (sempre, antes de rodar este script): dados brutos obtidos manualmente —
# data/raw/ e data/interim/ não são versionados (ver "Como reproduzir" no README).
#
# --sem-treino (padrão): ingestão, preparação, QA, EDA, baselines e agregação/figuras finais
#   a partir das previsões da LSTM JÁ SALVAS em results/previsoes/ (versionadas). Não treina
#   nada. Minutos, roda inteiro em .venv.
# --completo: tudo acima, mais TODOS os treinos — reprodução de Zampieri (A/B/C x 3 sementes)
#   e a LSTM multi-horizonte (10 dobras de validação + teste final, x 3 sementes cada).
#   ATENÇÃO: demorado. Estimativa ~4h no total com GPU (a partir dos tempos por treino
#   registrados em docs/decisoes.md). Precisa de .venv-ml com GPU configurada.
#
# Uso:
#   scripts/reproduzir.sh                 # equivale a --sem-treino
#   scripts/reproduzir.sh --sem-treino
#   scripts/reproduzir.sh --completo

set -euo pipefail
cd "$(dirname "$0")/.."

MODO="${1:---sem-treino}"
if [[ "$MODO" != "--sem-treino" && "$MODO" != "--completo" ]]; then
    echo "uso: $0 [--sem-treino|--completo]" >&2
    exit 1
fi

echo "== ingestão (precisa de data/raw/, obtido manualmente — ver README) =="
.venv/bin/python src/ingest/ana_vazoes_parser.py
.venv/bin/python src/ingest/zampieri_loader.py
# xavier_recorte.py não entra aqui: recebe o caminho dos NetCDF brutos como argumento
# manual (download separado, ver README) e já deixa pronto em data/interim/meteo/.

echo "== preparação (tabela diária por posto) =="
.venv/bin/python src/prep/montar_base.py

echo "== QA (consistência chuva-vazão) =="
.venv/bin/python -m src.qa.consistencia_chuva
.venv/bin/python -m src.qa.dupla_massa

echo "== EDA (figuras 01-04 + resumo_postos.csv) =="
.venv/bin/python src/eda/figuras_eda.py

echo "== baselines (persistência e climatologia na validação) =="
.venv/bin/python -m src.rodar_baselines

if [[ "$MODO" == "--completo" ]]; then
    echo
    echo "== --completo: a partir daqui ENTRAM OS TREINOS. Isso demora de verdade =="
    echo "   (reprodução de Zampieri: minutos. LSTM multi-horizonte: ~4h com GPU.)"
    source .venv-ml/bin/activate

    echo "-- reprodução de Zampieri (A/B/C, 3 sementes) --"
    python -m src.repro.zampieri
    python src/eda/figuras_repro.py

    echo "-- LSTM multi-horizonte: validação walk-forward (10 dobras x 3 sementes) --"
    echo "   dobras puladas (amostras insuficientes) são registradas e seguem sem parar o script."
    for posto in ivr gap fsb; do
        for dobra in val2005 val2006 val2007 val2008 val2009 val2010 val2011 val2012 val2013 val2014; do
            for semente in 42 43 44; do
                python -m src.treinar_lstm --posto "$posto" --dobra "$dobra" --semente "$semente" || true
            done
        done
    done

    echo "-- LSTM: teste final (2015-2019), 3 sementes — roda uma única vez (ver docs/decisoes.md) --"
    for posto in ivr gap fsb; do
        for semente in 42 43 44; do
            python -m src.treinar_lstm --posto "$posto" --dobra teste --semente "$semente"
        done
    done
fi

echo
echo "== agregação (usa as previsões em results/previsoes/, salvas ou recém-treinadas) =="
.venv/bin/python -m src.agregar_validacao --dobra validacao
.venv/bin/python -m src.agregar_validacao --dobra teste
.venv/bin/python -m src.agregar_validacao --dobra comparar

echo
echo "== tabelas do README (results/tabelas/*.csv -> markdown) =="
.venv/bin/python -m src.relatorio.tabelas_readme

echo
echo "concluído ($MODO)."
