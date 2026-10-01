"""Treina o LSTM multi-horizonte numa única dobra da validação walk-forward
(um modelo por posto/dobra/semente — ver docs/decisoes.md, 2026-10-01).

Decisões fixadas a priori: vazão em log1p+MinMax (o fit treina e monitora o early stopping
NESSA escala; todas as métricas abaixo são calculadas depois de desfazer para m³/s), chuva/
tmax/tmin só MinMax, janela N=60, saída Dense(30) com os 30 leads de uma vez, parada
antecipada nas últimas N_DEV amostras VÁLIDAS do treino da dobra, em ordem cronológica — pode
atravessar uma lacuna, já que cada amostra individual já respeita o segmento via
indices_validos (ver docs/decisoes.md, 2026-10-01: o "último ano calendário" original colidia
com os períodos removidos em configs/periodos_invalidos.yaml). O scaler é ajustado só com os
dias anteriores à primeira amostra de dev. Dobras com menos de MIN_VAL_AMOSTRAS amostras de
validação são puladas (ver mesma entrada). IVR mantém os dias "suspeito" no treino e na
avaliação, mas reporta a métrica separada.

Persistência e climatologia (src/models/baselines.py) são calculadas nos MESMOS dias de
validação da dobra, com o mesmo treino (tudo antes de fim_treino) que src/rodar_baselines.py
usa — sem isso não dá pra interpretar se o NSE da LSTM numa dobra isolada é bom ou ruim.

--alvo (ver docs/decisoes.md, 2026-10-01, PRÉ-REGISTRO): "nivel" (padrão) treina o LSTM pra
prever log1p(Q[t+h]) direto (escalado com MinMax). "delta" treina pra prever
log1p(Q[t+h]) - log1p(Q[t]) (escalado à parte, só com o treino); a previsão final soma de volta
log1p(Q[t]) antes de desfazer pra m³/s. Escolha entre os dois é feita UMA vez por
src/selecionar_alvo.py, antes da validação completa — não é um hiperparâmetro de cada rodada.

Uso (teste rápido, poucas épocas, 1 dobra, 1 semente):
    python -m src.treinar_lstm --posto gap --dobra val2014 --semente 42 --epocas 5 --alvo delta

Saídas (por execução): results/previsoes/lstm_<posto>_<dobra>_<alvo>_s<semente>.csv (data, lead,
obs, lstm; todos os 30 leads) e ..._historico.csv (época, loss, val_loss). Nada é agregado em
disco ainda entre dobras/sementes — isso entra quando o código estiver fechado e rodarmos as
10 dobras x 3 sementes de uma vez (ver plano em docs/decisoes.md).
"""
import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.eval import metricas
from src.features import amostras, divisao, escala, janelas
from src.models import baselines, lstm

N = 60  # janela de entrada, fixada a priori (memória medida: ACF log Q 43-50 dias)
N_DEV = 365  # amostras de parada antecipada: as últimas N_DEV válidas do treino, cronológicas
MIN_VAL_AMOSTRAS = 60  # dobra com menos que isso de validação é pulada (ver docs/decisoes.md)
LEADS_RESUMO = [1, 7, 30]
PROC = Path("data/processed")
PREV = Path("results/previsoes")


class DobraPulada(Exception):
    """Dobra sem amostras de validação suficientes (colide com período removido)."""


def preparar_dobra(df: pd.DataFrame, dobra: str):
    _, fim_treino, ini_val, fim_val = next(d for d in divisao.dobras() if d[0] == dobra)

    t_idx = janelas.indices_validos(df["segmento"], amostras.N_MAX_BASELINE, amostras.H_MAX_BASELINE)
    datas_todas = df["data"].iloc[t_idx].to_numpy()
    suspeito_todas = df["suspeito"].to_numpy()[t_idx]
    q_t_todas = df["vazao"].to_numpy()[t_idx]  # vazão no dia de emissão, p/ a persistência

    m_val = divisao.mascara_periodo(datas_todas, amostras.H_MAX_BASELINE, ini_val, fim_val)
    if m_val.sum() < MIN_VAL_AMOSTRAS:
        raise DobraPulada(f"{dobra}: só {m_val.sum()} amostras de validação "
                          f"(< {MIN_VAL_AMOSTRAS}); colide com período removido em "
                          f"configs/periodos_invalidos.yaml")

    # dev = as últimas N_DEV amostras VÁLIDAS do treino (cronológicas; t_idx já é crescente),
    # mesmo que atravessem uma lacuna — cada amostra em si já respeita o segmento.
    idx_treino = np.where(datas_todas < fim_treino.to_datetime64())[0]
    n_dev = min(N_DEV, len(idx_treino))
    idx_dev, idx_fit = idx_treino[-n_dev:], idx_treino[:-n_dev]
    dev_inicio_data = datas_todas[idx_dev[0]]

    # scaler ajustado só com os dias anteriores à primeira amostra de dev
    esc = escala.ajustar(df[df["data"] < dev_inicio_data])
    X, Y, datas = amostras.preparar(df, esc, N, t_idx)

    # baselines de referência: MESMO treino (tudo antes de fim_treino) e MESMOS dias de
    # validação que src/rodar_baselines.py usa, pra comparação ser direta.
    treino_bl = df[df["data"] < fim_treino]
    clim = baselines.ajustar_climatologia(treino_bl["data"], treino_bl["vazao"])
    referencia = {
        "persistencia": baselines.persistencia(q_t_todas[m_val], amostras.H_MAX_BASELINE),
        "climatologia": baselines.climatologia(clim, datas[m_val], amostras.H_MAX_BASELINE),
    }
    val = (X[m_val], Y[m_val], suspeito_todas[m_val], datas[m_val], q_t_todas[m_val])
    return esc, (X[idx_fit], Y[idx_fit], q_t_todas[idx_fit]), (X[idx_dev], Y[idx_dev], q_t_todas[idx_dev]), \
        val, referencia


def _alvo_fit(alvo: str, esc: escala.Escala, Y, q_t):
    """Ajusta o alvo de TREINO: nível (log1p+MinMax de Q[t+h], escala de `esc`) ou delta
    (log1p(Q[t+h]) - log1p(Q[t]), MinMax ajustado agora, só com este conjunto). Devolve o array
    escalado e o scaler do delta (None em modo nível) — usado depois em `_alvo_aplicar`."""
    if alvo == "nivel":
        return escala.transformar_vazao(esc, Y), None
    delta = np.log1p(Y) - np.log1p(q_t)[:, None]
    sc_delta = escala.ajustar_serie(delta)
    return escala.transformar_serie(sc_delta, delta), sc_delta


def _alvo_aplicar(alvo: str, esc: escala.Escala, sc_delta, Y, q_t):
    """Aplica a MESMA escala do treino (nunca reajusta) ao alvo de outro conjunto (dev)."""
    if alvo == "nivel":
        return escala.transformar_vazao(esc, Y)
    delta = np.log1p(Y) - np.log1p(q_t)[:, None]
    return escala.transformar_serie(sc_delta, delta)


def _desfazer_pred(alvo: str, esc: escala.Escala, sc_delta, pred_escalado, q_t):
    if alvo == "nivel":
        return escala.desfazer_vazao(esc, pred_escalado)
    delta_pred = escala.desfazer_serie(sc_delta, pred_escalado)
    return np.expm1(np.log1p(q_t)[:, None] + delta_pred)


def rodar(posto: str, dobra: str, semente: int, epocas: int, alvo: str = "nivel"):
    tf.keras.utils.set_random_seed(semente)
    df = pd.read_parquet(PROC / f"{posto}.parquet")
    esc, (X_tr, Y_tr, q_tr), (X_dev, Y_dev, q_dev), \
        (X_val, Y_val, suspeito_val, datas_val, q_val), ref = preparar_dobra(df, dobra)

    y_tr, sc_delta = _alvo_fit(alvo, esc, Y_tr, q_tr)
    y_dev = _alvo_aplicar(alvo, esc, sc_delta, Y_dev, q_dev)

    modelo = lstm.construir(n_entradas=X_tr.shape[-1], n=N, h=amostras.H_MAX_BASELINE)
    parar = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    hist = modelo.fit(X_tr, y_tr, validation_data=(X_dev, y_dev), epochs=epocas, batch_size=32,
                      shuffle=True, verbose=2, callbacks=[parar])

    pred = _desfazer_pred(alvo, esc, sc_delta, modelo.predict(X_val, verbose=0), q_val)

    linhas = []
    for k in LEADS_RESUMO:
        linhas.append({"lead": k, "grupo": "todos", "modelo": "LSTM",
                       **metricas.todas(Y_val[:, k - 1], pred[:, k - 1])})
        for nome, prev in ref.items():
            linhas.append({"lead": k, "grupo": "todos", "modelo": nome,
                           **metricas.todas(Y_val[:, k - 1], prev[:, k - 1])})
    if posto == "ivr":
        for k in LEADS_RESUMO:
            for grupo, m in [("suspeito", suspeito_val), ("nao_suspeito", ~suspeito_val)]:
                if m.any():
                    linhas.append({"lead": k, "grupo": grupo, "modelo": "LSTM",
                                   **metricas.todas(Y_val[m, k - 1], pred[m, k - 1])})

    H = amostras.H_MAX_BASELINE
    prev_df = pd.DataFrame({"data": np.repeat(datas_val, H), "lead": np.tile(np.arange(1, H + 1), len(datas_val)),
                            "obs": Y_val.ravel(), "lstm": pred.ravel()})
    hist_df = pd.DataFrame({"epoca": range(1, len(hist.history["loss"]) + 1),
                            "loss": hist.history["loss"], "val_loss": hist.history["val_loss"]})
    PREV.mkdir(parents=True, exist_ok=True)
    prev_df.to_csv(PREV / f"lstm_{posto}_{dobra}_{alvo}_s{semente}.csv", index=False)
    hist_df.to_csv(PREV / f"lstm_{posto}_{dobra}_{alvo}_s{semente}_historico.csv", index=False)

    return pd.DataFrame(linhas), len(hist.history["loss"]), len(X_tr), len(X_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posto", required=True, choices=["ivr", "gap", "fsb"])
    ap.add_argument("--dobra", required=True, help="ex.: val2014 (ver src/features/divisao.py)")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--epocas", type=int, default=200)
    ap.add_argument("--alvo", choices=["nivel", "delta"], default="nivel",
                    help="nivel = prevê log1p(Q); delta = prevê log1p(Q[t+h])-log1p(Q[t]) "
                         "(ver docs/decisoes.md, 2026-10-01, PRÉ-REGISTRO)")
    args = ap.parse_args()

    print("GPU:", tf.config.list_physical_devices("GPU") or "nenhuma (CPU)")
    try:
        res, epocas_usadas, n_tr, n_val = rodar(args.posto, args.dobra, args.semente, args.epocas, args.alvo)
    except DobraPulada as e:
        print(f"\n[pulada] {args.posto.upper()} {e}")
        return
    print(f"\n{args.posto.upper()} {args.dobra} alvo={args.alvo} semente {args.semente}: "
          f"{n_tr} amostras de treino, {n_val} de validação, {epocas_usadas} épocas treinadas")
    with pd.option_context("display.width", 160):
        print(res.round(3).to_string(index=False))
    print(f"\nprevisões em {PREV}/lstm_{args.posto}_{args.dobra}_{args.alvo}_s{args.semente}.csv e "
          f"..._historico.csv")


if __name__ == "__main__":
    main()
