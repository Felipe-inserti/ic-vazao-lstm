"""Treina o LSTM multi-horizonte numa única dobra da validação walk-forward
(um modelo por posto/dobra/semente — ver docs/decisoes.md, 2026-10-01).

Decisões fixadas a priori: vazão em log1p+MinMax (o fit treina e monitora o early stopping
NESSA escala; todas as métricas abaixo são calculadas depois de desfazer para m³/s), chuva/
tmax/tmin só MinMax (ajustados no treino da dobra, sem o último ano), janela N=60, saída
Dense(30) com os 30 leads de uma vez, parada antecipada no último ano do treino da dobra
(cronológico, sem embaralhar segmentos). IVR mantém os dias "suspeito" no treino e na
avaliação, mas reporta a métrica separada.

Persistência e climatologia (src/models/baselines.py) são calculadas nos MESMOS dias de
validação da dobra, com o mesmo treino (tudo antes de fim_treino) que src/rodar_baselines.py
usa — sem isso não dá pra interpretar se o NSE da LSTM numa dobra isolada é bom ou ruim.

Uso (teste rápido, poucas épocas, 1 dobra, 1 semente):
    python -m src.treinar_lstm --posto gap --dobra val2014 --semente 42 --epocas 5

Saídas (por execução): results/previsoes/lstm_<posto>_<dobra>_s<semente>.csv (data, lead, obs,
lstm; todos os 30 leads) e ..._historico.csv (época, loss, val_loss). Nada é agregado em disco
ainda entre dobras/sementes — isso entra quando o código estiver fechado e rodarmos as
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
LEADS_RESUMO = [1, 7, 30]
PROC = Path("data/processed")
PREV = Path("results/previsoes")


def preparar_dobra(df: pd.DataFrame, dobra: str):
    _, fim_treino, ini_val, fim_val = next(d for d in divisao.dobras() if d[0] == dobra)
    dev_inicio = fim_treino - pd.DateOffset(years=1)

    # scaler ajustado só no treino da dobra, excluindo o último ano (parada antecipada)
    esc = escala.ajustar(df[df["data"] < dev_inicio])

    t_idx = janelas.indices_validos(df["segmento"], amostras.N_MAX_BASELINE, amostras.H_MAX_BASELINE)
    X, Y, datas = amostras.preparar(df, esc, N, t_idx)
    suspeito = df["suspeito"].to_numpy()[t_idx]
    q_t = df["vazao"].to_numpy()[t_idx]  # vazão no dia de emissão, p/ a persistência

    m_treino = datas < dev_inicio.to_datetime64()
    m_dev = (datas >= dev_inicio.to_datetime64()) & (datas < fim_treino.to_datetime64())
    m_val = divisao.mascara_periodo(datas, amostras.H_MAX_BASELINE, ini_val, fim_val)

    # baselines de referência: MESMO treino (tudo antes de fim_treino) e MESMOS dias de
    # validação que src/rodar_baselines.py usa, pra comparação ser direta.
    treino_bl = df[df["data"] < fim_treino]
    clim = baselines.ajustar_climatologia(treino_bl["data"], treino_bl["vazao"])
    referencia = {
        "persistencia": baselines.persistencia(q_t[m_val], amostras.H_MAX_BASELINE),
        "climatologia": baselines.climatologia(clim, datas[m_val], amostras.H_MAX_BASELINE),
    }
    val = (X[m_val], Y[m_val], suspeito[m_val], datas[m_val])
    return esc, (X[m_treino], Y[m_treino]), (X[m_dev], Y[m_dev]), val, referencia


def rodar(posto: str, dobra: str, semente: int, epocas: int):
    tf.keras.utils.set_random_seed(semente)
    df = pd.read_parquet(PROC / f"{posto}.parquet")
    esc, (X_tr, Y_tr), (X_dev, Y_dev), (X_val, Y_val, suspeito_val, datas_val), ref = \
        preparar_dobra(df, dobra)

    y_tr = escala.transformar_vazao(esc, Y_tr)
    y_dev = escala.transformar_vazao(esc, Y_dev)

    modelo = lstm.construir(n_entradas=X_tr.shape[-1], n=N, h=amostras.H_MAX_BASELINE)
    parar = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    hist = modelo.fit(X_tr, y_tr, validation_data=(X_dev, y_dev), epochs=epocas, batch_size=32,
                      shuffle=True, verbose=2, callbacks=[parar])

    pred = escala.desfazer_vazao(esc, modelo.predict(X_val, verbose=0))

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
    prev_df.to_csv(PREV / f"lstm_{posto}_{dobra}_s{semente}.csv", index=False)
    hist_df.to_csv(PREV / f"lstm_{posto}_{dobra}_s{semente}_historico.csv", index=False)

    return pd.DataFrame(linhas), len(hist.history["loss"]), len(X_tr), len(X_val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posto", required=True, choices=["ivr", "gap", "fsb"])
    ap.add_argument("--dobra", required=True, help="ex.: val2014 (ver src/features/divisao.py)")
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--epocas", type=int, default=200)
    args = ap.parse_args()

    print("GPU:", tf.config.list_physical_devices("GPU") or "nenhuma (CPU)")
    res, epocas_usadas, n_tr, n_val = rodar(args.posto, args.dobra, args.semente, args.epocas)
    print(f"\n{args.posto.upper()} {args.dobra} semente {args.semente}: {n_tr} amostras de treino, "
          f"{n_val} de validação, {epocas_usadas} épocas treinadas")
    with pd.option_context("display.width", 160):
        print(res.round(3).to_string(index=False))
    print(f"\nprevisões em {PREV}/lstm_{args.posto}_{args.dobra}_s{args.semente}.csv e "
          f"..._historico.csv")


if __name__ == "__main__":
    main()
