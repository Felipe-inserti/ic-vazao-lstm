"""Reprodução de Zampieri (2025): LSTM "somente dados", horizonte de 1 dia.

Dados: CSVs do repositório do autor (2022-2024), já carregados em data/interim/zampieri/.
ATENÇÃO: a coluna "Vazao Observada" é, muito provavelmente, COTA (cm) — ver docs/decisoes.md.
As métricas abaixo estão na unidade da série (cm), não em m³/s.

Três versões, mesmo modelo (17 variáveis do INMET, janela de 10 dias):
  A (fiel)        : exatamente como o código original:
                    - MinMaxScaler ajustado em TODO o conjunto antes da divisão;
                    - janelas por LINHA (atravessam os dias ausentes);
                    - divisão ALEATÓRIA (train_test_split): 80% treino, 10% validação, 10% teste;
                    - data augmentation (2 cópias com ruído gaussiano 0,01, inclusive no alvo).
  B (cronológica) : as mesmas escolhas de modelo, mas sem vazamento:
                    - divisão por TEMPO: primeiros 80% treino, 10% validação, 10% finais teste;
                    - scaler ajustado SÓ no treino;
                    - janelas não atravessam dias ausentes.
  C (B + cota)    : igual à B, mas a janela de entrada inclui também a cota dos 10 dias
                    anteriores (dias i-10 a i-1; o dia previsto nunca entra, sem vazamento).
Modelo (código original): LSTM(64)x3 tanh, Dense(1), Adam 1e-3, MSE, lote 32.
Treino: até 200 épocas com parada antecipada na validação (paciência 20, restaura os
melhores pesos) e embaralhamento dos lotes.
Cada versão roda com 3 sementes; reporta-se média ± desvio.
Referência trivial: persistência (valor de ontem) nos MESMOS dias de teste.

Uso: python -m src.repro.zampieri            (tudo)
     python -m src.repro.zampieri --rapido   (2 épocas, 1 semente: só para testar o código)
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.eval import metricas

FEATURES = ["Precipitacao", "Pressao Atmosferica Estacao", "Pressao Atmosferica Max",
            "Pressao Atmosferica Min", "Radiacao Global", "Temperatura Bulbo",
            "Temperatura Ponto Orvalho", "Temperatura Max", "Temperatura Min",
            "Temperatura Orvalho Max", "Temperatura Orvalho Min", "Umidade Max Hora Ant",
            "Umidade Min Hora Ant", "Umidade Relativa Ar", "Vento Direcao Horaria",
            "Vento Rajada", "Vento Velocidade Horaria"]
ALVO = "Vazao Observada"
SEQ = 10
POSTOS = ["ivr", "gap", "fsb"]
VERSOES = ["A", "B", "C"]
DADOS, TAB = Path("data/interim/zampieri"), Path("results/tabelas")


def modelo(n_feat: int) -> tf.keras.Model:
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(SEQ, n_feat)),
        tf.keras.layers.LSTM(64, activation="tanh", return_sequences=True),
        tf.keras.layers.LSTM(64, activation="tanh", return_sequences=True),
        tf.keras.layers.LSTM(64, activation="tanh"),
        tf.keras.layers.Dense(1),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m


def aumentar(X, y, rng, ruido=0.01):
    """Augmentation do código original: 2 cópias com ruído gaussiano (X e y)."""
    Xs = [X] + [X + ruido * rng.normal(size=X.shape) for _ in range(2)]
    ys = [y] + [y + ruido * rng.normal(size=y.shape) for _ in range(2)]
    return np.concatenate(Xs).astype("float32"), np.concatenate(ys).astype("float32")


def janelas(Xs, ys, validos):
    """X = entradas dos SEQ dias anteriores; y = alvo no dia i (como no original)."""
    idx = [i for i in range(SEQ, len(Xs)) if validos[i]]
    X = np.stack([Xs[i - SEQ:i] for i in idx]).astype("float32")
    y = ys[idx].astype("float32")
    return X, y, np.array(idx)


def preparar_A(df, semente):
    sc = MinMaxScaler().fit(df[FEATURES + [ALVO]])                 # conjunto inteiro (vazamento)
    Z = sc.transform(df[FEATURES + [ALVO]])
    X, y, idx = janelas(Z[:, :-1], Z[:, -1], np.ones(len(df), bool))  # ignora dias ausentes
    X_tr, X_tmp, y_tr, y_tmp, i_tr, i_tmp = train_test_split(X, y, idx, test_size=0.2, random_state=semente)
    X_va, X_te, y_va, y_te, i_va, i_te = train_test_split(X_tmp, y_tmp, i_tmp, test_size=0.5, random_state=semente)
    return sc, (X_tr, y_tr), (X_va, y_va), (X_te, y_te, i_te)


def preparar_B(df, com_cota=False):
    n = len(df)
    n_tr, n_va = int(0.8 * n), int(0.9 * n)
    sc = MinMaxScaler().fit(df[FEATURES + [ALVO]].iloc[:n_tr])     # só treino
    Z = sc.transform(df[FEATURES + [ALVO]])
    # janela válida só se os SEQ+1 dias forem consecutivos no calendário
    dias = df["data"].to_numpy().astype("datetime64[D]").astype(int)
    validos = np.zeros(n, bool)
    validos[SEQ:] = (dias[SEQ:] - dias[:-SEQ]) == SEQ
    entradas = Z if com_cota else Z[:, :-1]                         # C: inclui a cota dos dias i-10..i-1
    X, y, idx = janelas(entradas, Z[:, -1], validos)
    tr, va, te = idx < n_tr, (idx >= n_tr + SEQ) & (idx < n_va), idx >= n_va + SEQ  # sem janelas na fronteira
    return sc, (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te], idx[te])


def desfazer(sc, z):
    return z / sc.scale_[-1] - sc.min_[-1] / sc.scale_[-1]


def rodar(df, versao, semente, epocas):
    tf.keras.utils.set_random_seed(semente)
    rng = np.random.default_rng(semente)
    if versao == "A":
        dados = preparar_A(df, semente)
    else:
        dados = preparar_B(df, com_cota=(versao == "C"))
    sc, (X_tr, y_tr), (X_va, y_va), (X_te, y_te, i_te) = dados
    X_tr, y_tr = aumentar(X_tr, y_tr, rng)
    m = modelo(X_tr.shape[-1])
    parar = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)
    hist = m.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epocas, batch_size=32,
                 shuffle=True, verbose=0, callbacks=[parar])
    epocas_usadas = len(hist.history["loss"])
    obs = desfazer(sc, y_te)
    prev = desfazer(sc, m(X_te, training=False).numpy().ravel())
    ontem = df[ALVO].to_numpy()[i_te - 1]                           # persistência nos mesmos dias
    return (metricas.todas(obs, prev), metricas.todas(obs, ontem), len(y_te),
            len(X_tr) // 3, df["data"].iloc[i_te].min().date(), df["data"].iloc[i_te].max().date(),
            epocas_usadas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    epocas, sementes = (2, [42]) if args.rapido else (200, [42, 43, 44])
    print("GPU:", tf.config.list_physical_devices("GPU") or "nenhuma (CPU)")

    linhas = []
    for p in POSTOS:
        df = pd.read_parquet(DADOS / f"{p}.parquet").sort_values("data").reset_index(drop=True)
        for versao in VERSOES:
            for s in sementes:
                t0 = time.time()
                m_lstm, m_pers, n_te, n_tr, ini, fim, ep = rodar(df, versao, s, epocas)
                comum = {"posto": p.upper(), "versao": versao, "semente": s, "n_treino": n_tr,
                         "n_teste": n_te, "teste_de": ini, "teste_ate": fim, "epocas": ep}
                linhas.append({**comum, "modelo": "LSTM", **m_lstm})
                linhas.append({**comum, "modelo": "persistencia", **m_pers})
                print(f"{p.upper()} versão {versao} semente {s}: NSE LSTM {m_lstm['NSE']:.3f} | "
                      f"persistência {m_pers['NSE']:.3f} | {ep} épocas | {time.time() - t0:.0f} s")

    res = pd.DataFrame(linhas)
    TAB.mkdir(parents=True, exist_ok=True)
    sufixo = "_rapido" if args.rapido else ""
    res.to_csv(TAB / f"repro_zampieri{sufixo}.csv", index=False)

    resumo = (res.groupby(["posto", "versao", "modelo"])[["NSE", "KGE", "PBIAS", "RMSE", "R2"]]
              .agg(["mean", "std"]).round(3))
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\nmédia e desvio entre sementes (métricas na unidade da série, provavelmente cm):")
        print(resumo.to_string())
        print("\nperíodo de teste por versão:")
        print(res.groupby(["posto", "versao"])[["teste_de", "teste_ate", "n_teste", "n_treino"]].first().to_string())
        print("\népocas treinadas (parada antecipada), média entre sementes:")
        print(res[res["modelo"] == "LSTM"].groupby(["posto", "versao"])["epocas"].mean().round(0).to_string())


if __name__ == "__main__":
    main()