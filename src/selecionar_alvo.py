"""PRÉ-REGISTRO: escolhe entre alvo "nivel" e "delta" ANTES da validação completa
(ver docs/decisoes.md, 2026-10-01). Critério fixado antes de rodar: maior média de NSE da
LSTM nos leads 1, 7 e 30, em GAP val2005, GAP val2014 e FSB val2014, semente 42 (9 valores por
alvo). Essas 3 dobras também entram na validação completa — leve otimismo declarado.

Roda as 6 combinações (3 dobras x 2 alvos) em sequência, reaproveitando
src/treinar_lstm.rodar(); no final imprime uma tabela única e o alvo vencedor.

Uso: python -m src.selecionar_alvo
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import pandas as pd
import tensorflow as tf

from src.treinar_lstm import LEADS_RESUMO, rodar

SEMENTE = 42
COMBINACOES = [("gap", "val2005"), ("gap", "val2014"), ("fsb", "val2014")]
ALVOS = ["nivel", "delta"]


def main():
    print("GPU:", tf.config.list_physical_devices("GPU") or "nenhuma (CPU)")
    tabelas = []
    for posto, dobra in COMBINACOES:
        for alvo in ALVOS:
            print(f"\n=== {posto.upper()} {dobra} alvo={alvo} ===")
            res, epocas, n_tr, n_val = rodar(posto, dobra, SEMENTE, epocas=200, alvo=alvo)
            print(f"{epocas} épocas treinadas ({n_tr} amostras de treino, {n_val} de validação)")

            r = res[(res["grupo"] == "todos") & res["modelo"].isin(["LSTM", "persistencia"]) &
                    res["lead"].isin(LEADS_RESUMO)]
            piv = r.pivot(index="lead", columns="modelo", values="NSE").reset_index()
            piv = piv.rename(columns={"LSTM": "NSE_LSTM", "persistencia": "NSE_persistencia"})
            piv.insert(0, "alvo", alvo)
            piv.insert(0, "dobra", dobra)
            piv.insert(0, "posto", posto.upper())
            tabelas.append(piv)

    tabela = pd.concat(tabelas, ignore_index=True)
    with pd.option_context("display.width", 160):
        print("\n=== tabela final (dobra, alvo, lead, NSE LSTM, NSE persistência) ===")
        print(tabela.round(3).to_string(index=False))

    media = tabela.groupby("alvo")["NSE_LSTM"].mean()
    print(f"\ncritério: média de NSE_LSTM nos leads {LEADS_RESUMO} das 3 dobras, por alvo:")
    print(media.round(3).to_string())
    print(f"\nalvo escolhido: {media.idxmax()}")


if __name__ == "__main__":
    main()
