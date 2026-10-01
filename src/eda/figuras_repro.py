"""Figura previsto x observado da reprodução de Zampieri (versões A e C), média das sementes."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PREV = Path("results/previsoes")
SAIDA = Path("results/figuras/07_repro_previsto_observado.png")


def carregar(versao, posto):
    arqs = sorted(PREV.glob(f"repro_{versao}_{posto}_s*.csv"))
    if not arqs:
        raise FileNotFoundError(f"sem previsões para {versao}/{posto}: rode src/repro/zampieri.py")
    d = pd.concat(pd.read_csv(a, parse_dates=["data"]) for a in arqs)
    return d.groupby("data")[["obs", "lstm", "persistencia"]].mean().sort_index()


def main():
    postos = ["ivr", "gap", "fsb"]
    fig, ax = plt.subplots(len(postos), 2, figsize=(13, 10), gridspec_kw={"width_ratios": [1, 2.2]})
    for i, p in enumerate(postos):
        a = ax[i, 0]
        d = carregar("A", p)
        lim = [d.min().min(), d.max().max()]
        a.scatter(d["obs"], d["lstm"], s=8, alpha=0.6, label="LSTM")
        a.scatter(d["obs"], d["persistencia"], s=8, alpha=0.6, label="persistência")
        a.plot(lim, lim, "k--", lw=0.8)
        a.set(xlabel="observado (cm)", ylabel="previsto (cm)", title=f"{p.upper()} — versão A (divisão aleatória)")

        a = ax[i, 1]
        d = carregar("C", p)
        a.plot(d.index, d["obs"], "k", lw=1.4, label="observado")
        a.plot(d.index, d["lstm"], lw=1.2, label="LSTM (média de 3 sementes)")
        a.plot(d.index, d["persistencia"], "--", lw=1, label="persistência")
        a.set(ylabel="cota (cm)", title=f"{p.upper()} — versão C (cronológica + cota)")
        if i == 0:
            ax[i, 0].legend(fontsize=8)
            a.legend(fontsize=8)
    fig.tight_layout()
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA, dpi=150)
    print("salvo", SAIDA)


if __name__ == "__main__":
    main()
