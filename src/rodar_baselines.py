"""Fase 3: baselines (persistência e climatologia) na validação walk-forward 2005-2014.

O teste (2015-2019) NÃO é tocado aqui.
Amostras: mesmas que o LSTM usará (janela máx. N=90, horizonte máx. H=30),
para que a comparação futura seja feita exatamente sobre os mesmos dias.

Uso: python -m src.rodar_baselines
Saídas: results/tabelas/baselines_validacao.csv e results/figuras/05_baselines_nse.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.eval import metricas
from src.features import divisao, janelas
from src.models import baselines

N_MAX, H_MAX = 90, 30
LEADS_RESUMO = [1, 7, 30]
POSTOS = yaml.safe_load(Path("configs/postos.yaml").read_text(encoding="utf-8"))
ROTULO = {"persistencia": "persistência", "climatologia": "climatologia"}
PROC, TAB, FIG = Path("data/processed"), Path("results/tabelas"), Path("results/figuras")


def avaliar_posto(p: str) -> pd.DataFrame:
    df = pd.read_parquet(PROC / f"{p}.parquet")
    t_idx = janelas.indices_validos(df["segmento"], N_MAX, H_MAX)
    _, Y, datas = janelas.montar(df, ["vazao"], "vazao", N_MAX, H_MAX, t_idx)
    q_t = df["vazao"].to_numpy()[t_idx]

    obs, pred = [], {"persistencia": [], "climatologia": []}
    for nome, fim_treino, ini_val, fim_val in divisao.dobras():
        m_val = divisao.mascara_periodo(datas, H_MAX, ini_val, fim_val)
        if not m_val.any():
            continue
        treino = df[df["data"] < fim_treino]
        clim = baselines.ajustar_climatologia(treino["data"], treino["vazao"])
        obs.append(Y[m_val])
        pred["persistencia"].append(baselines.persistencia(q_t[m_val], H_MAX))
        pred["climatologia"].append(baselines.climatologia(clim, datas[m_val], H_MAX))

    obs = np.concatenate(obs)
    linhas = []
    for modelo, ps in pred.items():
        ps = np.concatenate(ps)
        for k in range(1, H_MAX + 1):
            linhas.append({"posto": p.upper(), "modelo": modelo, "lead": k, "n_amostras": len(obs),
                           **metricas.todas(obs[:, k - 1], ps[:, k - 1])})
    return pd.DataFrame(linhas)


def figura(res: pd.DataFrame):
    postos = list(res["posto"].unique())
    fig, axs = plt.subplots(1, len(postos), figsize=(9, 3.2), sharey=True)
    estilo = {"persistencia": dict(color="#52514e", linestyle="-"),
              "climatologia": dict(color="#a3a29d", linestyle="--")}
    for ax, p in zip(np.atleast_1d(axs), postos):
        for modelo, st in estilo.items():
            r = res[(res["posto"] == p) & (res["modelo"] == modelo)]
            ax.plot(r["lead"], r["NSE"], linewidth=2, label=ROTULO[modelo], **st)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(p, loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("antecedência (dias)")
        ax.grid(color="#e6e5e1", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel("NSE (validação 2005–2014)")
    np.atleast_1d(axs)[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "05_baselines_nse.png", dpi=200)
    plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    res = pd.concat([avaliar_posto(p) for p in POSTOS], ignore_index=True)
    res.to_csv(TAB / "baselines_validacao.csv", index=False)
    figura(res)
    resumo = res[res["lead"].isin(LEADS_RESUMO)].copy()
    resumo[["NSE", "KGE", "R2"]] = resumo[["NSE", "KGE", "R2"]].round(3)
    resumo[["PBIAS", "RMSE"]] = resumo[["PBIAS", "RMSE"]].round(1)
    with pd.option_context("display.width", 200):
        print(resumo.drop(columns="n_amostras").to_string(index=False))
    print("\namostras de validação por posto:", res.groupby("posto")["n_amostras"].first().to_dict())


if __name__ == "__main__":
    main()
