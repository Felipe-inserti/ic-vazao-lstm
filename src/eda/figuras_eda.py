"""Análise exploratória: figuras e tabela-resumo da seção "Dados" do relatório.

Gera em results/figuras/:
  01_disponibilidade.png   linha do tempo dos segmentos contínuos + período de teste
  02_sazonalidade.png      vazão específica mediana por mês + chuva média mensal
  03_permanencia.png       curva de permanência (vazão específica, escala log)
  04_memoria.png           autocorrelação da vazão + resposta chuva -> aumento da vazão
e results/tabelas/resumo_postos.csv

Uso: python src/eda/figuras_eda.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

POSTOS = yaml.safe_load(Path("configs/postos.yaml").read_text(encoding="utf-8"))
PROC = Path("data/processed")
FIG = Path("results/figuras")
TAB = Path("results/tabelas")
TESTE = ("2015-01-01", "2019-12-31")

# paleta categorical fixa: IVR, GAP, FSB sempre com a mesma cor
COR = {"ivr": "#2a78d6", "gap": "#eb6834", "fsb": "#1baf7a"}
TXT, TXT2, GRADE = "#0b0b0b", "#52514e", "#e6e5e1"
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
    "axes.edgecolor": TXT2, "axes.labelcolor": TXT2, "xtick.color": TXT2, "ytick.color": TXT2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "lines.linewidth": 2, "legend.frameon": False, "axes.titlesize": 10, "axes.titleweight": "bold",
})


def carregar() -> dict[str, pd.DataFrame]:
    dados = {}
    for p, cfg in POSTOS.items():
        df = pd.read_parquet(PROC / f"{p}.parquet").set_index("data")
        df["q_esp"] = df["vazao"] / cfg["area_km2"] * 1000  # L/s·km²
        dados[p] = df
    return dados


def fig_disponibilidade(d):
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.axvspan(pd.Timestamp(TESTE[0]), pd.Timestamp(TESTE[1]), color="#f0efec", zorder=0)
    ax.text(pd.Timestamp("2017-07-01"), -0.6, "teste", ha="center", va="center", color=TXT2, fontsize=8)
    for i, p in enumerate(d):
        seg = d[p].dropna(subset=["vazao"]).groupby("segmento")
        for _, g in seg:
            ax.barh(i, g.index.max() - g.index.min(), left=g.index.min(), height=0.5,
                    color=COR[p], edgecolor="white", linewidth=1)
    ax.set_yticks(range(len(d)), [p.upper() for p in d])
    ax.invert_yaxis()
    ax.set_ylim(2.5, -0.9)
    ax.grid(axis="y", visible=False)
    ax.set_title("Períodos contínuos de vazão (lacunas > 7 dias separam segmentos)", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "01_disponibilidade.png")
    plt.close(fig)


def fig_sazonalidade(d):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.5, 5), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 2]})
    for p, df in d.items():
        m = df.groupby(df.index.month)["q_esp"].median()
        a1.plot(m.index, m.values, color=COR[p], marker="o", markersize=4, label=p.upper())
        a1.annotate(p.upper(), (12, m.iloc[-1]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=TXT, fontsize=8)
    a1.set_ylabel("vazão específica mediana\n(L/s·km²)")
    a1.set_ylim(bottom=0)
    a1.legend(loc="best")
    a1.set_title("Sazonalidade da vazão e da chuva", loc="left")

    chuva = pd.concat([df["chuva"] for df in d.values()], axis=1, sort=True).mean(axis=1)
    mensal = chuva.groupby([chuva.index.year, chuva.index.month]).sum().groupby(level=1).mean()
    a2.bar(mensal.index, mensal.values, color="#86b6ef", width=0.7)
    a2.set_ylabel("chuva média\n(mm/mês)")
    a2.set_xticks(range(1, 13), MESES)
    fig.tight_layout()
    fig.savefig(FIG / "02_sazonalidade.png")
    plt.close(fig)


def fig_permanencia(d):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for p, df in d.items():
        q = np.sort(df["q_esp"].dropna().values)[::-1]
        exc = np.arange(1, len(q) + 1) / (len(q) + 1) * 100
        ax.plot(exc, q, color=COR[p], label=p.upper())
    ax.set_yscale("log")
    ax.set_xlabel("% do tempo em que a vazão é igualada ou superada")
    ax.set_ylabel("vazão específica (L/s·km²)")
    ax.set_title("Curva de permanência", loc="left")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "03_permanencia.png")
    plt.close(fig)


def acf_segmentada(df, lags):
    """Autocorrelação de log(Q) sem cruzar lacunas longas (pares só dentro do mesmo segmento)."""
    x = np.log1p(df["vazao"])
    seg = df["segmento"]
    return [x.corr(x.shift(k).where(seg == seg.shift(k))) for k in lags]


def resposta_chuva(df, lags):
    dq = df["vazao"].diff()
    return [dq.corr(df["chuva"].shift(k), method="spearman") for k in lags]


def fig_memoria(d):
    lags_q, lags_c = range(0, 91), range(0, 11)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [3, 2]})
    memoria = {}
    for p, df in d.items():
        acf = acf_segmentada(df, lags_q)
        a1.plot(list(lags_q), acf, color=COR[p], label=p.upper())
        memoria[p] = next((k for k, v in zip(lags_q, acf) if v < 0.5), None)
        a2.plot(list(lags_c), resposta_chuva(df, lags_c), color=COR[p], marker="o", markersize=4, label=p.upper())
    for k in (14, 30, 60):
        a1.axvline(k, color=TXT2, linewidth=0.8, linestyle=":")
        a1.text(k, 1.0, f"{k} d", ha="center", va="bottom", fontsize=7, color=TXT2)
    a1.axhline(0.5, color=TXT2, linewidth=0.8)
    a1.set_xlabel("defasagem (dias)")
    a1.set_ylabel("autocorrelação de log(Q+1)")
    a1.set_title("Memória da vazão", loc="left")
    a1.legend()
    a2.axhline(0, color=TXT2, linewidth=0.8)
    a2.set_xlabel("defasagem da chuva (dias)")
    a2.set_ylabel("Spearman: chuva(t−k) × ΔQ(t)")
    a2.set_title("Resposta à chuva", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "04_memoria.png")
    plt.close(fig)
    return memoria


def tabela_resumo(d, memoria):
    linhas = []
    for p, df in d.items():
        q = df["vazao"].dropna()
        linhas.append({
            "posto": p.upper(), "area_km2": POSTOS[p]["area_km2"],
            "inicio": q.index.min().date(), "fim": q.index.max().date(),
            "anos_com_dado": round(len(q) / 365.25, 1),
            "q_media": round(q.mean(), 1), "q_mediana": round(q.median(), 1),
            "q95_estiagem": round(q.quantile(0.05), 1), "q5_cheia": round(q.quantile(0.95), 1),
            "assimetria": round(q.skew(), 2), "assimetria_log": round(np.log1p(q).skew(), 2),
            "q_esp_media_Lskm2": round(df["q_esp"].mean(), 1),
            "chuva_anual_mm": round(df["chuva"].resample("YS").sum().mean()),
            "coef_escoamento": round(q.mean() * 86400 * 365.25 / (POSTOS[p]["area_km2"] * 1e6)
                                     / (df["chuva"].resample("YS").sum().mean() / 1000), 2),
            "memoria_dias_acf<0.5": memoria[p],
        })
    t = pd.DataFrame(linhas)
    t.to_csv(TAB / "resumo_postos.csv", index=False)
    return t


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    d = carregar()
    fig_disponibilidade(d)
    fig_sazonalidade(d)
    fig_permanencia(d)
    memoria = fig_memoria(d)
    t = tabela_resumo(d, memoria)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(t.T.to_string(header=False))
    print(f"\nfiguras em {FIG}/ e tabela em {TAB}/resumo_postos.csv")


if __name__ == "__main__":
    main()
