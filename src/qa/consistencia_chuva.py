"""Consistência da vazão contra a chuva (independente da rede de réguas).

Os postos vizinhos não têm dados após 2010, então a referência é a chuva
média da bacia (grade de Xavier), que não depende da medição de nível.

1. Balanço anual por ano hidrológico (out-set, rotulado pelo ano em que termina):
   escoamento R (mm) e chuva P (mm). Ajusta-se, de forma robusta,
       log R = a + b·log P + c·log P_ano_anterior
   (o ano anterior entra porque o fluxo de base guarda memória).
   Anos com resíduo robusto |z| > 2,5 são suspeitos: escoaram muito mais
   (ou menos) do que a chuva explica.
2. Dupla massa mensal: chuva acumulada × escoamento acumulado. A figura
   mostra o desvio em relação à reta; quebras de inclinação marcam
   mudanças de patamar da medição.
Usa a vazão SEM a limpeza (data/interim/vazao). Os períodos já marcados
como inválidos aparecem sombreados, para conferência.

Uso: python -m src.qa.consistencia_chuva
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

POSTOS = yaml.safe_load(Path("configs/postos.yaml").read_text(encoding="utf-8"))
INVALIDOS = yaml.safe_load(Path("configs/periodos_invalidos.yaml").read_text(encoding="utf-8")) or {}
VAZAO, PROC = Path("data/interim/vazao"), Path("data/processed")
FIG, TAB = Path("results/figuras"), Path("results/tabelas")
MIN_DIAS_ANO, MIN_DIAS_MES, Z_LIMIAR = 300, 25, 2.5
COR = {"ivr": "#2a78d6", "gap": "#eb6834", "fsb": "#1baf7a"}


def serie_diaria(p: str) -> pd.DataFrame:
    q = pd.read_parquet(VAZAO / f"{p}.parquet").set_index("data")["vazao"]
    chuva = pd.read_parquet(PROC / f"{p}.parquet").set_index("data")["chuva"]
    df = pd.concat({"vazao": q, "chuva": chuva}, axis=1, sort=True)
    df["esc_mm"] = df["vazao"] * 86400 / (POSTOS[p]["area_km2"] * 1e6) * 1000  # mm/dia
    return df.dropna(subset=["chuva"])


def ano_hidrologico(idx: pd.DatetimeIndex) -> np.ndarray:
    return np.where(idx.month >= 10, idx.year + 1, idx.year)


def balanco_anual(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(ano_hidrologico(df.index))
    a = pd.DataFrame({"P": g["chuva"].sum(), "dias": g["vazao"].count(),
                      "R": g["esc_mm"].mean() * g["chuva"].size()})
    a.index.name = "ano_hidro"
    a["P_ant"] = a["P"].shift(1)
    return a


def ajuste_robusto(a: pd.DataFrame, iteracoes: int = 3) -> pd.DataFrame:
    ok = a[(a["dias"] >= MIN_DIAS_ANO) & a["P_ant"].notna()].copy()
    X = np.column_stack([np.ones(len(ok)), np.log(ok["P"]), np.log(ok["P_ant"])])
    y = np.log(ok["R"].to_numpy())
    usar = np.ones(len(ok), bool)
    for _ in range(iteracoes):
        beta, *_ = np.linalg.lstsq(X[usar], y[usar], rcond=None)
        res = y - X @ beta
        mad = np.median(np.abs(res[usar] - np.median(res[usar]))) * 1.4826
        z = (res - np.median(res[usar])) / mad
        usar = np.abs(z) <= Z_LIMIAR
    ok["R_esperado"] = np.exp(X @ beta)
    ok["razao_obs/esp"] = ok["R"] / ok["R_esperado"]
    ok["z"] = z
    ok["suspeito"] = np.abs(z) > Z_LIMIAR
    return ok


def dupla_massa(df: pd.DataFrame) -> pd.Series:
    m = df.resample("MS").agg({"chuva": "sum", "esc_mm": "mean", "vazao": "count"})
    m = m[m["vazao"] >= MIN_DIAS_MES]
    m["esc_mes"] = m["esc_mm"] * m.index.days_in_month
    cp, cr = m["chuva"].cumsum(), m["esc_mes"].cumsum()
    inclinacao = cr.iloc[-1] / cp.iloc[-1]
    return (cr - inclinacao * cp) / cr.iloc[-1] * 100


def figura(curvas: dict):
    fig, axs = plt.subplots(len(curvas), 1, figsize=(8, 2.1 * len(curvas)), sharex=True, squeeze=False)
    for ax, (p, desvio) in zip(axs[:, 0], curvas.items()):
        for per in INVALIDOS.get(p, []):
            ax.axvspan(pd.Timestamp(per["inicio"]), pd.Timestamp(per["fim"]), color="#f0efec", zorder=0)
        ax.plot(desvio.index, desvio.values, color=COR[p], linewidth=1.6)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(p.upper(), loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("desvio da\nreta (%)", fontsize=8)
        ax.grid(color="#e6e5e1", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Dupla massa chuva × escoamento (cinza = períodos já marcados como inválidos)",
                 x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "06_dupla_massa_chuva.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    curvas, tabelas = {}, []
    for p in POSTOS:
        df = serie_diaria(p)
        aj = ajuste_robusto(balanco_anual(df))
        aj.insert(0, "posto", p.upper())
        tabelas.append(aj)
        curvas[p] = dupla_massa(df)

        print(f"\n== {p.upper()} ==  (ano hidrológico out-set; * = |z| > {Z_LIMIAR})")
        mostra = aj[["P", "R", "R_esperado", "razao_obs/esp", "z"]].round(
            {"P": 0, "R": 0, "R_esperado": 0, "razao_obs/esp": 2, "z": 1})
        mostra["flag"] = np.where(aj["suspeito"], "*", "")
        print(mostra[mostra.index >= 1995].to_string())
    pd.concat(tabelas).to_csv(TAB / "consistencia_chuva_anual.csv")
    figura(curvas)
    print(f"\nfigura em {FIG}/06_dupla_massa_chuva.png | tabela em {TAB}/consistencia_chuva_anual.csv")


if __name__ == "__main__":
    main()
