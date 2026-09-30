"""Consistência da vazão: comparação com postos vizinhos (razão anual e dupla massa).

Para cada par (nosso posto, vizinho no mesmo rio), usando só os dias em que
os dois têm dado, compara a vazão ESPECÍFICA (L/s·km²):
  - razão anual  = soma(q_esp posto) / soma(q_esp vizinho) no ano.
    Medição consistente -> razão estável ao longo dos anos.
    Salto de patamar num dos postos -> a razão pula naquele período.
  - dupla massa  = vazão acumulada do posto × acumulada do vizinho.
    Consistente -> reta. A figura mostra o DESVIO em relação à reta,
    onde uma quebra de inclinação marca o início/fim de um problema.
Usa a vazão SEM a limpeza (data/interim/vazao), para testar os próprios
períodos marcados em configs/periodos_invalidos.yaml (sombreados na figura).

Uso: python -m src.qa.dupla_massa
"""
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.ingest.ana_vazoes_parser import escolher_nivel, para_formato_longo

POSTOS = yaml.safe_load(Path("configs/postos.yaml").read_text(encoding="utf-8"))
INVALIDOS = yaml.safe_load(Path("configs/periodos_invalidos.yaml").read_text(encoding="utf-8")) or {}
RAW = Path("data/raw/hidroweb/vizinhos")
VAZAO = Path("data/interim/vazao")
FIG, TAB = Path("results/figuras"), Path("results/tabelas")
DESDE = 1995            # anos analisados
MIN_DIAS_ANO = 200      # dias em comum para um ano entrar na razão
LIMIAR = 0.25           # desvio da razão anual em relação à mediana do par que merece alerta

# vizinhos no mesmo rio (inventário ANA): código -> (nome, área km²)
VIZINHOS = {
    "ivr": {"62754000": ("Faz. Pau d'Alho", 2450), "62755000": ("Faz. Santo Anselmo", 2520)},
    "gap": {"62777000": ("Faz. Candoca", 2430), "62778000": ("Faz. Boa Vista de Jacaré", 3520)},
    "fsb": {"62795000": ("Faz. Santa Heloísa", 1360)},
}
COR = {"ivr": "#2a78d6", "gap": "#eb6834", "fsb": "#1baf7a"}


def ler_vizinho(codigo: str) -> pd.Series | None:
    zips = sorted(RAW.glob(f"Estacao_{codigo}_CSV_*.zip"))
    if not zips:
        return None
    with zipfile.ZipFile(zips[-1]) as z:
        nome = next((n for n in z.namelist() if n.endswith("_Vazoes.csv")), None)
        if nome is None:
            return None
        texto = z.read(nome).decode("latin-1")
    linhas = texto.splitlines()
    ini = next(i for i, l in enumerate(linhas) if l.startswith("EstacaoCodigo"))
    df = pd.read_csv(io.StringIO("\n".join(linhas[ini:])), sep=";", decimal=",",
                     dtype=str, keep_default_na=False)
    if df.empty:
        return None
    serie = escolher_nivel(para_formato_longo(df))
    (VAZAO / "vizinhos").mkdir(parents=True, exist_ok=True)
    serie.to_parquet(VAZAO / "vizinhos" / f"{codigo}.parquet", index=False)
    return serie.set_index("data")["vazao"]


def comparar(posto: str, codigo: str, q_viz: pd.Series, area_viz: float):
    q = pd.read_parquet(VAZAO / f"{posto}.parquet").set_index("data")["vazao"]
    x = q / POSTOS[posto]["area_km2"] * 1000
    v = q_viz / area_viz * 1000
    comum = pd.concat({"x": x, "v": v}, axis=1).dropna()
    comum = comum[comum.index.year >= DESDE]
    if len(comum) < 365:
        return None, None
    anual = comum.groupby(comum.index.year).agg(x=("x", "sum"), v=("v", "sum"), dias=("x", "size"))
    anual = anual[anual["dias"] >= MIN_DIAS_ANO]
    razao = (anual["x"] / anual["v"]).rename(f"{posto.upper()}/{codigo}")

    mensal = comum.resample("MS").mean().dropna()
    cx, cv = mensal["x"].cumsum(), mensal["v"].cumsum()
    inclinacao = cx.iloc[-1] / cv.iloc[-1]
    desvio = (cx - inclinacao * cv) / cx.iloc[-1] * 100   # % do total acumulado
    return razao, desvio


def figura(curvas):
    fig, axs = plt.subplots(len(curvas), 1, figsize=(8, 1.9 * len(curvas)), sharex=True, squeeze=False)
    for ax, (posto, codigo, nome, desvio) in zip(axs[:, 0], curvas):
        for per in INVALIDOS.get(posto, []):
            ax.axvspan(pd.Timestamp(per["inicio"]), pd.Timestamp(per["fim"]), color="#f0efec", zorder=0)
        ax.plot(desvio.index, desvio.values, color=COR[posto], linewidth=1.8)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(f"{posto.upper()} × {codigo} ({nome})", loc="left", fontsize=9, fontweight="bold")
        ax.set_ylabel("desvio da\nreta (%)", fontsize=8)
        ax.grid(color="#e6e5e1", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Dupla massa: desvio da vazão acumulada em relação à reta "
                 "(cinza = períodos já marcados como inválidos)", x=0.01, ha="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "06_dupla_massa.png", dpi=200)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    razoes, curvas = [], []
    for posto, viz in VIZINHOS.items():
        for codigo, (nome, area) in viz.items():
            qv = ler_vizinho(codigo)
            if qv is None:
                print(f"[sem arquivo] {codigo} ({nome})")
                continue
            razao, desvio = comparar(posto, codigo, qv, area)
            validos = qv.dropna()
            if razao is None or razao.empty:
                print(f"[pouco dado em comum] {posto.upper()} × {codigo} ({nome}): "
                      f"vizinho tem dado de {validos.index.min().date()} a {validos.index.max().date()}")
                continue
            print(f"[ok] {posto.upper()} × {codigo} ({nome}): {len(razao)} anos em comum "
                  f"({razao.index.min()}-{razao.index.max()}), razão mediana {razao.median():.2f}")
            razoes.append(razao)
            curvas.append((posto, codigo, nome, desvio))
    if not razoes:
        print("nenhum par com dados em comum; nada a comparar.")
        return
    tabela = pd.concat(razoes, axis=1).sort_index()
    tabela.to_csv(TAB / "razao_anual_vizinhos.csv")
    figura(curvas)

    rel = tabela / tabela.median() - 1
    marcada = tabela.round(2).astype(str).where(rel.abs() <= LIMIAR, tabela.round(2).astype(str) + " *")
    marcada = marcada.where(tabela.notna(), "")
    print(f"\nrazão anual da vazão específica (posto/vizinho); * = desvio > {LIMIAR:.0%} da mediana do par")
    with pd.option_context("display.width", 200):
        print(marcada.to_string())
    print(f"\nfigura em {FIG}/06_dupla_massa.png | tabela em {TAB}/razao_anual_vizinhos.csv")


if __name__ == "__main__":
    main()
