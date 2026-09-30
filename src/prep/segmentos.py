"""Lista blocos contínuos de vazão após interpolar lacunas curtas.

Uso: python src/prep/segmentos.py [max_lacuna_dias]   (padrão: 7)
"""
import sys
from pathlib import Path

import pandas as pd

MAX_GAP = int(sys.argv[1]) if len(sys.argv) > 1 else 7
IN = Path("data/interim/vazao")


def segmentar(s: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    falta = s["vazao"].isna()
    bloco = (falta != falta.shift()).cumsum()
    tam = falta.groupby(bloco).transform("sum")
    # lacuna longa = fronteira; lacuna curta = será interpolada
    fronteira = falta & (tam > max_gap)
    seg_id = (fronteira != fronteira.shift()).cumsum()
    validos = s[~fronteira].assign(seg=seg_id[~fronteira])
    return (validos.groupby("seg")
            .agg(inicio=("data", "min"), fim=("data", "max"), dias=("data", "size"),
                 interpolados=("vazao", lambda v: v.isna().sum()))
            .reset_index(drop=True))


for arq in sorted(IN.glob("*.parquet")):
    s = pd.read_parquet(arq)
    segs = segmentar(s, MAX_GAP)
    grandes = segs[segs["dias"] >= 365].copy()
    grandes["anos"] = (grandes["dias"] / 365.25).round(1)
    print(f"\n== {arq.stem.upper()} (lacunas <= {MAX_GAP} dias interpoladas) ==")
    print(grandes.to_string(index=False))
    print(f"total em blocos >= 1 ano: {grandes['dias'].sum()} dias "
          f"({grandes['dias'].sum() / 365.25:.1f} anos)")
    zeros = s.loc[s["vazao"] == 0, "data"].dt.date.tolist()
    if zeros:
        print(f"dias com vazão zero: {zeros}")
