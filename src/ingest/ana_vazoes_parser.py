"""Converte o *_Vazoes.csv do HidroWeb (formato largo, 1 linha por mês)
em série diária longa: data, vazao, nivel, status, metodo.

Regras:
- Dia sem valor no arquivo  -> NaN (sem medição). Zero só se a ANA gravou 0.
- Mesmo dia com nível 1 (bruto) e 2 (consistido) -> fica o consistido.
- Dias inexistentes no mês (ex.: 31/11) são descartados.

Uso:
    python src/ingest/ana_vazoes_parser.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

POSTOS = {"ivr": "62752000", "gap": "62776800", "fsb": "62800000"}
RAW = Path("data/raw/hidroweb")
OUT = Path("data/interim/vazao")


def ler_csv_ana(caminho: Path) -> pd.DataFrame:
    linhas = caminho.read_text(encoding="latin-1").splitlines()
    inicio = next(i for i, l in enumerate(linhas) if l.startswith("EstacaoCodigo"))
    return pd.read_csv(
        caminho, sep=";", skiprows=inicio, encoding="latin-1",
        decimal=",", dtype=str, keep_default_na=False,
    )


def para_formato_longo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["mes"] = pd.to_datetime(df["Data"], format="%d/%m/%Y")
    df["nivel"] = df["NivelConsistencia"].astype(int)
    df["metodo"] = pd.to_numeric(df["MetodoObtencaoVazoes"], errors="coerce")

    base = ["mes", "nivel", "metodo"]
    dias = [f"Vazao{d:02d}" for d in range(1, 32)]
    stat = [f"Vazao{d:02d}Status" for d in range(1, 32)]

    v = df.melt(id_vars=base, value_vars=dias, var_name="dia", value_name="vazao")
    s = df.melt(id_vars=base, value_vars=stat, var_name="dia", value_name="status")
    v["dia"] = v["dia"].str[5:7].astype(int)
    s["dia"] = s["dia"].str[5:7].astype(int)
    longo = v.merge(s, on=base + ["dia"])

    # descarta dias que não existem no mês
    longo = longo[longo["dia"] <= longo["mes"].dt.days_in_month]
    longo["data"] = longo["mes"] + pd.to_timedelta(longo["dia"] - 1, unit="D")

    # "" -> NaN (sem medição); "0" continua 0
    longo["vazao"] = pd.to_numeric(
        longo["vazao"].str.replace(",", ".", regex=False).replace("", np.nan),
        errors="coerce",
    )
    longo["status"] = pd.to_numeric(longo["status"].replace("", np.nan), errors="coerce")
    return longo[["data", "vazao", "nivel", "status", "metodo"]]


def escolher_nivel(longo: pd.DataFrame) -> pd.DataFrame:
    """Por dia, fica o consistido (2) se ele tiver valor; senão o bruto (1)."""
    longo = longo.assign(tem_valor=longo["vazao"].notna())
    longo = longo.sort_values(["data", "tem_valor", "nivel"], ascending=[True, False, False])
    diario = longo.drop_duplicates("data", keep="first").drop(columns="tem_valor")
    # índice diário contínuo: dias ausentes do arquivo viram NaN explícito
    idx = pd.date_range(diario["data"].min(), diario["data"].max(), freq="D")
    return diario.set_index("data").reindex(idx).rename_axis("data").reset_index()


def resumo(nome: str, s: pd.DataFrame) -> None:
    falta = s["vazao"].isna()
    blocos = (falta != falta.shift()).cumsum()
    lacunas = falta.groupby(blocos).sum()
    lacunas = lacunas[lacunas > 0].sort_values(ascending=False)
    print(f"\n== {nome.upper()} ==")
    print(f"período        : {s['data'].min().date()} a {s['data'].max().date()} ({len(s)} dias)")
    print(f"faltantes      : {falta.sum()} ({falta.mean():.1%})")
    print(f"vazão zero     : {(s['vazao'] == 0).sum()} dias")
    com_valor = s[~falta]
    print(f"nível 2 / 1    : {(com_valor['nivel'] == 2).sum()} / {(com_valor['nivel'] == 1).sum()} dias com valor")
    print(f"5 maiores lacunas (dias): {lacunas.head(5).astype(int).tolist()}")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for nome, codigo in POSTOS.items():
        arq = RAW / nome / f"{codigo}_Vazoes.csv"
        if not arq.exists():
            print(f"[pulando] {arq} não encontrado")
            continue
        serie = escolher_nivel(para_formato_longo(ler_csv_ana(arq)))
        serie.to_parquet(OUT / f"{nome}.parquet", index=False)
        resumo(nome, serie)


if __name__ == "__main__":
    main()
