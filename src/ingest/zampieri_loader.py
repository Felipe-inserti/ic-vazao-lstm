"""Lê os CSVs consolidados de Zampieri (2022-2024) e salva em Parquet.

Problema do arquivo original: a vazão vem entre aspas com um \r dentro
("111.0\r"), o que quebra cada registro em duas linhas no pandas.
Solução: remover os \r antes de interpretar o CSV.

Uso: python src/ingest/zampieri_loader.py
"""
import io
from pathlib import Path

import pandas as pd

RAW = Path("data/raw/zampieri")
OUT = Path("data/interim/zampieri")
ARQUIVOS = {
    "ivr": "InvernadaRecreio_DadosConsolidados_2022_2024-CSV (a ser processado).csv",
    "gap": "GaviaoPeixoto_DadosConsolidados_2022_2024-CSV (a ser processado).csv",
    "fsb": "FazendaSaoBenedito_DadosConsolidados_2022_2024_CSV (a ser processado).csv",
}


def carregar(caminho: Path) -> pd.DataFrame:
    texto = caminho.read_text(encoding="utf-8-sig").replace("\r", "")
    df = pd.read_csv(io.StringIO(texto), sep=";")
    df["data"] = pd.to_datetime(df.pop("Data Leitura"), format="%d/%m/%Y")
    df = df.drop(columns=["Bacia Estudo"])
    num = df.columns.drop("data")
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df[["data", *num]].sort_values("data").reset_index(drop=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for nome, arq in ARQUIVOS.items():
        caminho = RAW / arq
        if not caminho.exists():
            print(f"[pulando] {caminho} não encontrado")
            continue
        df = carregar(caminho)
        df.to_parquet(OUT / f"{nome}.parquet", index=False)
        esperado = pd.date_range(df["data"].min(), df["data"].max(), freq="D")
        faltam = esperado.difference(df["data"])
        print(f"\n== {nome.upper()} ==")
        print(f"linhas: {len(df)}  colunas: {df.shape[1]}  "
              f"período: {df['data'].min().date()} a {df['data'].max().date()}")
        print(f"datas duplicadas: {df['data'].duplicated().sum()}  "
              f"dias ausentes no calendário: {len(faltam)}")
        if len(faltam):
            print(f"  primeiros ausentes: {[d.date().isoformat() for d in faltam[:10]]}")
        nulos = df.isna().sum()
        print(f"valores nulos: {dict(nulos[nulos > 0]) or 'nenhum'}")
        print(f"vazão (m³/s): min {df['Vazao Observada'].min():.1f}  "
              f"mediana {df['Vazao Observada'].median():.1f}  máx {df['Vazao Observada'].max():.1f}")


if __name__ == "__main__":
    main()
