"""Monta a tabela diária de cada posto: vazão + chuva + Tmax + Tmin.

Passos:
1. Meteorologia da bacia = média dos pontos da grade de Xavier num círculo
   com a mesma área da bacia, centrado no posto (versão 1, simplificada).
2. Limpeza documentada (docs/decisoes.md): zeros suspeitos viram falta.
3. Interpola lacunas de vazão <= MAX_GAP dias; lacunas maiores ficam NaN
   e separam segmentos (nenhuma janela do modelo pode atravessá-las).
4. Salva data/processed/{posto}.parquet e imprime um resumo + checagem
   chuva -> vazão (correlação por defasagem).

Uso: python src/prep/montar_base.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

MAX_GAP = 7
POSTOS = yaml.safe_load(Path("configs/postos.yaml").read_text(encoding="utf-8"))
METEO = Path("data/interim/meteo")
VAZAO = Path("data/interim/vazao")
OUT = Path("data/processed")

# datas tratadas como falta (ver docs/decisoes.md)
DATAS_INVALIDAS = {"fsb": ["2016-11-28", "2016-11-29"]}
PERIODOS_INVALIDOS = yaml.safe_load(Path("configs/periodos_invalidos.yaml").read_text(encoding="utf-8")) or {}


def mascara_circulo(lat, lon, lat0, lon0, raio_km):
    """True para pontos da grade a até raio_km do posto."""
    la, lo = np.meshgrid(lat, lon, indexing="ij")
    dy = (la - lat0) * 110.57
    dx = (lo - lon0) * 111.32 * np.cos(np.radians(lat0))
    return np.sqrt(dx**2 + dy**2) <= raio_km


def meteo_da_bacia(cfg: dict) -> pd.DataFrame:
    raio = np.sqrt(cfg["area_km2"] / np.pi)
    series, n_pontos = {}, None
    for var in ["pr", "Tmax", "Tmin"]:
        da = xr.open_dataset(METEO / f"xavier_{var}_tietejacare.nc")[var]
        m = mascara_circulo(da.latitude.values, da.longitude.values,
                            cfg["lat"], cfg["lon"], raio)
        n_pontos = int(m.sum())
        mask = xr.DataArray(m, dims=("latitude", "longitude"),
                            coords={"latitude": da.latitude, "longitude": da.longitude})
        series[var] = da.where(mask).mean(["latitude", "longitude"]).to_pandas()
    df = pd.DataFrame(series).rename(columns={"pr": "chuva", "Tmax": "tmax", "Tmin": "tmin"})
    df.index.name = "data"
    return df, raio, n_pontos


def limpar_e_segmentar(s: pd.DataFrame, posto: str) -> pd.DataFrame:
    s = s[["data", "vazao"]].copy()
    ruins = pd.to_datetime(DATAS_INVALIDAS.get(posto, []))
    s.loc[s["data"].isin(ruins), "vazao"] = np.nan
    for per in PERIODOS_INVALIDOS.get(posto, []):
        dentro = s["data"].between(pd.Timestamp(per["inicio"]), pd.Timestamp(per["fim"]))
        s.loc[dentro, "vazao"] = np.nan

    falta = s["vazao"].isna()
    bloco = (falta != falta.shift()).cumsum()
    tam = falta.groupby(bloco).transform("sum")
    curta = falta & (tam <= MAX_GAP)
    longa = falta & ~curta

    s["interpolado"] = curta
    s["vazao"] = s["vazao"].interpolate(limit_area="inside")  # preenche tudo por dentro...
    s.loc[longa, "vazao"] = np.nan                            # ...e devolve NaN às lacunas longas
    s["segmento"] = (longa != longa.shift()).cumsum().where(~longa)
    s["segmento"] = s["segmento"].rank(method="dense").astype("Int64")
    return s


def checar_resposta(df: pd.DataFrame) -> tuple[int, float]:
    """Spearman entre chuva(t-k) e o aumento da vazão Q(t)-Q(t-1); devolve o k de pico."""
    dq = df["vazao"].diff()
    corr = {k: dq.corr(df["chuva"].shift(k), method="spearman") for k in range(0, 11)}
    k = max(corr, key=corr.get)
    return k, corr[k]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for posto, cfg in POSTOS.items():
        q = pd.read_parquet(VAZAO / f"{posto}.parquet")
        q = limpar_e_segmentar(q, posto)
        met, raio, n = meteo_da_bacia(cfg)
        df = q.merge(met, left_on="data", right_index=True, how="left")
        df.to_parquet(OUT / f"{posto}.parquet", index=False)

        k, r = checar_resposta(df)
        validos = df["vazao"].notna()
        print(f"\n== {posto.upper()} ==")
        print(f"raio {raio:.0f} km, {n} pontos da grade")
        print(f"período {df['data'].min().date()} a {df['data'].max().date()}  "
              f"| dias com vazão {validos.sum()}  | interpolados {df['interpolado'].sum()}  "
              f"| segmentos {df['segmento'].nunique()}")
        print(f"meteo faltante nos dias com vazão: {df.loc[validos, ['chuva', 'tmax', 'tmin']].isna().sum().to_dict()}")
        print(f"chuva média {df['chuva'].mean():.2f} mm/dia  | Tmax {df['tmax'].mean():.1f}  | Tmin {df['tmin'].mean():.1f} °C")
        print(f"chuva -> vazão: pico de correlação com {k} dia(s) de defasagem (Spearman {r:.2f})")


if __name__ == "__main__":
    main()
