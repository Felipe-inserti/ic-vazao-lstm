"""Recorta a grade de Xavier (BR-DWGD) para a região da bacia Tietê-Jacaré.

Lê os NetCDF do Brasil inteiro (grandes, temporários) e salva só a região
da bacia em arquivos pequenos. Depois disso os originais podem ser apagados.

Uso:
    python src/ingest/xavier_recorte.py CAMINHO_DOS_NC
Ex.:
    python src/ingest/xavier_recorte.py /mnt/c/Users/User/Downloads/xavier
"""
import sys
from pathlib import Path

import xarray as xr

# caixa em torno da UGRHI 13 (Tietê-Jacaré), com margem de ~0,3°
LAT_MIN, LAT_MAX = -23.2, -21.2
LON_MIN, LON_MAX = -49.6, -47.4
VARIAVEIS = ["pr", "Tmax", "Tmin"]
OUT = Path("data/interim/meteo")


def main(pasta: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for var in VARIAVEIS:
        arquivos = sorted(pasta.glob(f"{var}_*.nc"))
        if not arquivos:
            print(f"[pulando] nenhum arquivo {var}_*.nc em {pasta}")
            continue
        print(f"{var}: {len(arquivos)} arquivo(s)")
        ds = xr.open_mfdataset(arquivos, combine="by_coords", chunks={"time": 365})
        ds = ds.sortby("latitude").sortby("longitude")
        rec = ds[[var]].sel(latitude=slice(LAT_MIN, LAT_MAX),
                            longitude=slice(LON_MIN, LON_MAX)).load()
        destino = OUT / f"xavier_{var}_tietejacare.nc"
        rec.to_netcdf(destino, encoding={var: {"zlib": True, "complevel": 4}})
        print(f"  -> {destino}  {dict(rec.sizes)}  "
              f"{rec.time.values[0].astype(str)[:10]} a {rec.time.values[-1].astype(str)[:10]}  "
              f"{destino.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]).expanduser())
