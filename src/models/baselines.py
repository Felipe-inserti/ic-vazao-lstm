"""Baselines de referência: persistência e climatologia.

- Persistência: Q(t+k) = Q(t) para todo k. Não treina nada.
- Climatologia: Q(t+k) = média da vazão no mesmo dia do ano, calculada SÓ no treino,
  suavizada com média móvel circular de 31 dias.
"""
import numpy as np
import pandas as pd


def persistencia(q_emissao: np.ndarray, h: int) -> np.ndarray:
    return np.repeat(q_emissao[:, None], h, axis=1)


def ajustar_climatologia(datas, vazao, janela: int = 31) -> pd.Series:
    s = pd.Series(np.asarray(vazao, float), index=pd.to_datetime(datas)).dropna()
    doy = s.index.dayofyear.where(s.index.dayofyear <= 365, 365)
    media = s.groupby(doy).mean().reindex(range(1, 366)).interpolate(limit_direction="both")
    ext = pd.concat([media.iloc[-janela:], media, media.iloc[:janela]])
    return ext.rolling(janela, center=True).mean().iloc[janela:-janela]


def climatologia(clim: pd.Series, datas_emissao, h: int) -> np.ndarray:
    t = pd.to_datetime(datas_emissao)
    saida = np.empty((len(t), h))
    for k in range(1, h + 1):
        d = (t + pd.Timedelta(days=k)).dayofyear
        saida[:, k - 1] = clim.reindex(np.minimum(d, 365)).to_numpy()
    return saida
