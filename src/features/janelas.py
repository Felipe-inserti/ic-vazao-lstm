"""Janelas de entrada/saída que NUNCA atravessam lacunas longas.

Uma amostra é "emitida" no dia t (fim do dia t, quando já se conhece tudo até t):
  entrada  = dias t-N+1 ... t      (N dias de histórico)
  alvo     = vazão em t+1 ... t+H  (H dias à frente)
Ela só é válida se os N+H dias pertencem ao MESMO segmento contínuo.
"""
import numpy as np
import pandas as pd


def indices_validos(segmento: pd.Series, n: int, h: int) -> np.ndarray:
    """Posições t (inteiras) em que [t-n+1, t+h] está inteiro dentro de um segmento."""
    seg = segmento.to_numpy(dtype="float64", na_value=np.nan)
    total = len(seg)
    t = np.arange(n - 1, total - h)
    ini, fim = seg[t - n + 1], seg[t + h]
    ok = np.isfinite(ini) & (ini == fim)
    # segmentos são blocos contíguos: mesmo id no início e no fim => tudo entre eles também
    return t[ok]


def montar(df: pd.DataFrame, entradas: list[str], alvo: str, n: int, h: int, t_idx=None):
    """Devolve X (amostras, n, n_entradas), Y (amostras, h) e as datas de emissão."""
    if t_idx is None:
        t_idx = indices_validos(df["segmento"], n, h)
    v = df[entradas].to_numpy(dtype="float32")
    q = df[alvo].to_numpy(dtype="float32")
    X = np.stack([v[t - n + 1: t + 1] for t in t_idx]) if len(t_idx) else np.empty((0, n, len(entradas)))
    Y = np.stack([q[t + 1: t + h + 1] for t in t_idx]) if len(t_idx) else np.empty((0, h))
    return X, Y, df["data"].iloc[t_idx].to_numpy()
