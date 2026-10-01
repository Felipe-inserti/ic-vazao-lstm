"""Monta as amostras de entrada/saída do LSTM multi-horizonte (ver docs/decisoes.md, 2026-10-01).

Entradas na janela [t-N+1, t]: vazão (log1p + escala do treino da dobra), chuva, tmax, tmin
(escala do treino da dobra) e seno/cosseno do dia do ano (sem escala, já em [-1, 1]). NENHUMA
coluna meteorológica usa dado de depois do dia t — garantido por tests/test_amostras.py.

Alvo: vazão em t+1..t+30, sempre em m³/s (escala original).

Os índices válidos de emissão vêm de janelas.indices_validos com N_MAX_BASELINE=90 e
H_MAX_BASELINE=30 — os MESMOS usados por src/rodar_baselines.py — para que os dias avaliados
sejam idênticos aos dos baselines, mesmo que o modelo só use os últimos N dias de histórico
(N é um corte da janela de 90, não uma nova varredura de índices).
"""
import numpy as np
import pandas as pd

from src.features import escala, janelas

N_MAX_BASELINE, H_MAX_BASELINE = 90, 30
COLUNAS_METEO = ["chuva", "tmax", "tmin"]
ENTRADAS = ["vazao_esc", *COLUNAS_METEO, "sin_doy", "cos_doy"]


def com_sazonalidade(df: pd.DataFrame) -> pd.DataFrame:
    doy = df["data"].dt.dayofyear.clip(upper=365)
    out = df.copy()
    out["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    out["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    return out


def preparar(df: pd.DataFrame, esc: escala.Escala, n: int, t_idx: np.ndarray | None = None):
    """X (amostras, n, 6) já escalado, Y (amostras, 30) em m³/s, datas de emissão."""
    if t_idx is None:
        t_idx = janelas.indices_validos(df["segmento"], N_MAX_BASELINE, H_MAX_BASELINE)
    d = com_sazonalidade(df)
    d["vazao_esc"] = escala.transformar_vazao(esc, d["vazao"].to_numpy())
    d[COLUNAS_METEO] = escala.transformar_meteo(esc, d[COLUNAS_METEO])
    X_full, Y, datas = janelas.montar(d, ENTRADAS, "vazao", N_MAX_BASELINE, H_MAX_BASELINE, t_idx)
    return X_full[:, -n:, :], Y, datas
