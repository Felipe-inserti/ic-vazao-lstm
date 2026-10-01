"""Baselines de referência: persistência e climatologia.

- Persistência: Q(t+k) = Q(t) para todo k. Não treina nada.
- Climatologia: Q(t+k) = média da vazão no mesmo dia do ano, calculada SÓ no treino,
  suavizada com média móvel circular de 31 dias.
"""
import numpy as np
import pandas as pd

from src.features import amostras, divisao, janelas


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


def referencia_dobra(df: pd.DataFrame, dobra: str) -> dict | None:
    """Persistência e climatologia nos dias de validação de UMA dobra (N_MAX=90, H_MAX=30,
    treino = tudo antes de fim_treino — mesmos critérios de src/rodar_baselines.py e
    src/treinar_lstm.py). Não depende de TensorFlow: útil pra comparar com a LSTM sem precisar
    do .venv-ml. Devolve None se a dobra não tiver nenhuma amostra de validação.
    """
    fim_treino, ini_val, fim_val = divisao.resolver(dobra)
    t_idx = janelas.indices_validos(df["segmento"], amostras.N_MAX_BASELINE, amostras.H_MAX_BASELINE)
    datas = df["data"].iloc[t_idx].to_numpy()
    q_t = df["vazao"].to_numpy()[t_idx]

    m_val = divisao.mascara_periodo(datas, amostras.H_MAX_BASELINE, ini_val, fim_val)
    if not m_val.any():
        return None

    treino = df[df["data"] < fim_treino]
    clim = ajustar_climatologia(treino["data"], treino["vazao"])
    return {
        "datas_val": datas[m_val],
        "persistencia": persistencia(q_t[m_val], amostras.H_MAX_BASELINE),
        "climatologia": climatologia(clim, datas[m_val], amostras.H_MAX_BASELINE),
    }
