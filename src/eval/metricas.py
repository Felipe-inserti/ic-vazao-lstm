"""Métricas de desempenho hidrológico (todas em m³/s, depois de desfazer log e escala).

Convenções:
- NSE  = 1 - SSE / variância das observações (Nash & Sutcliffe, 1970).
- KGE  = 1 - sqrt((r-1)² + (alfa-1)² + (beta-1)²), alfa = sd_sim/sd_obs, beta = média_sim/média_obs (Gupta et al., 2009).
- PBIAS = 100 * soma(obs - sim) / soma(obs). Positivo = modelo SUBESTIMA (Moriasi et al., 2007).
- R²   = quadrado da correlação de Pearson (convenção hidrológica; não é o r2_score do sklearn, que é o NSE).
"""
import numpy as np


def _limpar(obs, sim):
    obs, sim = np.asarray(obs, float), np.asarray(sim, float)
    ok = np.isfinite(obs) & np.isfinite(sim)
    return obs[ok], sim[ok]


def nse(obs, sim):
    obs, sim = _limpar(obs, sim)
    return 1 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2)


def kge(obs, sim):
    obs, sim = _limpar(obs, sim)
    r = np.corrcoef(obs, sim)[0, 1]
    alfa = sim.std() / obs.std()
    beta = sim.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alfa - 1) ** 2 + (beta - 1) ** 2)


def pbias(obs, sim):
    obs, sim = _limpar(obs, sim)
    return 100 * np.sum(obs - sim) / np.sum(obs)


def rmse(obs, sim):
    obs, sim = _limpar(obs, sim)
    return np.sqrt(np.mean((obs - sim) ** 2))


def r2(obs, sim):
    obs, sim = _limpar(obs, sim)
    return np.corrcoef(obs, sim)[0, 1] ** 2


def todas(obs, sim) -> dict:
    return {"NSE": nse(obs, sim), "KGE": kge(obs, sim), "PBIAS": pbias(obs, sim),
            "RMSE": rmse(obs, sim), "R2": r2(obs, sim)}
