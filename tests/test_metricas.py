import numpy as np
import pytest

from src.eval import metricas as m

OBS = np.array([10.0, 12.0, 30.0, 15.0, 11.0, 50.0, 20.0])


def test_previsao_perfeita():
    r = m.todas(OBS, OBS)
    assert r["NSE"] == pytest.approx(1)
    assert r["KGE"] == pytest.approx(1)
    assert r["PBIAS"] == pytest.approx(0)
    assert r["RMSE"] == pytest.approx(0)
    assert r["R2"] == pytest.approx(1)


def test_prever_a_media_da_nse_zero():
    assert m.nse(OBS, np.full_like(OBS, OBS.mean())) == pytest.approx(0)


def test_pbias_positivo_quando_subestima():
    assert m.pbias(OBS, OBS * 0.9) == pytest.approx(10)


def test_ignora_nan():
    sim = OBS.copy()
    sim[2] = np.nan
    assert np.isfinite(m.nse(OBS, sim))
