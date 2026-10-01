import numpy as np
import pandas as pd

from src.features import amostras, escala


def _df_identificavel(n_dias=150):
    """Cada coluna meteorológica vale o índice do dia, pra rastrear exatamente quais dias
    entraram em cada janela."""
    idx = np.arange(n_dias, dtype=float)
    return pd.DataFrame({
        "data": pd.date_range("2010-01-01", periods=n_dias),
        "vazao": idx + 1.0,  # > 0, pra não quebrar o log1p
        "chuva": idx,
        "tmax": idx,
        "tmin": idx,
        "segmento": pd.array([1] * n_dias, dtype="Int64"),
    })


def test_entradas_meteorologicas_nao_usam_dias_futuros():
    df = _df_identificavel()
    esc = escala.ajustar(df)
    X, _, _ = amostras.preparar(df, esc, n=60)
    assert len(X) > 0

    for nome in ["chuva", "tmax", "tmin"]:
        col = amostras.ENTRADAS.index(nome)
        # chuva/tmax/tmin crescem com o dia e a escala (log1p+MinMax ou só MinMax) preserva
        # essa ordem: se algum dia depois de t tivesse entrado na janela, ele seria MAIOR que
        # o valor do dia t e apareceria como o máximo. O último passo da janela (dia t) tem
        # que ser sempre o maior valor dela.
        assert np.array_equal(X[:, -1, col], X[:, :, col].max(axis=1))
