import numpy as np
import pandas as pd

from src.features import janelas

# 3 segmentos (1, 2, 3) separados por lacunas (NaN)
SEG = pd.Series([1] * 20 + [np.nan] * 5 + [2] * 8 + [np.nan] * 3 + [3] * 30, dtype="Int64")
DF = pd.DataFrame({"data": pd.date_range("2000-01-01", periods=len(SEG)),
                   "vazao": np.arange(len(SEG), dtype=float), "segmento": SEG})


def test_nenhuma_janela_atravessa_lacuna():
    n, h = 5, 3
    for t in janelas.indices_validos(DF["segmento"], n, h):
        trecho = DF["segmento"].iloc[t - n + 1: t + h + 1]
        assert trecho.notna().all() and trecho.nunique() == 1


def test_segmento_curto_demais_nao_gera_amostra():
    # segmento 2 tem 8 dias; n + h = 10 não cabe
    t = janelas.indices_validos(DF["segmento"], 7, 3)
    assert not (DF["segmento"].iloc[t] == 2).any()


def test_alinhamento_entrada_e_alvo():
    n, h = 4, 2
    X, Y, datas = janelas.montar(DF, ["vazao"], "vazao", n, h)
    t0 = DF.index[DF["data"] == datas[0]][0]
    assert X[0, -1, 0] == DF["vazao"][t0]          # último dia da entrada = dia t
    assert Y[0, 0] == DF["vazao"][t0 + 1]          # primeiro alvo = dia t+1
    assert Y[0, -1] == DF["vazao"][t0 + h]
