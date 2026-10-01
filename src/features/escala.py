"""Escala do pipeline da LSTM multi-horizonte (ver docs/decisoes.md, 2026-10-01).

Vazão: log1p (a série é assimétrica; a memória da bacia já é medida em log Q) e depois MinMax.
Chuva/Tmax/Tmin: só MinMax. Sempre ajustada no treino da dobra (sem o último ano, reservado
para a parada antecipada) — nunca no conjunto inteiro. As métricas finais desfazem essa escala
e voltam para m³/s antes de NSE/KGE/PBIAS/RMSE/R².

MinMax implementado em numpy puro (sem scikit-learn): este módulo é usado também fora de
`.venv-ml`, e a conta é trivial o bastante pra não justificar a dependência extra.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

COLUNAS_METEO = ["chuva", "tmax", "tmin"]


@dataclass
class _MinMax:
    minimo: np.ndarray
    amplitude: np.ndarray  # máximo - mínimo; nunca zero (ver _ajustar)

    def transformar(self, x):
        return (np.asarray(x, dtype="float64") - self.minimo) / self.amplitude

    def inverso(self, z):
        return np.asarray(z, dtype="float64") * self.amplitude + self.minimo


def _ajustar(valores) -> _MinMax:
    valores = np.asarray(valores, dtype="float64")
    minimo, maximo = np.nanmin(valores, axis=0), np.nanmax(valores, axis=0)
    amplitude = np.where(maximo > minimo, maximo - minimo, 1.0)  # evita divisão por zero
    return _MinMax(minimo, amplitude)


@dataclass
class Escala:
    vazao: _MinMax
    meteo: _MinMax


def ajustar(treino: pd.DataFrame) -> Escala:
    """Ajusta só com os dias de `treino` (já sem o ano de parada antecipada)."""
    vazao_log = np.log1p(treino["vazao"].dropna().to_numpy())
    sc_vazao = _ajustar(vazao_log)
    sc_meteo = _ajustar(treino[COLUNAS_METEO].dropna().to_numpy())
    return Escala(sc_vazao, sc_meteo)


def transformar_vazao(esc: Escala, vazao) -> np.ndarray:
    """Aplica log1p + MinMax; aceita qualquer formato."""
    return esc.vazao.transformar(np.log1p(np.asarray(vazao, dtype="float64")))


def desfazer_vazao(esc: Escala, z) -> np.ndarray:
    """Inverso de `transformar_vazao`: volta para m³/s."""
    return np.expm1(esc.vazao.inverso(z))


def transformar_meteo(esc: Escala, valores) -> np.ndarray:
    return esc.meteo.transformar(valores)
