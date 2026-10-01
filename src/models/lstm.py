"""Modelo LSTM multi-horizonte: um único modelo por posto, mesma arquitetura para os 30 leads
de uma vez (substitui "um modelo por horizonte" — ver docs/decisoes.md, 2026-10-01).

Arquitetura fixada a priori, sem busca de hiperparâmetros: uma camada LSTM(64) + Dense(H).
Treinada na escala do log1p+MinMax (src/features/escala.py); as previsões são desfeitas para
m³/s antes de qualquer métrica.
"""
import tensorflow as tf


def construir(n_entradas: int, n: int, h: int, unidades: int = 64) -> tf.keras.Model:
    m = tf.keras.Sequential([
        tf.keras.Input(shape=(n, n_entradas)),
        tf.keras.layers.LSTM(unidades, activation="tanh"),
        tf.keras.layers.Dense(h),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")
    return m
