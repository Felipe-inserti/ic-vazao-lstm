# Previsão multi-horizonte de vazão com LSTM — Bacia Tietê-Jacaré

Extensão de Zampieri (2025) para horizontes de 1, 7 e 30 dias nos postos IVR, GAP e FSB.

## Ambiente
- Dados: `.venv` (Python 3.14). Modelos: `.venv-ml` (Python 3.12 + TensorFlow 2.21 com GPU), `requirements-ml.txt`.
- WSL: o TensorFlow não acha sozinho as bibliotecas NVIDIA instaladas via pip. Correção (acrescentada ao fim de `.venv-ml/bin/activate`):
  `export LD_LIBRARY_PATH=$(find <site-packages>/nvidia -maxdepth 2 -type d -name lib | paste -sd:):$LD_LIBRARY_PATH`
