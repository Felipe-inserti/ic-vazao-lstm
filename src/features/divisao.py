"""Divisão temporal dos dados (ver docs/decisoes.md).

- Teste: 2015-01-01 a 2019-12-31 -> trancado até a avaliação final.
- Validação walk-forward (janela expansiva): 10 dobras, cada uma valida um ano
  de 2005 a 2014 (anos sem dado válido não contribuem) e treina com TUDO o que vem antes desse ano.
A data que conta é a de EMISSÃO da previsão (dia t). Para o alvo não invadir
o período seguinte, exige-se que t + H também esteja no mesmo período.
"""
import pandas as pd

INICIO_TESTE, FIM_TESTE = pd.Timestamp("2015-01-01"), pd.Timestamp("2019-12-31")
ANOS_VALIDACAO = list(range(2005, 2015))


def dobras():
    """Lista de (nome, fim_do_treino_exclusivo, inicio_val, fim_val)."""
    return [(f"val{a}", pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{a}-01-01"), pd.Timestamp(f"{a}-12-31"))
            for a in ANOS_VALIDACAO]


def resolver(dobra: str):
    """(fim_treino, inicio_avaliacao, fim_avaliacao) de uma dobra de validação (ex.: "val2014")
    ou do teste final ("teste": treino = tudo antes de 2015, avaliação = 2015-2019, roda uma
    única vez — ver docs/decisoes.md, 2026-10-01, PRÉ-REGISTRO)."""
    if dobra == "teste":
        return INICIO_TESTE, INICIO_TESTE, FIM_TESTE
    _, fim_treino, ini_val, fim_val = next(d for d in dobras() if d[0] == dobra)
    return fim_treino, ini_val, fim_val


def mascara_periodo(datas_emissao, h: int, inicio, fim):
    """True se a emissão E o último dia do alvo (t+h) caem dentro de [inicio, fim]."""
    t = pd.to_datetime(datas_emissao)
    return (t >= inicio) & (t + pd.Timedelta(days=h) <= fim)
