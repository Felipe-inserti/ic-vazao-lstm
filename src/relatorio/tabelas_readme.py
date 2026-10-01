"""Gera, em markdown, as tabelas usadas no README.md — direto de results/tabelas/*.csv e
configs/postos.yaml. Nenhum número do README é digitado à mão; o texto do README cita este
script como fonte e cola a saída dele.

Uso: python -m src.relatorio.tabelas_readme
"""
from pathlib import Path

import pandas as pd
import yaml

from src.agregar_validacao import faixa_leads

TAB = Path("results/tabelas")
LEADS = [1, 7, 30]


def para_markdown(df: pd.DataFrame) -> str:
    """Tabela markdown (GFM) simples, sem depender do pacote tabulate."""
    cols = list(df.columns)
    linhas = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        linhas.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join(linhas)


def tabela_teste() -> str:
    """NSE da LSTM, persistência e climatologia no teste final (2015-2019), leads 1/7/30."""
    df = pd.read_csv(TAB / "lstm_teste.csv")
    df = df[(df["lead"].isin(LEADS)) & (df["grupo"] == "todos")]
    piv = df.pivot(index=["posto", "lead"], columns="modelo", values="NSE").reset_index()
    piv = piv.rename(columns={"LSTM": "NSE LSTM", "persistencia": "NSE persistência",
                              "climatologia": "NSE climatologia"})
    piv = piv[["posto", "lead", "NSE LSTM", "NSE persistência", "NSE climatologia"]]
    piv = piv.sort_values(["posto", "lead"]).round(3)
    return para_markdown(piv)


def tabela_faixa_skill_teste() -> str:
    """Faixa de leads em que a LSTM supera as DUAS referências, no teste final."""
    df = pd.read_csv(TAB / "lstm_teste_skill_score.csv")
    linhas = []
    for p in ["ivr", "gap", "fsb"]:
        sub = df[df["posto"] == p.upper()]
        leads_ok = sub.loc[sub["supera_as_duas"], "lead"].tolist()
        linhas.append({"Posto": p.upper(), "Leads em que a LSTM supera as duas referências": faixa_leads(leads_ok),
                       "Total": f"{len(leads_ok)}/30"})
    return para_markdown(pd.DataFrame(linhas))


def tabela_repro_zampieri() -> str:
    """NSE médio entre sementes, reprodução de Zampieri (2025), LSTM e persistência."""
    df = pd.read_csv(TAB / "repro_zampieri.csv")
    df = df[df["modelo"].isin(["LSTM", "persistencia"])]
    res = df.groupby(["posto", "versao", "modelo"])[["NSE"]].mean().reset_index()
    piv = res.pivot(index=["posto", "versao"], columns="modelo", values="NSE").reset_index()
    piv = piv.rename(columns={"LSTM": "NSE LSTM", "persistencia": "NSE persistência"})
    piv = piv.sort_values(["posto", "versao"]).round(3)
    return para_markdown(piv)


def tabela_postos() -> str:
    """Postos: código ANA, rio, área, período com dado (configs/postos.yaml + resumo_postos.csv)."""
    cfg = yaml.safe_load(Path("configs/postos.yaml").read_text(encoding="utf-8"))
    resumo = pd.read_csv(TAB / "resumo_postos.csv").set_index("posto")
    linhas = []
    for p, c in cfg.items():
        r = resumo.loc[p.upper()]
        linhas.append({"Posto": p.upper(), "Código ANA": c["codigo"], "Rio": c["rio"],
                       "Área (km²)": c["area_km2"], "Dado de": r["inicio"][:4], "até": r["fim"][:4]})
    return para_markdown(pd.DataFrame(linhas))


def main():
    for titulo, fn in [
        ("Teste final (2015-2019) — NSE, leads 1/7/30", tabela_teste),
        ("Teste final — faixa de leads em que a LSTM supera as duas referências", tabela_faixa_skill_teste),
        ("Reprodução de Zampieri (2025) — NSE médio entre sementes", tabela_repro_zampieri),
        ("Postos", tabela_postos),
    ]:
        print(f"### {titulo}\n")
        print(fn())
        print()


if __name__ == "__main__":
    main()
