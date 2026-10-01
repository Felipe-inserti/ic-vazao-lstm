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


ARQ_TABELA = {"validacao": "lstm_validacao.csv", "teste": "lstm_teste.csv"}
ARQ_SKILL = {"validacao": "lstm_skill_score.csv", "teste": "lstm_teste_skill_score.csv"}


def tabela_resumo(fonte: str) -> str:
    """NSE da LSTM, persistência e climatologia, leads 1/7/30. fonte: "validacao" ou "teste"."""
    df = pd.read_csv(TAB / ARQ_TABELA[fonte])
    df = df[(df["lead"].isin(LEADS)) & (df["grupo"] == "todos")]
    piv = df.pivot(index=["posto", "lead"], columns="modelo", values="NSE").reset_index()
    piv = piv.rename(columns={"LSTM": "NSE LSTM", "persistencia": "NSE persistência",
                              "climatologia": "NSE climatologia"})
    piv = piv[["posto", "lead", "NSE LSTM", "NSE persistência", "NSE climatologia"]]
    piv = piv.sort_values(["posto", "lead"]).round(3)
    return para_markdown(piv)


def tabela_faixa_skill(fonte: str) -> str:
    """Faixa de leads em que a LSTM supera as DUAS referências. fonte: "validacao" ou "teste"."""
    df = pd.read_csv(TAB / ARQ_SKILL[fonte])
    linhas = []
    for p in ["ivr", "gap", "fsb"]:
        sub = df[df["posto"] == p.upper()]
        leads_ok = sub.loc[sub["supera_as_duas"], "lead"].tolist()
        linhas.append({"Posto": p.upper(), "Faixa de leads": faixa_leads(leads_ok),
                       "Total": f"{len(leads_ok)}/30"})
    return para_markdown(pd.DataFrame(linhas))


def tabela_comparacao_val_teste() -> str:
    """NSE lado a lado, validação x teste, leads 1/7/30, grupo "todos"."""
    val = pd.read_csv(TAB / ARQ_TABELA["validacao"]).query("lead in @LEADS and grupo == 'todos'")
    tst = pd.read_csv(TAB / ARQ_TABELA["teste"]).query("lead in @LEADS and grupo == 'todos'")
    comb = pd.concat([val.assign(fase="validacao"), tst.assign(fase="teste")], ignore_index=True)
    piv = comb.pivot_table(index=["posto", "lead", "modelo"], columns="fase", values="NSE").reset_index()
    piv = piv[["posto", "lead", "modelo", "validacao", "teste"]]
    piv["diferença"] = piv["teste"] - piv["validacao"]
    piv = piv.sort_values(["posto", "lead", "modelo"]).round(3)
    return para_markdown(piv)


def tabela_ivr_parte1(fonte: str) -> str:
    """NSE e skill score vs. persistência — IVR, GAP, FSB, leads 1/7/30 (H-IVR, parte 1)."""
    df = pd.read_csv(TAB / ARQ_SKILL[fonte])
    df = df[df["lead"].isin(LEADS)][["posto", "lead", "NSE_LSTM", "skill_vs_persistencia"]]
    df = df.rename(columns={"NSE_LSTM": "NSE", "skill_vs_persistencia": "skill vs. persistência"})
    return para_markdown(df.sort_values(["lead", "posto"]).round(3))


def tabela_ivr_parte2(fonte: str) -> str:
    """NSE e skill score (dentro do subconjunto) do IVR, suspeito x não-suspeito, leads 1/7/30
    (H-IVR, parte 2). Skill recomputado com `habilidade()`, reaproveitado de
    src/agregar_validacao.py — não duplica a lógica."""
    from src.agregar_validacao import habilidade
    df = pd.read_csv(TAB / ARQ_TABELA[fonte])
    sub = df[(df["posto"] == "IVR") & (df["grupo"] != "todos")]
    linhas = []
    for grupo in ["suspeito", "nao_suspeito"]:
        g = sub[sub["grupo"] == grupo]
        piv = habilidade(g, ["lead"])
        piv.insert(0, "grupo", grupo)
        linhas.append(piv[["grupo", "lead", "NSE_LSTM", "skill_vs_persistencia"]])
    out = pd.concat(linhas, ignore_index=True).rename(
        columns={"NSE_LSTM": "NSE", "skill_vs_persistencia": "skill vs. persistência"})
    return para_markdown(out.sort_values(["lead", "grupo"]).round(3))


def tabela_dias_ano_ivr(fonte: str) -> str:
    """Dias por ano em cada subconjunto (suspeito/não-suspeito) do IVR. Recomputado a partir
    das previsões salvas (results/previsoes/) + data/processed/ivr.parquet — reaproveita
    src/agregar_validacao.py inteiro, sem duplicar a lógica de marcação de suspeito."""
    from src.agregar_validacao import (CONFIG_MODO, carregar_previsoes, carregar_referencias,
                                       dobras_disponiveis, quebra_ivr_suspeito)
    df_proc = pd.read_parquet("data/processed/ivr.parquet")
    dobras = dobras_disponiveis("ivr", CONFIG_MODO[fonte]["padrao_dobra"])
    previsoes = carregar_previsoes("ivr", dobras)
    referencias = carregar_referencias(df_proc, "ivr", dobras)
    _, contagem = quebra_ivr_suspeito(df_proc, previsoes, referencias)
    piv = contagem.pivot(index="ano", columns="grupo", values="dias").fillna(0).astype(int).reset_index()
    return para_markdown(piv)


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
    secoes = [
        ("Postos", lambda: tabela_postos()),
        ("Reprodução de Zampieri (2025) — NSE médio entre sementes", lambda: tabela_repro_zampieri()),
        ("Validação walk-forward (2005-2014) — NSE, leads 1/7/30", lambda: tabela_resumo("validacao")),
        ("Validação — faixa de leads em que a LSTM supera as duas referências", lambda: tabela_faixa_skill("validacao")),
        ("Teste final (2015-2019) — NSE, leads 1/7/30", lambda: tabela_resumo("teste")),
        ("Teste — faixa de leads em que a LSTM supera as duas referências", lambda: tabela_faixa_skill("teste")),
        ("Validação x teste, lado a lado — NSE, leads 1/7/30", lambda: tabela_comparacao_val_teste()),
        ("H-IVR parte 1 — validação: NSE e skill vs. persistência", lambda: tabela_ivr_parte1("validacao")),
        ("H-IVR parte 1 — teste: NSE e skill vs. persistência", lambda: tabela_ivr_parte1("teste")),
        ("H-IVR parte 2 — validação: IVR suspeito x não-suspeito", lambda: tabela_ivr_parte2("validacao")),
        ("H-IVR parte 2 — teste: IVR suspeito x não-suspeito", lambda: tabela_ivr_parte2("teste")),
        ("IVR — dias por ano em cada subconjunto, validação", lambda: tabela_dias_ano_ivr("validacao")),
        ("IVR — dias por ano em cada subconjunto, teste", lambda: tabela_dias_ano_ivr("teste")),
    ]
    for titulo, fn in secoes:
        print(f"### {titulo}\n")
        print(fn())
        print()


if __name__ == "__main__":
    main()
