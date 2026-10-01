"""Agrega a validação walk-forward (10 dobras x 3 sementes) OU o teste final (dobra única,
2015-2019 — ver docs/decisoes.md, 2026-10-01, PRÉ-REGISTRO), alvo "delta", por posto e lead.

Lê results/previsoes/lstm_<posto>_<dobra>_delta_s<semente>.csv (já gerados por
src/treinar_lstm.py; dobras puladas simplesmente não têm arquivo), junta as dobras
disponíveis de cada posto e calcula NSE/KGE/PBIAS/RMSE/R² por lead (1 a 30), com média e
desvio entre as 3 sementes. Persistência e climatologia são recalculadas nos MESMOS dias via
src/models/baselines.referencia_dobra (não depende de TensorFlow, roda em .venv).

IVR é reportado SEPARADAMENTE (nunca entra em médias com GAP/FSB — pré-registro de
docs/decisoes.md) e ganha uma quebra adicional suspeito vs. não-suspeito (hipótese H-IVR),
juntando as dobras, nos leads 1/7/30 — com persistência/climatologia e skill score DENTRO de
cada subconjunto (NSE puro não é comparável entre subséries de variância diferente).

Skill score: 1 - MSE_LSTM / MSE_referência, sempre contra a persistência e também contra a
MELHOR referência (maior NSE entre persistência e climatologia) quando há as duas.

Uso:
  python -m src.agregar_validacao                   # validação (padrão)
  python -m src.agregar_validacao --dobra teste      # teste final (2015-2019)
  python -m src.agregar_validacao --dobra comparar   # tabela lado a lado (exige os dois acima)

Saídas (nomes trocam "validacao"/08/09 por "teste"/10/11 conforme --dobra):
  results/tabelas/lstm_validacao.csv            posto, lead, grupo, modelo, NSE, KGE, PBIAS,
                                                 RMSE, R2 (+ _desvio pra LSTM, entre sementes)
  results/tabelas/lstm_validacao_por_dobra.csv  idem, por posto+dobra (só no modo validação)
  results/tabelas/lstm_skill_score.csv          posto, lead, NSE/RMSE de cada modelo, skill
                                                 score, melhor referência, se a LSTM supera as duas
  results/figuras/08_lstm_validacao_nse.png     NSE x lead, LSTM (média+faixa) x persistência x
                                                 climatologia, 1 painel por posto
  results/figuras/09_lstm_skill_score.png       skill score x lead (vs. melhor referência),
                                                 1 painel por posto, linha zero
  results/tabelas/lstm_teste.csv, lstm_teste_skill_score.csv,
  results/figuras/10_lstm_teste_nse.png, 11_lstm_teste_skill_score.png   idem, teste final
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.eval import metricas
from src.models import baselines

PREV = Path("results/previsoes")
PROC = Path("data/processed")
TAB = Path("results/tabelas")
FIG = Path("results/figuras")
ALVO = "delta"
SEMENTES = [42, 43, 44]
POSTOS = ["ivr", "gap", "fsb"]
LEADS_RESUMO = [1, 7, 30]
COR = {"ivr": "#2a78d6", "gap": "#eb6834", "fsb": "#1baf7a"}


def dobras_disponiveis(posto: str, padrao: str = "val*") -> list[str]:
    """Dobras com previsão salva pra TODAS as sementes (as que não foram puladas). `padrao` é
    o glob do nome da dobra: "val*" pra validação, "teste" pro teste final (dobra única)."""
    por_semente = [{a.stem.split("_")[2] for a in PREV.glob(f"lstm_{posto}_{padrao}_{ALVO}_s{s}.csv")}
                  for s in SEMENTES]
    comum = set.intersection(*por_semente) if por_semente else set()
    faltando = set.union(*por_semente) - comum if por_semente else set()
    if faltando:
        print(f"[aviso] {posto.upper()}: dobra(s) {sorted(faltando)} não têm previsão em "
              f"todas as sementes — ignoradas")
    return sorted(comum)


def carregar_previsoes(posto: str, dobras: list[str]) -> pd.DataFrame:
    """Todas as previsões da LSTM do posto (todas as dobras, todas as sementes), formato
    longo: posto, dobra, semente, data, lead, obs, lstm."""
    partes = []
    for dobra in dobras:
        for semente in SEMENTES:
            d = pd.read_csv(PREV / f"lstm_{posto}_{dobra}_{ALVO}_s{semente}.csv", parse_dates=["data"])
            d["posto"], d["dobra"], d["semente"] = posto, dobra, semente
            partes.append(d)
    return pd.concat(partes, ignore_index=True)


def carregar_referencias(df_proc: pd.DataFrame, posto: str, dobras: list[str]) -> pd.DataFrame:
    """Persistência e climatologia de todas as dobras do posto, formato longo: dobra, data,
    lead, modelo, previsto (SEM obs — é anexado depois, a partir das previsões da LSTM, que já
    carregam a vazão observada nos mesmos dias)."""
    partes = []
    for dobra in dobras:
        ref = baselines.referencia_dobra(df_proc, dobra)
        h = ref["persistencia"].shape[1]
        n = len(ref["datas_val"])
        for modelo in ["persistencia", "climatologia"]:
            partes.append(pd.DataFrame({
                "dobra": dobra,
                "data": np.repeat(ref["datas_val"], h),
                "lead": np.tile(np.arange(1, h + 1), n),
                "modelo": modelo,
                "previsto": ref[modelo].ravel(),
            }))
    return pd.concat(partes, ignore_index=True)


def metricas_por_grupo(df: pd.DataFrame, grupos: list[str], leads=range(1, 31)) -> pd.DataFrame:
    """`df` precisa ter colunas obs/previsto (+ as colunas de `grupos`, ex. lead, semente,
    modelo). Uma linha de métricas por combinação de `grupos` presente, restrita a `leads`."""
    linhas = []
    for chave, g in df[df["lead"].isin(leads)].groupby(grupos, sort=False):
        chave = chave if isinstance(chave, tuple) else (chave,)
        linhas.append({**dict(zip(grupos, chave)), **metricas.todas(g["obs"], g["previsto"])})
    return pd.DataFrame(linhas)


def media_desvio_entre_sementes(df_metricas: pd.DataFrame, chave: list[str]) -> pd.DataFrame:
    """Recebe uma linha de métrica por (chave + semente); devolve média e desvio entre
    sementes, uma linha por `chave`."""
    cols = ["NSE", "KGE", "PBIAS", "RMSE", "R2"]
    g = df_metricas.groupby(chave)[cols].agg(["mean", "std"])
    g.columns = [f"{c}" if estat == "mean" else f"{c}_desvio" for c, estat in g.columns]
    return g.reset_index()


def habilidade(tab: pd.DataFrame, chave: list[str]) -> pd.DataFrame:
    """A partir de uma tabela longa (colunas `chave` + modelo + NSE + RMSE; modelo em
    {LSTM, persistencia, climatologia}), calcula o skill score da LSTM: 1 - MSE_lstm/MSE_ref.
    Sempre contra a persistência; também contra a MELHOR referência (maior NSE entre
    persistência e climatologia), quando há climatologia no grupo — é essa comparação que
    importa pra saber se a LSTM vale o treino: perder pra persistência mas ganhar da
    climatologia ainda não é suficiente.
    """
    piv = tab.pivot_table(index=chave, columns="modelo", values=["NSE", "RMSE"])
    piv.columns = [f"{c}_{m}" for c, m in piv.columns]
    piv = piv.reset_index()

    mse_lstm = piv["RMSE_LSTM"] ** 2
    mse_pers = piv["RMSE_persistencia"] ** 2
    piv["skill_vs_persistencia"] = 1 - mse_lstm / mse_pers

    if "NSE_climatologia" in piv.columns:
        pers_melhor = piv["NSE_persistencia"] >= piv["NSE_climatologia"]
        piv["modelo_melhor_referencia"] = np.where(pers_melhor, "persistencia", "climatologia")
        piv["NSE_melhor_referencia"] = np.where(pers_melhor, piv["NSE_persistencia"], piv["NSE_climatologia"])
        mse_melhor = np.where(pers_melhor, mse_pers, piv["RMSE_climatologia"] ** 2)
        piv["skill_vs_melhor_referencia"] = 1 - mse_lstm / mse_melhor
        piv["supera_as_duas"] = (piv["NSE_LSTM"] > piv["NSE_persistencia"]) & \
                                (piv["NSE_LSTM"] > piv["NSE_climatologia"])
    return piv


def faixa_leads(leads) -> str:
    """Descreve um conjunto de leads como faixa(s) contígua(s), ex.: '3-9, 14-30'."""
    leads = sorted(leads)
    if not leads:
        return "nenhum"
    faixas, ini = [], leads[0]
    for a, b in zip(leads, leads[1:] + [None]):
        if b is None or b != a + 1:
            faixas.append(f"{ini}" if ini == a else f"{ini}-{a}")
            ini = b
    return ", ".join(faixas)


def tabela_posto(posto: str, previsoes: pd.DataFrame, referencias: pd.DataFrame, pool_dobras: bool):
    """Monta a tabela de métricas de um posto: LSTM (média+desvio entre sementes) e
    persistência/climatologia (determinísticas), nos leads 1-30.
    `pool_dobras=True` junta todas as dobras antes de calcular (tabela agregada);
    `pool_dobras=False` mantém a dobra como mais um agrupador (tabela por dobra)."""
    grupos_lstm = (["lead", "semente"] if pool_dobras else ["dobra", "lead", "semente"])
    m_lstm = metricas_por_grupo(previsoes.rename(columns={"lstm": "previsto"}), grupos_lstm)
    chave_agg = [g for g in grupos_lstm if g != "semente"]
    agg_lstm = media_desvio_entre_sementes(m_lstm, chave_agg)
    agg_lstm.insert(len(chave_agg), "modelo", "LSTM")

    ref_com_obs = referencias.merge(
        previsoes[["dobra", "data", "lead", "obs"]].drop_duplicates(), on=["dobra", "data", "lead"], how="left")
    assert ref_com_obs["obs"].notna().all(), "referência sem obs correspondente — dobra/data/lead não bateram"
    grupos_ref = (["lead", "modelo"] if pool_dobras else ["dobra", "lead", "modelo"])
    m_ref = metricas_por_grupo(ref_com_obs, grupos_ref)

    tab = pd.concat([agg_lstm, m_ref], ignore_index=True)
    tab.insert(0, "posto", posto.upper())
    tab.insert(tab.columns.get_loc("lead") + 1, "grupo", "todos")
    return tab


def quebra_ivr_suspeito(df_proc: pd.DataFrame, previsoes: pd.DataFrame, referencias: pd.DataFrame):
    """IVR nos leads de resumo, separado por suspeito/não-suspeito (hipótese H-IVR), juntando
    todas as dobras: LSTM (média+desvio entre sementes) E persistência/climatologia, DENTRO de
    cada subconjunto — comparar NSE entre suspeito e não-suspeito direto não é justo (são
    subséries com variâncias diferentes, e o NSE é normalizado pela variância do próprio
    subconjunto); o skill score da LSTM contra a referência DO MESMO subconjunto é que permite
    comparar. Devolve (tabela de métricas, contagem de dias por ano em cada subconjunto)."""
    suspeito_por_data = df_proc.set_index("data")["suspeito"]
    p = previsoes.rename(columns={"lstm": "previsto"}).copy()
    p["suspeito"] = p["data"].map(suspeito_por_data)
    ref = referencias.merge(previsoes[["dobra", "data", "lead", "obs"]].drop_duplicates(),
                            on=["dobra", "data", "lead"], how="left")
    ref["suspeito"] = ref["data"].map(suspeito_por_data)

    linhas = []
    for grupo, susp in [("suspeito", True), ("nao_suspeito", False)]:
        sub_p, sub_ref = p[p["suspeito"] == susp], ref[ref["suspeito"] == susp]
        if sub_p.empty:
            continue
        m = metricas_por_grupo(sub_p, ["lead", "semente"], leads=LEADS_RESUMO)
        agg = media_desvio_entre_sementes(m, ["lead"])
        agg.insert(0, "posto", "IVR")
        agg.insert(2, "grupo", grupo)
        agg.insert(3, "modelo", "LSTM")
        linhas.append(agg)

        m_ref = metricas_por_grupo(sub_ref, ["lead", "modelo"], leads=LEADS_RESUMO)
        m_ref.insert(0, "posto", "IVR")
        m_ref.insert(2, "grupo", grupo)
        linhas.append(m_ref)

    tab = pd.concat(linhas, ignore_index=True)

    # dias por ano de emissão (um por data, não por lead/semente), nos DOIS subconjuntos
    dias_unicos = p[(p["lead"] == 1) & (p["semente"] == SEMENTES[0])].copy()
    dias_unicos["grupo"] = np.where(dias_unicos["suspeito"], "suspeito", "nao_suspeito")
    dias_unicos["ano"] = dias_unicos["data"].dt.year
    contagem_anos = dias_unicos.groupby(["grupo", "ano"]).size().rename("dias").reset_index()
    return tab, contagem_anos


def figura(tabelas: dict, arquivo: str, rotulo_y: str):
    postos = [p for p in POSTOS if p in tabelas]
    fig, axs = plt.subplots(1, len(postos), figsize=(4.2 * len(postos), 3.4), sharey=True)
    estilo = {"persistencia": dict(color="#52514e", linestyle="--", label="persistência"),
              "climatologia": dict(color="#a3a29d", linestyle=":", label="climatologia")}
    for ax, p in zip(np.atleast_1d(axs), postos):
        t = tabelas[p]
        lstm = t[(t["modelo"] == "LSTM") & (t["grupo"] == "todos")].sort_values("lead")
        ax.plot(lstm["lead"], lstm["NSE"], color=COR[p], linewidth=2, label="LSTM (média)")
        ax.fill_between(lstm["lead"], lstm["NSE"] - lstm["NSE_desvio"], lstm["NSE"] + lstm["NSE_desvio"],
                        color=COR[p], alpha=0.2, linewidth=0)
        for modelo, st in estilo.items():
            r = t[(t["modelo"] == modelo) & (t["grupo"] == "todos")].sort_values("lead")
            ax.plot(r["lead"], r["NSE"], linewidth=1.6, **st)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(p.upper(), loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("antecedência (dias)")
        ax.grid(color="#e6e5e1", linewidth=0.6)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel(rotulo_y)
    np.atleast_1d(axs)[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / arquivo, dpi=200)
    plt.close(fig)


def figura_skill(skill_por_posto: dict, arquivo: str, titulo: str):
    postos = [p for p in POSTOS if p in skill_por_posto]
    fig, axs = plt.subplots(1, len(postos), figsize=(4.2 * len(postos), 3.0), sharey=True)
    for ax, p in zip(np.atleast_1d(axs), postos):
        s = skill_por_posto[p].sort_values("lead")
        ax.plot(s["lead"], s["skill_vs_melhor_referencia"], color=COR[p], linewidth=2)
        ax.axhline(0, color="#52514e", linewidth=0.8)
        ax.set_title(p.upper(), loc="left", fontweight="bold", fontsize=10)
        ax.set_xlabel("antecedência (dias)")
        ax.grid(color="#e6e5e1", linewidth=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    np.atleast_1d(axs)[0].set_ylabel("skill score (1 - MSE/MSE melhor ref.)")
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(FIG / arquivo, dpi=200)
    plt.close(fig)


CONFIG_MODO = {
    "validacao": dict(
        padrao_dobra="val*", titulo="validação walk-forward (10 dobras)",
        arquivo_tabela="lstm_validacao.csv", arquivo_por_dobra="lstm_validacao_por_dobra.csv",
        arquivo_skill="lstm_skill_score.csv",
        figura_nse="08_lstm_validacao_nse.png", figura_skill="09_lstm_skill_score.png"),
    "teste": dict(
        padrao_dobra="teste", titulo="teste final (2015-2019)",
        arquivo_tabela="lstm_teste.csv", arquivo_por_dobra=None,
        arquivo_skill="lstm_teste_skill_score.csv",
        figura_nse="10_lstm_teste_nse.png", figura_skill="11_lstm_teste_skill_score.png"),
}


def agregar(modo: str):
    """Monta as mesmas saídas (tabela agregada, skill score, figuras) pra "validacao" (10
    dobras walk-forward) ou "teste" (a dobra única 2015-2019, avaliação final — ver
    docs/decisoes.md). IVR sempre reportado separado, com a quebra suspeito/não-suspeito."""
    cfg = CONFIG_MODO[modo]
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    tabelas_agregadas, tabelas_por_dobra, extras_ivr = {}, [], []
    contagem_anos_ivr = None
    for posto in POSTOS:
        dobras = dobras_disponiveis(posto, cfg["padrao_dobra"])
        if not dobras:
            print(f"[sem previsões] {posto.upper()}: rode src/treinar_lstm.py --dobra "
                  f"{'<val...>' if modo == 'validacao' else 'teste'} primeiro")
            continue
        df_proc = pd.read_parquet(PROC / f"{posto}.parquet")
        previsoes = carregar_previsoes(posto, dobras)
        referencias = carregar_referencias(df_proc, posto, dobras)

        tabelas_agregadas[posto] = tabela_posto(posto, previsoes, referencias, pool_dobras=True)
        if cfg["arquivo_por_dobra"]:
            tabelas_por_dobra.append(tabela_posto(posto, previsoes, referencias, pool_dobras=False))

        if posto == "ivr":
            tab_ivr, contagem_anos_ivr = quebra_ivr_suspeito(df_proc, previsoes, referencias)
            extras_ivr.append(tab_ivr)

    agregada = pd.concat(list(tabelas_agregadas.values()) + extras_ivr, ignore_index=True)

    col_redonda_3 = ["NSE", "KGE", "R2", "NSE_desvio", "KGE_desvio", "R2_desvio"]
    col_redonda_1 = ["PBIAS", "RMSE", "PBIAS_desvio", "RMSE_desvio"]
    tabelas_para_arredondar = [agregada]
    if tabelas_por_dobra:
        por_dobra = pd.concat(tabelas_por_dobra, ignore_index=True)
        tabelas_para_arredondar.append(por_dobra)
    for tab in tabelas_para_arredondar:
        tab[col_redonda_3] = tab[col_redonda_3].round(3)
        tab[col_redonda_1] = tab[col_redonda_1].round(1)

    agregada.to_csv(TAB / cfg["arquivo_tabela"], index=False)
    if tabelas_por_dobra:
        por_dobra.to_csv(TAB / cfg["arquivo_por_dobra"], index=False)
    figura(tabelas_agregadas, cfg["figura_nse"], f"NSE ({cfg['titulo']})")

    # --- habilidade (skill score) por posto e lead, grupo "todos" ---
    skill_por_posto = {}
    for posto, t in tabelas_agregadas.items():
        piv = habilidade(t[t["grupo"] == "todos"], ["lead"])
        piv.insert(0, "posto", posto.upper())
        skill_por_posto[posto] = piv
    skill_tab = pd.concat(skill_por_posto.values(), ignore_index=True)
    col_skill_3 = [c for c in skill_tab.columns if c.startswith(("NSE_", "skill_"))]
    col_skill_1 = [c for c in skill_tab.columns if c.startswith("RMSE_")]
    skill_tab[col_skill_3] = skill_tab[col_skill_3].round(3)
    skill_tab[col_skill_1] = skill_tab[col_skill_1].round(1)
    skill_tab.to_csv(TAB / cfg["arquivo_skill"], index=False)
    figura_skill(skill_por_posto, cfg["figura_skill"],
                 f"Habilidade da LSTM sobre a melhor referência, {cfg['titulo']} (> 0 = LSTM melhor)")

    print(f"=== resumo ({cfg['titulo']}; leads 1, 7, 30; grupo=todos) ===")
    resumo = agregada[(agregada["lead"].isin(LEADS_RESUMO)) & (agregada["grupo"] == "todos")]
    with pd.option_context("display.width", 200):
        print(resumo.sort_values(["posto", "lead", "modelo"]).to_string(index=False))

    print("\n=== faixa de leads em que a LSTM supera as DUAS referências (persistência E climatologia) ===")
    for posto, piv in skill_por_posto.items():
        leads_ok = piv.loc[piv["supera_as_duas"], "lead"].tolist()
        print(f"{posto.upper()}: {faixa_leads(leads_ok)}  ({len(leads_ok)}/30 leads)")

    print(f"\n=== H-IVR parte 1 ({cfg['titulo']}): NSE pré-registrado x skill score vs. "
          "persistência, leads 1/7/30 ===")
    for posto in ["ivr", "gap", "fsb"]:
        if posto not in skill_por_posto:
            continue
        r = skill_por_posto[posto][skill_por_posto[posto]["lead"].isin(LEADS_RESUMO)]
        print(f"-- {posto.upper()} --")
        print(r[["lead", "NSE_LSTM", "skill_vs_persistencia"]].round(3).to_string(index=False))

    print(f"\n=== IVR ({cfg['titulo']}): suspeito x não-suspeito (leads 1, 7, 30), LSTM + "
          "referências no mesmo subconjunto ===")
    ivr_susp = agregada[(agregada["posto"] == "IVR") & (agregada["grupo"] != "todos")]
    with pd.option_context("display.width", 200):
        print(ivr_susp.sort_values(["grupo", "lead", "modelo"]).to_string(index=False))

    print("\nH-IVR parte 2: skill da LSTM DENTRO de cada subconjunto (compara com a "
          "referência da própria subsérie, não a mistura NSE entre variâncias diferentes):")
    for grupo in ["suspeito", "nao_suspeito"]:
        sub = ivr_susp[ivr_susp["grupo"] == grupo]
        if sub.empty:
            continue
        piv = habilidade(sub, ["lead"])
        cols = ["lead", "skill_vs_persistencia"] + \
            (["skill_vs_melhor_referencia"] if "skill_vs_melhor_referencia" in piv.columns else [])
        print(f"-- IVR {grupo} --")
        print(piv[cols].round(3).to_string(index=False))

    print(f"\ndias por ano em cada subconjunto do IVR, {cfg['titulo']} "
          "(1 por data, não por lead/semente):")
    tab_anos = contagem_anos_ivr.pivot(index="ano", columns="grupo", values="dias").fillna(0).astype(int)
    print(tab_anos.to_string())

    arquivos = f"{TAB}/{cfg['arquivo_tabela']}"
    if cfg["arquivo_por_dobra"]:
        arquivos += f", {TAB}/{cfg['arquivo_por_dobra']}"
    print(f"\ntabelas em {arquivos} e {TAB}/{cfg['arquivo_skill']}")
    print(f"figuras em {FIG}/{cfg['figura_nse']} e {FIG}/{cfg['figura_skill']}")


def comparar():
    """Tabela lado a lado validação x teste, leads 1/7/30, grupo "todos" — exige que
    `agregar("validacao")` e `agregar("teste")` já tenham rodado (lê os CSVs salvos)."""
    arq_val = TAB / CONFIG_MODO["validacao"]["arquivo_tabela"]
    arq_tst = TAB / CONFIG_MODO["teste"]["arquivo_tabela"]
    if not (arq_val.exists() and arq_tst.exists()):
        print(f"[faltando] rode primeiro: python -m src.agregar_validacao --dobra validacao "
              f"e --dobra teste (preciso de {arq_val} e {arq_tst})")
        return

    val = pd.read_csv(arq_val).query("lead in @LEADS_RESUMO and grupo == 'todos'")
    tst = pd.read_csv(arq_tst).query("lead in @LEADS_RESUMO and grupo == 'todos'")
    val, tst = val.assign(fase="validacao"), tst.assign(fase="teste")
    comb = pd.concat([val, tst], ignore_index=True)
    piv = comb.pivot_table(index=["posto", "lead", "modelo"], columns="fase", values="NSE")
    piv = piv.reset_index()[["posto", "lead", "modelo", "validacao", "teste"]]
    piv["diferenca"] = (piv["teste"] - piv["validacao"]).round(3)

    print("=== NSE: validação x teste, leads 1/7/30, grupo=todos ===")
    with pd.option_context("display.width", 160):
        print(piv.sort_values(["posto", "lead", "modelo"]).round(3).to_string(index=False))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dobra", choices=["validacao", "teste", "comparar"], default="validacao",
                    help="validacao (padrão) = agrega as 10 dobras walk-forward; teste = "
                         "agrega só a dobra teste (2015-2019); comparar = tabela lado a lado "
                         "(exige que validacao e teste já tenham rodado)")
    args = ap.parse_args()
    if args.dobra == "comparar":
        comparar()
    else:
        agregar(args.dobra)


if __name__ == "__main__":
    main()
