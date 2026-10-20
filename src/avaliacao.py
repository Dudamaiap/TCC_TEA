"""
avaliacao.py — funções compartilhadas pelos notebooks 08, 09 e 10 (TEA e TDAH).

Onde colocar:
  TEA  -> TCC_TEA_TDAH/src/avaliacao.py      (import: from src import avaliacao as av)
  TDAH -> TCC_TDAH/avaliacao.py (ao lado do config.py)   (import: import avaliacao as av)

Princípios (todos conferem com o plano revisado):
  * os folds externos NÃO são recriados aqui: vêm do notebook 07 de cada base;
  * nenhuma estatística aprendida (scaler, normalização, seleção de atributos,
    hiperparâmetros) usa o fold externo de teste;
  * splits internos são agrupados por site e só são aceitos se tiverem as duas
    classes no treino e na validação; nunca se força um split aleatório;
  * todo resultado vai para uma tabela longa (uma linha por base/modelo/fold/semente).
"""
from __future__ import annotations

import os
import time
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
METRICAS = ["accuracy", "balanced_accuracy", "precision", "recall",
            "specificity", "f1", "auc_roc"]
# TCC_RAPIDO=1 -> versão curta só para conferir se o notebook roda de ponta a ponta.
RAPIDO = os.environ.get("TCC_RAPIDO", "0") == "1"


# --------------------------------------------------------------------------
# 1. Dataset
# --------------------------------------------------------------------------
def carregar_dataset(caminho):
    """Lê o dataset preparado no notebook 07 (TDAH: .npz; TEA: .pkl).

    Retorna dict com X, y, ids, sites, fold_id (0 = controle, 1 = caso).
    """
    caminho = Path(caminho)
    if not caminho.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {caminho}. Rode o notebook 07.")
    if caminho.suffix == ".npz":
        with np.load(caminho, allow_pickle=True) as d:
            obrig = {"X", "y", "ids", "sites", "fold_id"}
            falta = obrig - set(d.files)
            if falta:
                raise KeyError(f"Chaves ausentes em {caminho.name}: {sorted(falta)}")
            out = dict(X=np.asarray(d["X"], dtype=np.float32),
                       y=np.asarray(d["y"]).astype(np.int64).ravel(),
                       ids=np.asarray(d["ids"]).astype(str),
                       sites=np.asarray(d["sites"]).astype(str),
                       fold_id=np.asarray(d["fold_id"]).astype(np.int64).ravel())
    else:
        with open(caminho, "rb") as f:
            d = pickle.load(f)
        n = len(d["y"])
        if d.get("ordem_atributos") != "triu_linha" or not d.get("fisher_z", False):
            raise ValueError("Este .pkl é da versão antiga do notebook 07 (Pearson e/ou ordem de atributos do nilearn). "
                             "Rode o notebook 07 revisado do TEA antes dos notebooks 08, 09 e 10.")
        if "fold_id" in d:
            fold_id = np.asarray(d["fold_id"]).astype(np.int64).ravel()
        else:  # pkl antigo: só havia a lista de (treino, teste)
            fold_id = np.full(n, -1, dtype=np.int64)
            for k, (_, te) in enumerate(d["folds"]):
                fold_id[np.asarray(te)] = k
        out = dict(X=np.asarray(d["X"], dtype=np.float32),
                   y=np.asarray(d["y"]).astype(np.int64).ravel(),
                   ids=np.asarray(d["sub_ids"]).astype(str),
                   sites=np.asarray(d["groups"]).astype(str),
                   fold_id=fold_id)
    checar_dataset(out)
    return out


def n_rois_de(n_features):
    n = int(round((1 + np.sqrt(1 + 8 * n_features)) / 2))
    if n * (n - 1) // 2 != n_features:
        raise ValueError(f"{n_features} atributos não formam um triângulo superior estrito.")
    return n


def checar_dataset(d):
    n = len(d["y"])
    if not (len(d["X"]) == len(d["ids"]) == len(d["sites"]) == len(d["fold_id"]) == n):
        raise ValueError("X, y, ids, sites e fold_id têm tamanhos diferentes.")
    if not np.isfinite(d["X"]).all():
        raise ValueError("X contém NaN ou infinito.")
    if set(np.unique(d["y"]).tolist()) != {0, 1}:
        raise ValueError(f"Esperadas as classes 0 e 1; encontradas {np.unique(d['y'])}.")
    if np.any(d["fold_id"] < 0):
        raise ValueError("Há participantes sem fold externo.")
    n_rois_de(d["X"].shape[1])
    # Fisher z de uma correlação com |r| <= 1-1e-7 fica abaixo de ~8.4; Pearson puro ficaria em [-1, 1].
    # Se todos os |X| <= 1, é quase certo que o Fisher z NÃO foi aplicado.
    if np.abs(d["X"]).max() <= 1.0:
        warnings.warn("Todos os |X| <= 1: parece Pearson puro. Rode o notebook 07 revisado (Fisher z).")


def resumo_folds(d):
    """Tabela de auditoria dos folds externos (tamanho, classes, sites)."""
    linhas = []
    for k in sorted(np.unique(d["fold_id"])):
        te = d["fold_id"] == k
        tr = ~te
        s_te, s_tr = set(d["sites"][te]), set(d["sites"][tr])
        linhas.append(dict(
            fold=int(k), n_treino=int(tr.sum()), n_teste=int(te.sum()),
            controles_teste=int((d["y"][te] == 0).sum()), casos_teste=int((d["y"][te] == 1).sum()),
            controles_treino=int((d["y"][tr] == 0).sum()), casos_treino=int((d["y"][tr] == 1).sum()),
            sites_teste=", ".join(sorted(s_te)), sobreposicao_sites=len(s_te & s_tr),
            duas_classes_no_teste=bool(len(np.unique(d["y"][te])) == 2)))
    df = pd.DataFrame(linhas)
    if (df["sobreposicao_sites"] > 0).any():
        raise RuntimeError("Há site em treino e teste ao mesmo tempo.")
    if not df["duas_classes_no_teste"].all():
        raise RuntimeError("Algum fold de teste tem só uma classe (AUC indefinida).")
    return df


# --------------------------------------------------------------------------
# 2. Splits internos agrupados por site (com checagem de viabilidade)
# --------------------------------------------------------------------------
def splits_internos(y, sites, k_max=3, min_por_classe=5, seed=SEED, tentativas=20):
    """Procura uma partição interna agrupada por site em que TODAS as validações e
    TODOS os treinos tenham as duas classes (e >= min_por_classe de cada na validação).

    Tenta k = k_max, depois k_max-1, ..., 2. Para cada k, testa `tentativas` sementes do
    StratifiedGroupKFold (a semente só muda quais sites caem juntos; os sites nunca são divididos).
    Retorna (lista_de_(tr, va), info). Se nada for viável, (None, info): o chamador deve
    usar hiperparâmetros padrão e registrar isso — não se improvisa split aleatório.
    """
    y, sites = np.asarray(y), np.asarray(sites)
    n_sites = len(np.unique(sites))
    info = dict(k_pedido=k_max, n_sites_treino=n_sites)
    for k in range(min(k_max, n_sites), 1, -1):
        for s in range(tentativas):
            cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed + s)
            try:
                splits = list(cv.split(np.zeros(len(y)), y, groups=sites))
            except ValueError:
                continue
            ok = True
            for tr, va in splits:
                if len(np.unique(y[tr])) < 2 or len(np.unique(y[va])) < 2:
                    ok = False
                    break
                if min((y[va] == 0).sum(), (y[va] == 1).sum()) < min_por_classe:
                    ok = False
                    break
            if ok:
                info.update(k_usado=k, semente_particao=seed + s, viavel=True,
                            tam_validacoes=[len(va) for _, va in splits])
                return splits, info
    info.update(k_usado=0, semente_particao=None, viavel=False, tam_validacoes=[])
    return None, info


def auditoria_splits_internos(d, k_max=3, **kw):
    """Roda `splits_internos` em cada fold externo e devolve a tabela de viabilidade."""
    linhas = []
    for k in sorted(np.unique(d["fold_id"])):
        tr = np.where(d["fold_id"] != k)[0]
        _, info = splits_internos(d["y"][tr], d["sites"][tr], k_max=k_max, **kw)
        linhas.append(dict(fold_externo=int(k), **info))
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------
# 3. Métricas e incerteza
# --------------------------------------------------------------------------
def metricas(y_true, y_pred, y_prob):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) else np.nan,
        f1=f1_score(y_true, y_pred, zero_division=0),
        auc_roc=auc, tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp),
        classe_unica_prevista=bool(len(np.unique(y_pred)) == 1))


def bootstrap_auc_por_site(y, prob, sites, n_boot=2000, seed=SEED):
    """IC 95% da AUC OOF por bootstrap de clusters (sites reamostrados com reposição).
    É um critério definido a priori, não prova de efeito. Com poucos sites o IC é largo."""
    rng = np.random.default_rng(seed)
    y, prob, sites = np.asarray(y), np.asarray(prob), np.asarray(sites)
    nomes = np.unique(sites)
    idx_por_site = {s: np.where(sites == s)[0] for s in nomes}
    aucs = []
    for _ in range(n_boot):
        esc = rng.choice(nomes, size=len(nomes), replace=True)
        idx = np.concatenate([idx_por_site[s] for s in esc])
        if len(np.unique(y[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y[idx], prob[idx]))
    aucs = np.asarray(aucs)
    if len(aucs) < 50:
        return dict(auc=roc_auc_score(y, prob), ic_inf=np.nan, ic_sup=np.nan, n_boot_validos=len(aucs))
    return dict(auc=roc_auc_score(y, prob), ic_inf=float(np.percentile(aucs, 2.5)),
                ic_sup=float(np.percentile(aucs, 97.5)), n_boot_validos=int(len(aucs)))


def tabela_por_site(d, proba, pred):
    """Desempenho por site (OOF). Sites com uma só classe não têm AUC: ficam como NaN."""
    linhas = []
    for s in np.unique(d["sites"]):
        m = d["sites"] == s
        y = d["y"][m]
        linhas.append(dict(site=s, fold=int(np.unique(d["fold_id"][m])[0]), n=int(m.sum()),
                           controles=int((y == 0).sum()), casos=int((y == 1).sum()),
                           acuracia=accuracy_score(y, pred[m]),
                           auc=roc_auc_score(y, proba[m]) if len(np.unique(y)) == 2 else np.nan))
    return pd.DataFrame(linhas)


def resumir_resultados(df_long, d=None, proba_oof=None):
    """Médias por fold (mean/std), média ponderada pelo tamanho do fold e AUC OOF com IC por site."""
    base = df_long[df_long["tipo"] == "externo"]
    resumo = (base.groupby("modelo")[METRICAS].agg(["mean", "std"]).round(4))
    pond = []
    for nome, g in base.groupby("modelo"):
        w = g["n_teste"].to_numpy(dtype=float)
        # média sobre sementes dentro do fold antes de ponderar
        gg = g.groupby("fold").agg({**{m: "mean" for m in METRICAS}, "n_teste": "first"})
        w = gg["n_teste"].to_numpy(dtype=float)
        pond.append({"modelo": nome, **{m + "_pond": float(np.average(gg[m].to_numpy(), weights=w))
                                         for m in METRICAS}})
    out = dict(por_fold=resumo, ponderado=pd.DataFrame(pond).round(4))
    if d is not None and proba_oof is not None:
        linhas = []
        for nome, p in proba_oof.items():
            linhas.append(dict(modelo=nome, **bootstrap_auc_por_site(d["y"], p, d["sites"])))
        out["auc_oof_ic"] = pd.DataFrame(linhas).round(4)
    return out


# --------------------------------------------------------------------------
# 4. Modelos clássicos e grades (exatamente as do plano)
# --------------------------------------------------------------------------
def xgb_device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


GRADES = {
    "SVM_RBF": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.01, 0.001]},
    "Random_Forest": {"n_estimators": [200, 500], "max_depth": [None, 10, 20],
                      "max_features": ["sqrt", "log2"]},
    "XGBoost": {"n_estimators": [100, 300], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
}
# Grade reduzida (só se a completa estourar o tempo; documentar no texto do TCC).
GRADES_REDUZIDAS = {
    "SVM_RBF": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 0.001]},
    "Random_Forest": {"n_estimators": [200], "max_depth": [None, 10], "max_features": ["sqrt"]},
    "XGBoost": {"n_estimators": [100, 300], "max_depth": [3], "learning_rate": [0.01, 0.1]},
}
GRADES_RAPIDAS = {  # só para TCC_RAPIDO=1
    "SVM_RBF": {"clf__C": [1], "clf__gamma": ["scale"]},
    "Random_Forest": {"n_estimators": [50], "max_depth": [None, 10], "max_features": ["sqrt"]},
    "XGBoost": {"n_estimators": [20], "max_depth": [3], "learning_rate": [0.1]},
}


def grade_para(nome, reduzida=False):
    if RAPIDO:
        return GRADES_RAPIDAS[nome]
    return (GRADES_REDUZIDAS if reduzida else GRADES)[nome]


def criar_modelo(nome, y_treino, seed=SEED, params=None):
    """Cria o estimador. `scale_pos_weight` do XGBoost vem só das classes do `y_treino` recebido."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from xgboost import XGBClassifier

    params = params or {}
    y_treino = np.asarray(y_treino)
    if nome == "SVM_RBF":
        m = Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", class_weight="balanced", random_state=seed))])
    elif nome == "Random_Forest":
        m = RandomForestClassifier(class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    elif nome == "XGBoost":
        n_pos, n_neg = int((y_treino == 1).sum()), int((y_treino == 0).sum())
        m = XGBClassifier(objective="binary:logistic", eval_metric="logloss", tree_method="hist",
                          device=xgb_device(), random_state=seed, n_jobs=-1,
                          scale_pos_weight=(n_neg / n_pos) if n_pos else 1.0,
                          subsample=0.8, colsample_bytree=0.8)
    else:
        raise ValueError(nome)
    return m.set_params(**params) if params else m


def pontuar(modelo, X):
    """Probabilidade (ou pseudo-probabilidade monotônica, no SVM) da classe 1."""
    if hasattr(modelo, "predict_proba"):
        return modelo.predict_proba(X)[:, list(modelo.classes_).index(1)]
    s = modelo.decision_function(X)
    return 1.0 / (1.0 + np.exp(-s))


def busca_hiperparametros(nome, X_tr, y_tr, splits, seed=SEED, reduzida=False):
    """GridSearch interno com AUC-ROC e splits agrupados por site já validados.
    Retorna (melhores_params, tabela_cv). O scale_pos_weight do XGBoost é recalculado em cada
    divisão interna (a partir só do treino daquela divisão) por um estimador próprio."""
    from itertools import product
    grade = grade_para(nome, reduzida)
    chaves = list(grade)
    lista_params, linhas = [], []
    for valores in product(*[grade[c] for c in chaves]):
        params = dict(zip(chaves, valores))
        aucs = []
        for tr, va in splits:
            m = criar_modelo(nome, y_tr[tr], seed=seed, params=params)
            m.fit(X_tr[tr], y_tr[tr])
            aucs.append(roc_auc_score(y_tr[va], pontuar(m, X_tr[va])))
        lista_params.append(params)
        linhas.append({**{k: str(v) for k, v in params.items()},
                       "auc_interno_medio": float(np.mean(aucs)), "auc_interno_dp": float(np.std(aucs))})
    i_best = int(np.argmax([l["auc_interno_medio"] for l in linhas]))
    tab = pd.DataFrame(linhas).sort_values("auc_interno_medio", ascending=False).reset_index(drop=True)
    return lista_params[i_best], tab


# --------------------------------------------------------------------------
# 5. Registro
# --------------------------------------------------------------------------
def linha_resultado(base, modelo, fold, semente, y_te, pred, proba, tipo="externo", **extra):
    return dict(base=base, modelo=modelo, tipo=tipo, fold=int(fold), semente=int(semente),
                n_teste=int(len(y_te)), **metricas(y_te, pred, proba), **extra)


def salvar_oof(caminho, d, proba_oof, pred_oof):
    arr = dict(y=d["y"], ids=d["ids"], sites=d["sites"], fold_id=d["fold_id"])
    for nome in proba_oof:
        arr[f"proba_{nome.lower()}"] = proba_oof[nome]
        arr[f"pred_{nome.lower()}"] = pred_oof[nome]
    np.savez_compressed(caminho, **arr)


def agora():
    return time.time()


def filtrar_sites(d, excluir):
    """Remove sites inteiros (análise de sensibilidade). Mantém o fold_id original dos demais.
    `excluir`: nomes (comparação sem diferenciar maiúsculas; aceita trecho do nome)."""
    if not excluir:
        return d
    pats = [str(e).lower() for e in excluir]
    manter = np.array([not any(p in s.lower() for p in pats) for s in d["sites"]])
    if manter.all():
        raise ValueError(f"Nenhum site casou com {excluir}. Sites: {sorted(set(d['sites']))}")
    out = {k: v[manter] for k, v in d.items()}
    removidos = sorted(str(s) for s in set(d["sites"][~manter]))
    print(f"Sites removidos: {removidos} -> {int((~manter).sum())} participantes; restam {int(manter.sum())} "
          f"(controles={int((out['y']==0).sum())}, casos={int((out['y']==1).sum())})")
    return out
