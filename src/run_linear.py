"""Execução explícita dos modelos lineares com validação externa agrupada congelada."""
from __future__ import annotations
import argparse, json, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score
from src.config import PATHS, SEED
from src import avaliacao as av


def make_model(name, p, seed):
    if name == 'logistic_l2': clf=LogisticRegression(penalty='l2',solver='liblinear',max_iter=3000,random_state=seed,**p)
    elif name == 'ridge': clf=RidgeClassifier(solver='lsqr', tol=1e-4, max_iter=2000, **p)
    elif name == 'linear_svm': clf=LinearSVC(dual='auto',max_iter=10000,random_state=seed,**p)
    elif name == 'elastic_net': clf=LogisticRegression(penalty='elasticnet',solver='saga',max_iter=1200,tol=5e-3,random_state=seed,**p)
    else: raise ValueError(name)
    return Pipeline([('scale',StandardScaler()),('clf',clf)])

def score(m,X):
    if hasattr(m,'predict_proba'): return m.predict_proba(X)[:,1]
    z=m.decision_function(X)
    return 1/(1+np.exp(-np.clip(z,-40,40)))

def grids(quick=False):
    cs=[1.] if quick else [.01,.1,1.,10.]
    cw=['balanced'] if quick else [None,'balanced']
    return {'logistic_l2':[dict(C=c,class_weight=w) for c in cs for w in cw],
      'ridge':[dict(alpha=a,class_weight=w) for a in ([1.] if quick else [.01,.1,1.,10.,100.]) for w in cw],
      'linear_svm':[dict(C=c,class_weight=w) for c in cs for w in cw],
      'elastic_net':[dict(C=c,l1_ratio=r,class_weight=w) for c in ([1.] if quick else [.1,1.]) for r in ([.5] if quick else [.1,.9]) for w in cw]}

# Full Elastic Net grid: 2 C x 2 L1 ratios x 2 class weights = 8 configurations,
# each evaluated on all 3 grouped inner splits. The iterative solver can still be slow.

def run(quick=False):
    d=av.carregar_dataset(PATHS['processed']/'dataset_abide_processado.pkl')
    # Legacy PKL contains folds but loader reconstructs fold_id. Persisted folds are unchanged.
    run_id=('quick_smoke' if quick else 'full') + '_' + time.strftime('%Y%m%d_%H%M%S')
    model_grids=grids(quick)
    if quick: model_grids={'logistic_l2': model_grids['logistic_l2']}
    out=Path(__file__).resolve().parents[1]/'experimentos/01_modelos_lineares/Results'
    out.mkdir(parents=True,exist_ok=True)
    cfg={'experiment':'01_modelos_lineares','seed':SEED,'quick':quick,'models':list(model_grids), 'quick_scope':'one outer fold, one estimator' if quick else 'all folds and estimators',
         'outer_protocol':'stored folds, grouped by SITE_ID','threshold':'0.5 probability / zero decision score',
         'dataset_shape':list(d['X'].shape),'class_counts':{str(k):int(v) for k,v in zip(*np.unique(d['y'],return_counts=True))},
         'hyperparameter_grid':grids(quick), 'elastic_net_solver':{'solver':'saga','max_iter':1200,'tol':5e-3},
         'ridge_solver':{'solver':'lsqr','max_iter':2000,'tol':1e-4}}
    (out/f'01_linear_{run_id}_config.json').write_text(json.dumps(cfg,indent=2))
    rows=[]; oof={n:np.full(len(d['y']),np.nan) for n in model_grids}; preds={n:np.full(len(d['y']),-1,dtype=np.int8) for n in model_grids}
    folds=sorted(np.unique(d['fold_id']))
    if quick: folds=folds[:1]
    print(f"Execução {'rápida' if quick else 'completa'}: {len(folds)} folds; modelos: {', '.join(model_grids)}", flush=True)
    for fold in folds:
      fold_start=time.perf_counter()
      te=np.flatnonzero(d['fold_id']==fold); tr=np.flatnonzero(d['fold_id']!=fold)
      splits,info=av.splits_internos(d['y'][tr],d['sites'][tr],k_max=2 if quick else 3,seed=SEED)
      if not info['viavel']:
        print(f"Fold {fold}: não há validação interna agrupada viável.", flush=True)
        rows.append({'fold':int(fold),'status':'sem_split_interno_viavel'}); continue
      print(f"Fold {fold}: treino={len(tr)}, teste={len(te)}, splits internos={info['k_usado']}", flush=True)
      for name,params in model_grids.items():
        model_start=time.perf_counter(); fit_warnings=[]
        print(f"  {name}: busca interna com {len(params)} configurações...", flush=True)
        results=[]
        for p in params:
          auc=[]
          for a,b in splits:
            m=make_model(name,p,SEED)
            with warnings.catch_warnings(record=True) as inner_warnings:
              warnings.simplefilter('always'); m.fit(d['X'][tr[a]],d['y'][tr[a]])
            fit_warnings.extend(str(w.message) for w in inner_warnings)
            auc.append(roc_auc_score(d['y'][tr[b]],score(m,d['X'][tr[b]])))
          results.append((float(np.mean(auc)),p,float(np.std(auc))))
        tempo_busca=time.perf_counter()-model_start
        best=max(results,key=lambda q:q[0]); m=make_model(name,best[1],SEED)
        print(f"  {name}: melhor AUC interna={best[0]:.3f}; ajustando treino externo...", flush=True)
        t=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
          warnings.simplefilter('always'); m.fit(d['X'][tr],d['y'][tr])
        fit_warnings.extend(str(w.message) for w in caught)
        elapsed=time.perf_counter()-t; p=score(m,d['X'][te]); pr=(p>=.5).astype(int)
        oof[name][te]=p;preds[name][te]=pr
        rows.append({'modelo':name,'tipo':'externo','fold':int(fold),'seed':SEED,'n_teste':len(te),
          **av.metricas(d['y'][te],pr,p),'auc_interno_media':best[0],'auc_interno_dp':best[2],
          'parametros':json.dumps(best[1],sort_keys=True),'k_interno':info['k_usado'],
          'tempo_busca_s':round(tempo_busca,1),'tempo_treino_s':round(elapsed,1),'avisos_convergencia':' | '.join(fit_warnings)})
        print(f"  {name}: AUC externa={rows[-1]['auc_roc']:.3f}, busca={tempo_busca/60:.1f} min, ajuste={elapsed/60:.1f} min.", flush=True)
      pd.DataFrame(rows).to_csv(out/f'01_linear_{run_id}_metrics_partial.csv',index=False)
      print(f"Fold {fold} concluído em {(time.perf_counter()-fold_start)/60:.1f} min; parcial gravado.", flush=True)
    df=pd.DataFrame(rows); df.to_csv(out/f'01_linear_{run_id}_metrics.csv',index=False)
    print(f"Execução finalizada. Resultados: {out}", flush=True)
    if not quick and all(np.isfinite(v).all() for v in oof.values()):
      av.salvar_oof(out/f'01_linear_{run_id}_oof.npz',d,oof,preds)
    return df

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true',help='teste de fumaça: um fold, Logistic Regression L2 e validação interna agrupada')
    a=ap.parse_args();print(run(a.quick).to_string(index=False))
