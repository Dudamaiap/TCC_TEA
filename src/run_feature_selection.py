"""Feature selection/reduction nested within grouped CV for ABIDE."""
import json,time,warnings
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src import avaliacao as av
from src.config import PATHS,SEED

def candidates(quick=False):
 elastic_pairs=[(1.,.5)] if quick else [(c,r) for c in [.1,1.,10.] for r in [.1,.5,.9]]
 return {'select_f':[{'select__k':k,'clf__C':1.,'clf__class_weight':'balanced'} for k in ([250] if quick else [100,250,1000,5000])],
 'elastic_net':[{'clf__penalty':'elasticnet','clf__solver':'saga','clf__C':c,'clf__l1_ratio':r,'clf__class_weight':'balanced'} for c,r in elastic_pairs],
 'pca':[{'pca__n_components':n,'clf__C':1.,'clf__class_weight':'balanced'} for n in ([50] if quick else [50,100,250])]}
def model(name,p):
 clf=LogisticRegression(max_iter=5000,random_state=SEED)
 if name=='select_f':return Pipeline([('select',SelectKBest(f_classif,k=p['select__k'])),('scale',StandardScaler()),('clf',clf)]).set_params(**p)
 if name=='elastic_net':return Pipeline([('scale',StandardScaler()),('clf',clf)]).set_params(**p)
 return Pipeline([('scale',StandardScaler()),('pca',PCA(random_state=SEED)),('clf',clf)]).set_params(**p)
def score(m,x):return m.predict_proba(x)[:,1]
def run(quick=False):
 d=av.carregar_dataset(PATHS['processed']/'dataset_abide_processado.pkl'); out=Path(__file__).resolve().parents[1]/'experimentos/03_selecao_caracteristicas/Results';out.mkdir(parents=True,exist_ok=True)
 params=candidates(quick); rows=[]; stability=[]
 for fold in sorted(np.unique(d['fold_id'])):
  te=np.flatnonzero(d['fold_id']==fold);tr=np.flatnonzero(d['fold_id']!=fold)
  splits,info=av.splits_internos(d['y'][tr],d['sites'][tr],k_max=3,seed=SEED)
  if not info['viavel']:rows.append({'fold':int(fold),'status':'sem_split_interno_viavel'});continue
  for name,grid in params.items():
   scored=[]
   for p in grid:
    vals=[]
    for a,b in splits:
     m=model(name,p);m.fit(d['X'][tr[a]],d['y'][tr[a]]);vals.append(roc_auc_score(d['y'][tr[b]],score(m,d['X'][tr[b]])))
    scored.append((float(np.mean(vals)),p))
   auc,p=max(scored,key=lambda x:x[0]);m=model(name,p);t=time.perf_counter()
   with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always');m.fit(d['X'][tr],d['y'][tr])
   elapsed=time.perf_counter()-t;pr=score(m,d['X'][te]);pred=(pr>=.5).astype(int)
   rows.append({'modelo':name,'fold':int(fold),'seed':SEED,'n_teste':len(te),**av.metricas(d['y'][te],pred,pr),'auc_interno_media':auc,'parametros':json.dumps(p),'tempo_treino_s':elapsed,'avisos':' | '.join(str(w.message) for w in ws)})
   if name=='select_f':
    ids=m.named_steps['select'].get_support(indices=True);stability.extend({'fold':int(fold),'feature_index':int(i)} for i in ids)
  pd.DataFrame(rows).to_csv(out/'03_features_metrics_partial.csv',index=False)
 df=pd.DataFrame(rows);df.to_csv(out/'03_features_metrics.csv',index=False);pd.DataFrame(stability).to_csv(out/'03_features_selected.csv',index=False)
 (out/'03_features_config.json').write_text(json.dumps({'seed':SEED,'quick':quick,'outer':'stored SITE_ID folds','methods':list(params)},indent=2))
 return df
if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--quick',action='store_true');print(run(a.parse_args().quick).to_string(index=False))
