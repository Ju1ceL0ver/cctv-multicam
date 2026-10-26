"""Tune only on non-test days; frozen held-day baseline comparison and saved predictor."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'4')
import hashlib,json,sys,time,pickle,warnings
from pathlib import Path
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from staff_model import Features,RoleModel,StaffClassifier,norm,geometry
from staff_audit import metrics
warnings.filterwarnings('ignore',category=ConvergenceWarning)
TEST=('20260923','20261001')
DEV=('20260920','20260928','20261002','20261004')
CONFIGS={
 'baseline':dict(kind='mlp'),
 'mlp_w2':dict(kind='mlp',staff_weight=2),
 'mlp_w3_reg':dict(kind='mlp',staff_weight=3,alpha=.01),
 'mlp_reg':dict(kind='mlp',alpha=.1),
 'mlp_small':dict(kind='mlp',layers=[64],alpha=.01),
 'mlp_small_w2':dict(kind='mlp',layers=[64],alpha=.01,staff_weight=2),
 'mlp_ensemble':dict(kind='mlp',alpha=.01,seeds=[0,1,2]),
 'knn3':dict(kind='knn',k=3),'knn7':dict(kind='knn',k=7),'knn15':dict(kind='knn',k=15),
 'svm1':dict(kind='svm',C=10,gamma=.5),
 'svm2':dict(kind='svm',C=10,gamma=2,staff_weight=2),
 'svm3':dict(kind='svm',C=100,gamma=.5),
}


def run(snapshot,out):
 out=Path(out);out.mkdir(parents=True,exist_ok=True)
 report_path=out/'report.json'
 if report_path.exists() and json.loads(report_path.read_text()).get('complete'):
  raise RuntimeError('Completed test already exists; do not re-evaluate while tuning')
 z=np.load(snapshot);c,s,y,days,src=z['cloth'],z['shape'],z['y'],z['days'],z['src'];g=geometry(z['boxes'],z['cam'])
 raw=np.concatenate([norm(c),norm(s)],1)/np.sqrt(2)
 test=np.isin(days,TEST);allowed=~test;dev=np.isin(days,DEV)
 assert not np.any(test&dev)
 ids=z['ids'];assert len(set(ids[test])&set(ids[allowed]))==0
 hashes=[hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest() for a,b in zip(c,s)]
 assert not (set(np.array(hashes)[test])&set(np.array(hashes)[allowed]))
 report={'test_days':list(TEST),'development_days':list(DEV),'train_days':sorted(set(days[allowed])),
         'snapshot_sha256':hashlib.sha256(Path(snapshot).read_bytes()).hexdigest(),'configs':CONFIGS,
         'protocol':'Test days excluded from every fit, PCA, model/ensemble/threshold selection; previously reviewed historical days; within-store repeated staff allowed.',
         'selection_metric':'pooled development accuracy, ties then staff recall; all cases answered',
         'development':{},'progress':[]}
 def save():report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
 save();oof={k:np.full(len(y),np.nan) for k in CONFIGS}
 for d in DEV:
  tr=allowed&(days!=d);va=days==d
  f=Features().fit(c[tr],s[tr]);xtr=f.transform(c[tr],s[tr],g[tr]);xva=f.transform(c[va],s[va],g[va])
  for name,cfg in CONFIGS.items():
   t=time.time();m=RoleModel(cfg).fit(xtr,y[tr],raw[tr]);oof[name][va]=m.predict(xva,raw[va])
   entry={'day':d,'model':name,'accuracy':float(((oof[name][va]>=.5)==y[va]).mean()),'seconds':time.time()-t}
   report['progress'].append(entry);save();print(entry,flush=True)
 recipes={k:([k],[1.]) for k in CONFIGS}
 for a in ['mlp_w2','mlp_reg','mlp_ensemble','svm1','svm2']:
  for b in ['knn3','knn7']:
   name=a+'_'+b;recipes[name]=([a,b],[.5,.5]);oof[name]=.5*(oof[a]+oof[b])
 # Threshold choice uses only out-of-day development predictions.
 choices=[]
 for name,p in oof.items():
  thresholds=[.5] if name=='baseline' else [.35,.45,.5,.55,.65]
  for th in thresholds:
   q=p[dev]>=th;acc=float((q==y[dev]).mean());rec=float(q[y[dev]].mean())
   choices.append((acc,rec,name,th))
  metric=metrics(y[dev],p[dev],src[dev]);report['development'][name]=metric
 chosen=max(choices,key=lambda t:(t[0],t[1]));acc,rec,name,th=chosen
 report['selected']={'name':name,'threshold':th,'development_accuracy':acc,'staff_recall':rec,'recipe':recipes[name]};save()
 print('SELECTED',report['selected'],flush=True)
 # Selection is written to disk before accessing test examples for final transformation/evaluation.
 f=Features().fit(c[allowed],s[allowed]);xtr=f.transform(c[allowed],s[allowed],g[allowed]);xt=f.transform(c[test],s[test],g[test])
 report['test']={};trained={}
 for nm in set(['baseline']+recipes[name][0]):trained[nm]=RoleModel(CONFIGS[nm]).fit(xtr,y[allowed],raw[allowed])
 for key,names,weights,threshold in [('baseline',['baseline'],[1.],.5),('selected',*recipes[name],th)]:
  predictor=StaffClassifier(f,[trained[n] for n in names],weights,threshold)
  t=time.time();p=predictor.predict_proba(c[test],s[test],z['boxes'][test],z['cam'][test])[:,1];elapsed=time.time()-t
  # metrics() uses .5: shift probabilities for count metrics only; preserve raw probabilities in archive.
  adjusted=np.clip(p-th+.5,0,1);v=metrics(y[test],adjusted,src[test]);v['threshold']=threshold;v['seconds_predict_all']=elapsed
  v['by_day']={d:metrics(y[test][days[test]==d],adjusted[days[test]==d],src[test][days[test]==d]) for d in TEST}
  report['test'][key]=v
  np.savez_compressed(out/(key+'_predictions.npz'),ids=ids[test],y=y[test],p=p,days=days[test])
  with open(out/(key+'.pkl'),'wb') as h:pickle.dump(predictor,h)
  print('TEST',key,v,flush=True)
 report['complete']=True;save()
 # Reload saved API; labels and predictions must match, weights exclude all test days.
 with open(out/'selected.pkl','rb') as h:loaded=pickle.load(h)
 with np.load(out/'selected_predictions.npz') as q:
  np.testing.assert_allclose(loaded.predict_proba(c[test],s[test],z['boxes'][test],z['cam'][test])[:,1],q['p'],atol=1e-7)
 print('RELOAD_PARITY_OK',flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):run(sys.argv[1],sys.argv[2])
