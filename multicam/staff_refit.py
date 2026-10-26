"""Fast supervised role refit including the consumed blind set; grouped-day development only.
No independent test is claimed. Immutable old blind evaluation and frozen weights are preserved.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ.setdefault(k,'4')
import hashlib,json,pickle,sys,time,warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from threadpoolctl import threadpool_limits
from sklearn.exceptions import ConvergenceWarning
from staff_model import Features,RoleModel,StaffClassifier,norm,geometry
from storage import atomic_json
warnings.filterwarnings('ignore',category=ConvergenceWarning)
ROOT=Path(__file__).resolve().parent
CONFIGS={
 'previous_recipe':dict(kind='svm',C=10,gamma=.5),
 'svm_w2':dict(kind='svm',C=10,gamma=.5,staff_weight=2),
 'svm_w3':dict(kind='svm',C=10,gamma=.5,staff_weight=3),
 'svm_g2':dict(kind='svm',C=10,gamma=2),
 'svm_g2_w2':dict(kind='svm',C=10,gamma=2,staff_weight=2),
 'svm_g8':dict(kind='svm',C=10,gamma=8),
 'svm_c100':dict(kind='svm',C=100,gamma=.5),
 'knn3':dict(kind='knn',k=3),'knn7':dict(kind='knn',k=7),
 'mlp_w2':dict(kind='mlp',alpha=.01,staff_weight=2),
 'mlp_ensemble':dict(kind='mlp',alpha=.01,seeds=[0,1,2]),
}


def metric(y,p,threshold):
 q=p>=threshold;sr=float(q[y].mean());cr=float((~q[~y]).mean())
 return {'n':len(y),'accuracy':float((q==y).mean()),'balanced_accuracy':(sr+cr)/2,
  'staff_recall':sr,'customer_recall':cr,'staff_as_customer':int((~q&y).sum()),
  'customer_as_staff':int((q&~y).sum()),'threshold':threshold}


def main():
 t=time.time();out=ROOT/'data'/'staff'/(sys.argv[1] if len(sys.argv)>1 else 'refit_guarded_20261008');out.mkdir(parents=True,exist_ok=True)
 if (out/'report.json').exists() and json.loads((out/'report.json').read_text()).get('complete'):raise RuntimeError('Completed refit exists; refusing overwrite')
 base=ROOT/'data'/'staff'/'audit_snapshot_20261008.npz';blind=ROOT/'data'/'staff_blind_20260911_20260912'
 model_hash=hashlib.sha256((blind/'frozen_model.pkl').read_bytes()).hexdigest()
 old_eval_hash=hashlib.sha256((blind/'evaluation.json').read_bytes()).hexdigest()
 z=np.load(base);b=np.load(blind/'features.npz');lab=json.loads((blind/'labels.json').read_text());meta=json.loads((blind/'manifest.json').read_text())['items']
 bi=[k for k,i in enumerate(b['ids']) if lab[str(i)]['label'] in (1,2)];bid=[str(b['ids'][k]) for k in bi]
 c=np.concatenate([z['cloth'],b['cloth'][bi]]);s=np.concatenate([z['shape'],b['shape'][bi]])
 y=np.concatenate([z['y'],np.array([lab[i]['label']==2 for i in bid])]);days=np.concatenate([z['days'],np.array([meta[i]['day'] for i in bid])])
 ids=np.concatenate([z['ids'],np.array(['blind_'+i for i in bid])]);boxes=np.concatenate([z['boxes'],b['boxes'][bi]]);cam=np.concatenate([z['cam'],np.array(['cam1']*len(bi))]);src=np.concatenate([z['src'],np.array(['counter']*len(bi))])
 assert len(set(ids))==len(ids)
 np.savez_compressed(out/'training.npz',ids=ids,cloth=c,shape=s,y=y,days=days,boxes=boxes,cam=cam,src=src)
 g=geometry(boxes,cam);raw=np.concatenate([norm(c),norm(s)],1)/np.sqrt(2)
 report={'n':len(y),'added':len(bi),'staff_n':int(y.sum()),'customer_n':int((~y).sum()),'days':sorted(set(days)),
  'configs':CONFIGS,'protocol':'4-fold grouped-day CV used for model and threshold selection, not an independent final test. Each fold PCA fits only its training days; final model trained on all 3523 labelled samples including consumed blind days.',
  'objective':'maximize balanced accuracy subject to ordinary accuracy at least previous recipe CV (.45); then ordinary accuracy; all samples answered',
  'old_blind_evaluation_preserved_sha256':old_eval_hash,'progress':[]}
 def save():atomic_json(out/'report.json',report)
 save();oof={name:np.full(len(y),np.nan) for name in CONFIGS};folds=np.full(len(y),-1,int)
 for fold,(tr,va) in enumerate(GroupKFold(4).split(c,y,days)):
  assert not set(days[tr])&set(days[va]);folds[va]=fold
  f=Features().fit(c[tr],s[tr]);xtr=f.transform(c[tr],s[tr],g[tr]);xva=f.transform(c[va],s[va],g[va])
  for name,cfg in CONFIGS.items():
   m=RoleModel(cfg).fit(xtr,y[tr],raw[tr]);oof[name][va]=m.predict(xva,raw[va])
   entry={'fold':fold,'name':name,'held_days':sorted(set(days[va])),'metrics':metric(y[va],oof[name][va],.5)}
   report['progress'].append(entry);save();print('CV',fold,name,entry['metrics']['balanced_accuracy'],flush=True)
 recipes={n:([n],[1.]) for n in CONFIGS}
 for a in ['previous_recipe','svm_w2','svm_g2','mlp_ensemble']:
  for bname in ['knn3','knn7']:
   name=a+'_'+bname;oof[name]=.5*(oof[a]+oof[bname]);recipes[name]=([a,bname],[.5,.5])
 assert all(np.isfinite(v).all() for v in oof.values())
 report['previous_recipe_cv']=metric(y,oof['previous_recipe'],.45)
 np.savez_compressed(out/'all_development_predictions.npz',ids=ids,y=y,days=days,folds=folds,**oof)
 floor=report['previous_recipe_cv']['accuracy'];report['accuracy_floor']=floor
 best=None;report['candidates']={}
 for name,p in oof.items():
  best_local=None
  for threshold in (.2,.3,.4,.45,.5,.6,.7):
   mm=metric(y,p,threshold);key=(mm['balanced_accuracy'],mm['accuracy'])
   if best_local is None or key>best_local[0]:best_local=(key,mm)
   if mm['accuracy']>=floor and (best is None or key>best[0]):best=(key,name,mm)
  report['candidates'][name]=best_local[1]
 _,chosen,selected=best;report['selected']={'name':chosen,'recipe':recipes[chosen],**selected}
 report['selected_cv_by_day']={d:metric(y[days==d],oof[chosen][days==d],selected['threshold']) for d in sorted(set(days))}
 save();print('SELECTED',report['selected'],flush=True)
 f=Features().fit(c,s);x=f.transform(c,s,g)
 names,weights=recipes[chosen];models=[RoleModel(CONFIGS[n]).fit(x,y,raw) for n in names]
 predictor=StaffClassifier(f,models,weights,selected['threshold'])
 p=predictor.predict_proba(c,s,boxes,cam)[:,1]
 report['training_fit']=metric(y,p,predictor.threshold)
 added=np.arange(len(y))>=len(z['y']);report['added_training_fit']=metric(y[added],p[added],predictor.threshold)
 tmp=out/'model.pkl.tmp'
 with tmp.open('wb') as h:pickle.dump(predictor,h)
 os.replace(tmp,out/'model.pkl')
 with (out/'model.pkl').open('rb') as h:loaded=pickle.load(h)
 np.testing.assert_allclose(loaded.predict_proba(c[:64],s[:64],boxes[:64],cam[:64])[:,1],p[:64],atol=1e-7)
 # Canonical current predictor: does not replace old frozen evaluation artefacts.
 current=ROOT/'data'/'staff'/'current_role.pkl';backup=out/'previous_current_role.pkl'
 if current.exists():
  import shutil;shutil.copyfile(current,backup)
 tmp=current.with_suffix('.pkl.tmp')
 with tmp.open('wb') as h:pickle.dump(predictor,h)
 os.replace(tmp,current)
 report['current_model_sha256']=hashlib.sha256(current.read_bytes()).hexdigest()
 assert hashlib.sha256((blind/'frozen_model.pkl').read_bytes()).hexdigest()==model_hash
 assert hashlib.sha256((blind/'evaluation.json').read_bytes()).hexdigest()==old_eval_hash
 report['seconds']=time.time()-t;report['complete']=True;report['reload_parity']=True
 save();atomic_json(ROOT/'data'/'staff'/'current_role.json',{'path':str(current),'sha256':report['current_model_sha256'],'report':str(out/'report.json'),'trained_days':report['days'],'threshold':predictor.threshold})
 np.savez_compressed(out/'development_predictions.npz',ids=ids,y=y,days=days,folds=folds,p=oof[chosen],threshold=predictor.threshold)
 print('FINAL',json.dumps({k:report[k] for k in ('selected','previous_recipe_cv','training_fit','added_training_fit','seconds','current_model_sha256')},ensure_ascii=False),flush=True)

if __name__=='__main__':
 with threadpool_limits(limits=4):main()
