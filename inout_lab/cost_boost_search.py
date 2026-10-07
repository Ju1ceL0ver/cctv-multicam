"""Cross-entropy + explicit inside/outside confusion cost, all 3 labels retained."""
from pathlib import Path
import numpy as np,json
from scipy.special import softmax
from xgboost import XGBClassifier
root=Path(__file__).parent

def loss(alpha):
 def objective(y,raw):
  y=y.astype(int);p=softmax(np.asarray(raw).reshape(-1,3),axis=1);target=np.eye(3)[y];bad=np.zeros_like(p);bad[np.arange(len(y)),1-np.minimum(y,1)]=1;bad[y==2]=0;cost=alpha*(p*bad).sum(1,keepdims=True)
  grad=p-target+cost*(bad-p)
  # Positive curvature bound for stable tree updates, rather than negative
  # diagonal Hessian where the probability penalty is locally concave.
  hess=2*p*(1-p)+cost*((bad-p)**2+p*(1-p))
  return grad,np.maximum(hess,1e-5)
 return objective

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;A=np.column_stack([np.load(root/('diagnostics/'+n+'_features.npy')) for n in ['rich','head','context']]);report={}
 for alpha in [0.,1.,3.,6.]:
  P=np.zeros((len(y),3))
  for d in np.unique(days):
   tr=(days!=d)&valid;te=days==d;model=XGBClassifier(n_estimators=220,max_depth=3,learning_rate=.07,subsample=.9,colsample_bytree=.8,min_child_weight=3,reg_lambda=3,n_jobs=3,random_state=19,objective=loss(alpha),num_class=3);model.fit(A[tr],y[tr]);P[te]=model.predict_proba(A[te]);print('cost',alpha,d,flush=True)
  pred=P.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);report[str(alpha)]={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int(((pred!=y)&valid).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'doorway_precision':float((y[valid&(pred==2)]==2).mean())};np.savez(root/('diagnostics/cost_boost_'+str(alpha)+'.npz'),probabilities=P);print('RESULT',alpha,report[str(alpha)],flush=True)
 (root/'cost_boost_report.json').write_text(json.dumps(report,indent=2))
