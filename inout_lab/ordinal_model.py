"""Treat doorway as the intermediate state instead of an unrelated class."""
from pathlib import Path
import numpy as np,json,pickle
from xgboost import XGBRegressor
from sklearn.metrics import confusion_matrix
root=Path(__file__).parent;out=root/'diagnostics';z=np.load(root/'io_cam1.npz');y=z['y'];day=z['day'];orig=np.load(out/'original_labels.npy');A=np.load(out/'rich_features.npy');tr=~np.isin(day,['20260918','20260919'])&(orig!=0);dev=day=='20260918';val=day=='20260919';target=np.array([0,2,1])[y];back=np.array([0,2,1]);r={};best=None
for depth in [2,3,4]:
 m=XGBRegressor(n_estimators=400,max_depth=depth,learning_rate=.04,subsample=.9,colsample_bytree=.8,n_jobs=4,random_state=0).fit(A[tr],target[tr]);p=m.predict(A[dev])
 for low,high in [(.5,1.5),(.6,1.4),(.7,1.3),(.8,1.2)]:
  pred=back[np.where(p<low,0,np.where(p>high,2,1))];valid=orig[dev]!=0;acc=float((pred[valid]==y[dev][valid]).mean())
  if best is None or acc>best['development18']['accuracy']:
   best={'depth':depth,'low':low,'high':high,'development18':{'accuracy':acc,'n':int(valid.sum()),'confusion':confusion_matrix(y[dev][valid],pred[valid],labels=[0,1,2]).tolist()}};winner=m
p=winner.predict(A[val]);pred=back[np.where(p<best['low'],0,np.where(p>best['high'],2,1))];valid=orig[val]!=0;best['validation19']={'accuracy':float((pred[valid]==y[val][valid]).mean()),'n':int(valid.sum()),'confusion':confusion_matrix(y[val][valid],pred[valid],labels=[0,1,2]).tolist()};print(best);(out/'ordinal_report.json').write_text(json.dumps(best,indent=2));pickle.dump(winner,open(out/'ordinal.pkl','wb'))
