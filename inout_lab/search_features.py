"""Development-only feature/model search; no held-out labels used for fitting.
Keep 18/19 split from diagnostics. 19 has already been inspected: report it as
validation, not a pristine final test. Do not claim 100% by changing labels.
"""
from pathlib import Path
import json,pickle,time
import numpy as np,cv2
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
root=Path(__file__).parent;out=root/'diagnostics';z=np.load(root/'io_cam1.npz');F=z['F'];X=z['X'];G=z['G'][:,1:];y=z['y'];days=z['day']
features=[]
for i,f in enumerate(F):
 m=f[:,:,3]/255.;floor=f[:,:,4]/255.;binary=m>.25
 rr,cc=np.where(binary);area=max(1,m.sum());yy,xx=np.indices(m.shape);cx=(m*xx).sum()/area;cy=(m*yy).sum()/area
 signed=distance_transform_edt(floor>.5)-distance_transform_edt(floor<=.5)
 vals=signed[binary];row0,row1=rr.min(),rr.max();col0,col1=cc.min(),cc.max()
 v=[area/m.size,cx/320,cy/180,row0/180,row1/180,col0/320,col1/320,(m*floor).sum()/area]
 v+=list(np.percentile(vals,[0,10,25,50,75,90,100])/180)
 for fraction in [.05,.1,.2,.35,.5]:
  cut=row1-max(1,int((row1-row0+1)*fraction));low=m.copy();low[:cut]=0;a=max(1,low.sum());v += [(low*floor).sum()/a,(low*signed).sum()/a/180,(low*xx).sum()/a/320]
 # Fixed-scene location map and the shape in the cropped person image.
 v+=list(cv2.resize(m,(16,8),interpolation=cv2.INTER_AREA).flatten())
 v+=list(cv2.resize(X[i,:,:,3]/255.,(8,8),interpolation=cv2.INTER_AREA).flatten())
 rgb=f[:,:,:3]/255.;v+=list((rgb*m[:,:,None]).sum((0,1))/area)
 features.append(v)
A=np.column_stack([G,features]).astype('float32');np.save(out/'rich_features.npy',A)
tr=~np.isin(days,['20260918','20260919']);dev=days=='20260918';val=days=='20260919';models={
 'extra_leaf1':ExtraTreesClassifier(n_estimators=400,min_samples_leaf=1,max_features=1.,random_state=1,n_jobs=4),
 'extra_leaf3':ExtraTreesClassifier(n_estimators=400,min_samples_leaf=3,max_features=.7,random_state=1,n_jobs=4),
 'forest':RandomForestClassifier(n_estimators=400,min_samples_leaf=2,max_features=.7,random_state=1,n_jobs=4),
 'hist':HistGradientBoostingClassifier(max_iter=300,max_leaf_nodes=15,l2_regularization=2,random_state=1),
 'svm':make_pipeline(StandardScaler(),SVC(C=10,gamma='scale',probability=True,random_state=1)),
 'xgb':XGBClassifier(n_estimators=400,max_depth=3,learning_rate=.04,subsample=.9,colsample_bytree=.8,n_jobs=4,random_state=1)}
r={};preds={};trained={}
for name,m in models.items():
 m.fit(A[tr],y[tr]);p=m.predict_proba(A[dev]);r[name]={'development18':{'accuracy':float((p.argmax(1)==y[dev]).mean()),'confusion':confusion_matrix(y[dev],p.argmax(1),labels=[0,1,2]).tolist()}};trained[name]=m;preds[name+'_development18']=p
 print(name,r[name],flush=True)
best=max(r,key=lambda k:r[k]['development18']['accuracy']);m=trained[best];p=m.predict_proba(A[val]);r[best]['validation19']={'accuracy':float((p.argmax(1)==y[val]).mean()),'confusion':confusion_matrix(y[val],p.argmax(1),labels=[0,1,2]).tolist()};preds[best+'_validation19']=p
r['selected']=best;(out/'rich_report.json').write_text(json.dumps(r,indent=2));pickle.dump(m,open(out/'rich_best.pkl','wb'));np.savez(out/'rich_predictions.npz',**preds)
print('SELECTED',best,r[best],flush=True)
