from pathlib import Path
import numpy as np,cv2,json
from sklearn.ensemble import ExtraTreesClassifier
root=Path(__file__).parent;z=np.load(root/'io_cam1.npz');F=z['F'];G=z['G'];y=z['y'];days=z['day'];valid=np.load(root/'diagnostics/original_labels.npy')!=0;rich=np.load(root/'diagnostics/rich_features.npy');head=np.load(root/'diagnostics/head_features.npy');rows=[]
for i,f in enumerate(F):
 rgb=f[:,:,:3]/255.;m=f[:,:,3]/255.;r,c=np.where(m>.25);top,bottom=r.min(),r.max();left,right=c.min(),c.max();v=[]
 centers=[(float(c[r<=top+max(1,int((bottom-top)*.2))].mean()),float(top)),(float(c[r>=bottom-2].mean()),float(bottom)),(G[i,7]*320,G[i,8]*180)]
 for cx,cy in centers:
  for radius in [4,10,20,40]:
   a,b=max(0,int(cx-radius)),max(0,int(cy-radius));e,d=min(320,int(cx+radius+1)),min(180,int(cy+radius+1));patch=rgb[b:d,a:e];mask=m[b:d,a:e]
   v+=cv2.resize(patch,(4,4),interpolation=cv2.INTER_AREA).ravel().tolist()
   bg=mask<.1;v+=patch[bg].mean(0).tolist() if bg.any() else [0]*3
   v+=patch[bg].std(0).tolist() if bg.any() else [0]*3
 rows.append(v)
context=np.asarray(rows,np.float32);np.save(root/'diagnostics/context_features.npy',context);A=np.column_stack([rich,head,context]);P=np.zeros((len(y),3))
for day in np.unique(days):
 tr=(days!=day)&valid;te=days==day;m=ExtraTreesClassifier(n_estimators=160,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=4).fit(A[tr],y[tr]);P[te]=m.predict_proba(A[te]);print('context',day,flush=True)
pred=P.argmax(1);cross=valid&(y<2)&(pred<2)&(pred!=y);r={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((pred[valid]!=y[valid]).sum()),'gross':int(cross.sum()),'days':{str(d):{'wrong':int(((pred!=y)&valid&(days==d)).sum()),'gross':int((cross&(days==d)).sum())} for d in np.unique(days)}}
np.savez(root/'diagnostics/context_predictions.npz',probabilities=P);(root/'context_features_report.json').write_text(json.dumps(r,indent=2));print(r,flush=True)
