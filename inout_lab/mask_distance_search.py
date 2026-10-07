"""Dense signed mask-distance geometry; preserve all classes and exclude days.
Computed only from observed target masks; no annotation-dependent maps.
"""
from pathlib import Path
import numpy as np,cv2,json,time
from scipy.ndimage import distance_transform_edt
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
root=Path(__file__).parent

def mask_distance_features(mask,g):
 m=mask>127
 signed=(distance_transform_edt(m)-distance_transform_edt(~m)).astype(np.float32)
 distances=cv2.resize(signed,(32,24),interpolation=cv2.INTER_LINEAR).ravel()/180
 n,labels,stats,centers=cv2.connectedComponentsWithStats(m.astype(np.uint8),8)
 components=[]
 for k in sorted(range(1,n),key=lambda k:stats[k,cv2.CC_STAT_AREA],reverse=True)[:8]:
  x,y,w,h,area=stats[k];cx,cy=centers[k];components.extend([x/320,y/180,w/320,h/180,area/(320*180),cx/320,cy/180])
 components+= [0.]*(56-len(components))
 return np.r_[distances,components].astype(np.float32)

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];g=z['G'][:,1:];v=np.load(root/'diagnostics/original_labels.npy')!=0
 path=root/'diagnostics/mask_distance_features.npy'
 if path.exists():D=np.load(path)
 else:
  F=z['F'];D=np.stack([mask_distance_features(f[:,:,3],a) for f,a in zip(F,g)]);del F;np.save(path,D)
 assert D.shape==(len(y),824) and np.isfinite(D).all()
 r=np.load(root/'diagnostics/rich_features.npy');h=np.load(root/'diagnostics/head_features.npy');c=np.load(root/'diagnostics/context_features.npy');sets={'geometry_distance':np.column_stack([r[:,:39],h[:,:126],D]),'context_distance':np.column_stack([r,h,c,D])};P={k:np.zeros((len(y),3)) for k in sets};report={}
 with threadpool_limits(limits=2):
  for d in np.unique(days):
   tr=(days!=d)&v;te=days==d
   for name,A in sets.items():
    m=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=2).fit(A[tr],y[tr]);P[name][te]=m.predict_proba(A[te])
   print('mask distance',d,flush=True)
 for name,p in P.items():
  pred=p.argmax(1);gross=v&(y<2)&(pred<2)&(pred!=y);report[name]={'accuracy':float((pred[v]==y[v]).mean()),'wrong':int(((pred!=y)&v).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean())};np.savez(root/('diagnostics/'+name+'_predictions.npz'),probabilities=p);print('DISTANCE',name,report[name],flush=True)
 (root/'mask_distance_report.json').write_text(json.dumps(report,indent=2))
