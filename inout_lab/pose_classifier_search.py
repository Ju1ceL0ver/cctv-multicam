"""All-day evaluation of anatomical pose + existing scene/person features."""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
root=Path(__file__).parent
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--imgsz',type=int,default=256);args=parser.parse_args();suffix='' if args.imgsz==256 else '_'+str(args.imgsz)
 assert np.load(root/('diagnostics/pose'+suffix+'_done.npy')).all(),'Wait for pose extraction; do not train on missing unfinished rows'
 z=np.load(root/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(root/'diagnostics/original_labels.npy')!=0;pose=np.load(root/('diagnostics/pose'+suffix+'_features.npy'));r=np.load(root/'diagnostics/rich_features.npy');h=np.load(root/'diagnostics/head_features.npy');c=np.load(root/'diagnostics/context_features.npy');crop=np.load(root/'diagnostics/crop_rgb_features.npy');sets={'pose_geometry':np.column_stack([r[:,:39],h[:,:126],pose]),'pose_context':np.column_stack([r,h,c,pose]),'pose_crop':np.column_stack([r,h,c,crop,pose])};P={name:np.zeros((len(y),3)) for name in sets};report={}
 with threadpool_limits(limits=3):
  for d in np.unique(days):
   tr=(days!=d)&v;te=days==d
   for name,A in sets.items():
    m=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=3).fit(A[tr],y[tr]);P[name][te]=m.predict_proba(A[te])
   print('pose evaluation',d,flush=True)
 for name,p in P.items():
  pred=p.argmax(1);gross=v&(y<2)&(pred<2)&(pred!=y);report[name]={'accuracy':float((pred[v]==y[v]).mean()),'wrong':int(((pred!=y)&v).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'doorway_precision':float((y[v&(pred==2)]==2).mean())};np.savez(root/('diagnostics/'+name+suffix+'_predictions.npz'),probabilities=p);print('POSE',name,report[name],flush=True)
 (root/('pose'+suffix+'_classifier_report.json')).write_text(json.dumps(report,indent=2))
