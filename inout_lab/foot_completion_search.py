"""Head-to-floor completion from reliable visible feet; complete day exclusion.
Pseudo-ground truth is observational, not manually verified physical ground.
No side labels enter the regressor; train pseudo-foot rows use OOB predictions.
"""
from pathlib import Path
import numpy as np,json
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from threadpoolctl import threadpool_limits
root=Path(__file__).parent
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--mask-only',action='store_true');args=parser.parse_args();suffix='_mask' if args.mask_only else ''
 z=np.load(root/'io_cam1.npz');g=z['G'][:,1:];y=z['y'];days=z['day'];v=np.load(root/'diagnostics/original_labels.npy')!=0
 pose=np.load(root/'diagnostics/pose_features.npy');h=np.load(root/'diagnostics/head_features.npy');r=np.load(root/'diagnostics/rich_features.npy');c=np.load(root/'diagnostics/context_features.npy');A=np.column_stack([r,h,c]);U=np.column_stack([h[:,:16],pose[:,:21],pose[:,126:128]])
 if args.mask_only:U=h[:,:16]
 left,right=pose[:,45:48],pose[:,48:51];feet=(left[:,:2]+right[:,:2])/2
 reliable=(left[:,2]>.7)&(right[:,2]>.7)&(np.abs(feet[:,1]-g[:,7])<.03)&(np.abs(feet[:,0]-g[:,6])<.05)
 P=np.zeros((len(y),3));completions=np.zeros((len(y),2));folds={}
 with threadpool_limits(limits=2):
  for d in np.unique(days):
   tr=np.where((days!=d)&v)[0];te=np.where(days==d)[0];fit=np.where((days!=d)&reliable)[0]
   reg=ExtraTreesRegressor(n_estimators=160,min_samples_leaf=3,max_features=.8,bootstrap=True,oob_score=True,random_state=23,n_jobs=2).fit(U[fit],g[fit,6:8])
   completed=reg.predict(U);completed[fit]=reg.oob_prediction_;completions[te]=completed[te]
   extra=np.column_stack([completed,completed-g[:,6:8]]) if args.mask_only else np.column_stack([completed,completed-g[:,6:8],pose[:,127],pose[:,120]])
   B=np.column_stack([A,extra]);m=ExtraTreesClassifier(n_estimators=200,min_samples_leaf=1,max_features=.7,random_state=19,n_jobs=2).fit(B[tr],y[tr]);P[te]=m.predict_proba(B[te])
   check=te[reliable[te]];mae=np.abs(completed[check]-g[check,6:8]).mean(0)*[1280,720] if len(check) else [0,0]
   folds[str(d)]={'training_visible_feet':int(len(fit)),'held_visible_feet':int(len(check)),'held_pseudo_foot_mae_pixels':np.asarray(mae).tolist()};print('foot completion',d,folds[str(d)],flush=True)
 pred=P.argmax(1);gross=v&(y<2)&(pred<2)&(pred!=y);report={'accuracy':float((pred[v]==y[v]).mean()),'wrong':int(((pred!=y)&v).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean()),'folds':folds}
 np.savez(root/('diagnostics/foot_completion'+suffix+'_predictions.npz'),probabilities=P,completed_feet=completions);(root/('foot_completion'+suffix+'_report.json')).write_text(json.dumps(report,indent=2));print('FOOT_COMPLETION',report,flush=True)
