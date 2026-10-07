"""Small depth+geometry classifier and fixed equal-weight baseline blend."""
from pathlib import Path
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parent
if __name__=='__main__':
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];day=z['day']
    D=np.load(ROOT/'diagnostics/depth_context_features.npy')
    valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    A=np.column_stack([z['G'][:,1:],D]);P=np.zeros((len(y),3))
    with threadpool_limits(limits=1):
        for d in np.unique(day):
            tr=valid&(day!=d);te=day==d
            P[te]=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19).fit(A[tr],y[tr]).predict_proba(A[te])
    B=(np.load(ROOT/'diagnostics/context_predictions.npz')['probabilities']+np.load(ROOT/'diagnostics/context_boost.npz')['probabilities'])/2
    for name,prob in [('depth_geometry',P),('depth_geometry_blend',(P+B)/2)]:
        pred=prob.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);covered=valid&(D[:,0]>0)
        report={'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'covered_wrong':int((covered&(pred!=y)).sum()),'covered_gross':int((covered&gross).sum()),'doorway_recall':float((pred[y==2]==2).mean())}
        (ROOT/f'{name}_report.json').write_text(json.dumps(report,indent=2));np.savez(ROOT/f'diagnostics/{name}_predictions.npz',probabilities=prob);print(name,report,flush=True)
