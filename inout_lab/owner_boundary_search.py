"""New hand-marked threshold + height hypotheses, complete held-day exclusion.
No labels enter geometric features; no metres/square tiles assumed. Research only.
"""
from pathlib import Path
import json,time
import cv2,numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).parent

def boundary_features(head,g):
    cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1'];K=np.array(cal['K'],float);K[:2]*=.5;dist=np.array(cal['dist'],float)
    ends=next(o['points'] for o in json.loads((ROOT/'owner_seams.json').read_text())['objects'] if o['id']=='threshold1')
    def ray(p):
        u=cv2.undistortPoints(np.asarray(p,float).reshape(-1,1,2),K,dist).reshape(-1,2)
        return np.column_stack([u,np.ones(len(u))])
    line=np.cross(*ray(ends));line/=np.linalg.norm(line[:2])
    if ray([[640,600]])[0]@line<0:line=-line
    old=cv2.Rodrigues(np.array(cal['rvec']))[0][:,2]
    seam=np.array(json.loads((ROOT/'owner_seams_geometry.json').read_text())['floor_normal_camera_coordinates'])
    parts=[]
    for p in [g[:,6:8]]+[head[:,8*j:8*j+2] for j in range(8)]:
        q=ray(p*[1280,720]);parts.append((q@line)[:,None])
        for n in [old,seam]:
            base=q@line;delta=(q@n)*(line@n)-base
            required=-base/np.where(abs(delta)>.001,delta,np.copysign(.001,delta+1e-12))
            parts.append(np.clip(required,-3,3)[:,None])
            for h in [.2,.35,.5,.65]:
                qp=(1-h)*q+h*(q@n)[:,None]*n
                denom=np.where(abs(qp[:,2])>.01,qp[:,2],.01)
                parts.append(np.clip((qp@line)/denom,-3,3)[:,None])
    return np.column_stack(parts).astype('float32')

def metrics(P,y,v,days):
    assert np.isfinite(P).all() and np.allclose(P.sum(1),1,atol=1e-4)
    p=P.argmax(1);gross=v&(y<2)&(p<2)&(p!=y);cm=np.zeros((3,3),int);np.add.at(cm,(y[v],p[v]),1)
    return {'defined':int(v.sum()),'wrong':int((v&(p!=y)).sum()),'accuracy':float((p[v]==y[v]).mean()),'gross':int(gross.sum()),'gross_indices':np.flatnonzero(gross).tolist(),'confusion':cm.tolist(),'doorway_recall':float((p[v&(y==2)]==2).mean()),'folds':{str(d):{'count':int((v&(days==d)).sum()),'wrong':int((v&(days==d)&(p!=y)).sum()),'gross':int((gross&(days==d)).sum())} for d in np.unique(days)}}

if __name__=='__main__':
    start=time.perf_counter();z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    A=np.load(ROOT/'diagnostics/aligned_features.npy');g=np.load(ROOT/'diagnostics/aligned_geometry.npy');H=A[:,234:936];E=boundary_features(H,g);np.save(ROOT/'diagnostics/owner_boundary_features.npy',E)
    sets={'geometry':np.column_stack([g,H[:,:126],E]),'context':np.column_stack([A[:,:1584],E]),'scale':np.column_stack([g,H[:,:126],E,np.load(ROOT/'diagnostics/aligned_density_features.npy')])}
    P={name+'_'+algo:np.full((len(y),3),np.nan) for name in sets for algo in ['extra','boost']};report={}
    with threadpool_limits(limits=2):
        for d in np.unique(days):
            tr=v&(days!=d);te=days==d;assert not set(days[tr])&set(days[te])
            for name,B in sets.items():
                for algo in ['extra','boost']:
                    model=ExtraTreesClassifier(n_estimators=160,max_features=.7,n_jobs=2,random_state=19) if algo=='extra' else HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19)
                    key=name+'_'+algo;P[key][te]=model.fit(B[tr],y[tr]).predict_proba(B[te])
            print('owner boundary completed',d,'seconds',round(time.perf_counter()-start),flush=True)
    for name,p in P.items():
        report[name]=metrics(p,y,v,days);np.savez(ROOT/f'diagnostics/owner_boundary_{name}.npz',probabilities=p,completed_days=np.unique(days));print(name,report[name]['wrong'],report[name]['gross'],report[name]['gross_indices'],flush=True)
    (ROOT/'owner_boundary_report.json').write_text(json.dumps({'protocol':'All 9 held-out days; repeatedly explored diagnostic set; no unknown abstention','models':report,'seconds':time.perf_counter()-start},indent=2))
