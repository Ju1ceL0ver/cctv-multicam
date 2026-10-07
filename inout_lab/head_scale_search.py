"""Label-free angular head-size / camera-ray proxies, not metric depth truth.
All labels including doorway remain; evaluation excludes complete days.
"""
from pathlib import Path
import json
import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parent

def head_scale_features(head,pose256,pose512,geometry,cal,mask_only=False):
    K=np.asarray(cal['K'],float);dist=np.asarray(cal['dist'],float)
    rv=np.asarray(cal['rvec'],float);tv=np.asarray(cal['tvec'],float)
    R=cv2.Rodrigues(rv)[0];C=-R.T@tv;H=float(C[2])
    if not mask_only:
        p256=pose256[:,:51].reshape(-1,17,3);p512=pose512[:,:51].reshape(-1,17,3)
        quality256=p256[:,:5,2].sum(1)*(pose256[:,127]>0)
        quality512=p512[:,:5,2].sum(1)*(pose512[:,127]>0)
        use512=quality512>=quality256
        p=np.where(use512[:,None,None],p512,p256)
    def undistort(points):
        return cv2.undistortPoints((points*[2560,1440]).reshape(-1,1,2).astype(float),K,dist).reshape(-1,2)
    proposals=[];parts=[] if mask_only else [use512[:,None].astype(float),p[:,:5,2],(p[:,:5,2]>.5).astype(float)]
    # Mask upper-body width candidates; upper portions can be incomplete.
    for j in range(4):
        h=head[:,8*j:8*j+8];center=h[:,:2]
        left=np.column_stack([h[:,4],h[:,1]]);right=np.column_stack([h[:,6],h[:,1]])
        width=np.linalg.norm(undistort(right)-undistort(left),axis=1)
        proposals.append((center,width,(h[:,2]>0).astype(float)))
    # Face spans can be foreshortened: preserve each independently and its confidence.
    for a,b in ([] if mask_only else [(1,2),(3,4),(0,3),(0,4),(1,3),(2,4)]):
        center=(p[:,a,:2]+p[:,b,:2])/2
        width=np.linalg.norm(undistort(p[:,a,:2])-undistort(p[:,b,:2]),axis=1)
        confidence=np.minimum(p[:,a,2],p[:,b,2]);proposals.append((center,width,confidence))
    for center,width,confidence in proposals:
        good=(confidence>.5)&(width>.001)&(width<.25)
        n=undistort(center);ray=np.column_stack([n,np.ones(len(n))])@R
        parts.extend([center,np.column_stack([width,confidence,good.astype(float),np.where(good,1/np.maximum(width,.001),0)]),ray])
        # Head-span/camera-height ratios cover varying head pose and scale.
        # They are feature hypotheses, not a hard floor-side override.
        for ratio in [.015,.03,.06]:
            lam=H*ratio/np.maximum(width,.001)
            xyz=C+lam[:,None]*ray;head_z=xyz[:,2]/H;xyz[:,2]=0
            foot=cv2.projectPoints(xyz,rv,tv,K,dist)[0].reshape(-1,2)/[2560,1440]
            foot=np.clip(np.nan_to_num(foot,nan=0,posinf=5,neginf=-5),-5,5)
            values=np.column_stack([foot,foot-geometry[:,6:8],np.clip(head_z,-5,5)])
            parts.append(np.where(good[:,None],values,0))
    result=np.column_stack(parts).astype(np.float32)
    assert np.isfinite(result).all()
    return result

def mask_head_scale_features(head,geometry,cal):
    return head_scale_features(head,None,None,geometry,cal,mask_only=True)

if __name__=='__main__':
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];G=z['G'][:,1:]
    valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    h=np.load(ROOT/'diagnostics/head_features.npy');p256=np.load(ROOT/'diagnostics/pose_features.npy');p512=np.load(ROOT/'diagnostics/pose_512_features.npy')
    cal=json.loads((ROOT/'diagnostics/calib_final_source.json').read_text())['cam1']
    S=head_scale_features(h,p256,p512,G,cal);np.save(ROOT/'diagnostics/head_scale_features.npy',S);print('head-scale features',S.shape,flush=True)
    A=np.column_stack([np.load(ROOT/f'diagnostics/{n}_features.npy') for n in ['rich','head','context']])
    sets={'geometry':np.column_stack([G,S]),'context':np.column_stack([A,S])};P={k:np.zeros((len(y),3)) for k in sets}
    with threadpool_limits(limits=2):
        for d in np.unique(days):
            tr=valid&(days!=d);te=days==d
            for name,B in sets.items():
                model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19) if name=='geometry' else ExtraTreesClassifier(n_estimators=160,max_features=.7,random_state=19,n_jobs=2)
                P[name][te]=model.fit(B[tr],y[tr]).predict_proba(B[te])
            print('head scale evaluated',d,flush=True)
    report={}
    for name,prob in P.items():
        pred=prob.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y)
        report[name]={'defined':int(valid.sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean())}
        np.savez(ROOT/f'diagnostics/head_scale_{name}_predictions.npz',probabilities=prob)
    (ROOT/'head_scale_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
