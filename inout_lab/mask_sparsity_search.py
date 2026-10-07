"""Actual local mask filling / holes / thickness at fixed source pixel scale.
No labels in extraction. Every defined row, including doorway, is evaluated.
"""
from pathlib import Path
import json,time
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parent

def sparsity_features(full):
    soft=full[:,:,3].astype(np.float32)/255;mask=(soft>.25).astype(np.uint8)
    rr,cc=np.where(mask);out=[]
    if not len(rr):return np.zeros(73,np.float32)
    top,bottom=int(rr.min()),int(rr.max());distance=cv2.distanceTransform(mask,cv2.DIST_L2,5)
    fills=[cv2.blur(mask.astype(np.float32),(k,k)) for k in [3,7,15]]
    for fraction in [.15,.3,1.]:
        region=mask.astype(bool);region[top+max(1,int((bottom-top+1)*fraction)):]=False
        values=distance[region]*4 # fixed 1280x720-equivalent source pixels
        out.extend([float(values.mean()),float(values.std()),*np.quantile(values,[.1,.5,.9]).tolist(),float((values>=8).mean()),float((values>=16).mean()),float((values>=32).mean())])
        out.extend([float(soft[region].mean()),float((soft[region]>.99).mean())])
        for fill in fills:
            v=fill[region];out.extend([float(v.mean()),*np.quantile(v,[.1,.5,.9]).tolist()])
    n,lab,stats,_=cv2.connectedComponentsWithStats(mask);areas=stats[1:,4]
    inv=1-mask;_,labels,bgstats,_=cv2.connectedComponentsWithStats(inv)
    boundary=set(np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]])))|{0}
    holes=[s[4] for i,s in enumerate(bgstats) if i not in boundary]
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    hull_area=cv2.contourArea(cv2.convexHull(np.concatenate(contours))) if contours else 0
    mass=float(mask.sum());out.extend([float(len(areas)),float(areas.max()/mass),float(len(holes)),float(sum(holes)/mass),float(mass/max(hull_area,1)),float(len(areas[areas>=3])/max(mass,1)),float((soft[mask>0]<.99).mean())])
    out=np.asarray(out,np.float32);assert out.shape==(73,) and np.isfinite(out).all()
    return out

if __name__=='__main__':
    z=np.load(ROOT/'io_cam1.npz');F=z['F'];start=time.perf_counter();S=np.stack([sparsity_features(f) for f in F]);del F
    np.save(ROOT/'diagnostics/mask_sparsity_features.npy',S);print('local sparsity prepared',S.shape,'seconds',round(time.perf_counter()-start,1),flush=True)
    y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    A=np.load(ROOT/'diagnostics/aligned_features.npy')[:,:1584];G=np.load(ROOT/'diagnostics/aligned_geometry.npy')
    sets={'sparsity_only':S,'geometry_sparsity':np.column_stack([G,S]),'aligned_context_sparsity':np.column_stack([A,S])};P={k:np.zeros((len(y),3)) for k in sets}
    with threadpool_limits(limits=2):
        for d in np.unique(days):
            tr=valid&(days!=d);te=days==d
            for name,B in sets.items():
                model=ExtraTreesClassifier(n_estimators=160,max_features=.7,random_state=19,n_jobs=2) if name=='aligned_context_sparsity' else HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19)
                P[name][te]=model.fit(B[tr],y[tr]).predict_proba(B[te])
            print('sparsity evaluated',d,flush=True)
    report={}
    for name,prob in P.items():
        p=prob.argmax(1);g=valid&(y<2)&(p<2)&(p!=y)
        report[name]={'defined':int(valid.sum()),'accuracy':float((p[valid]==y[valid]).mean()),'wrong':int((valid&(p!=y)).sum()),'gross':int(g.sum()),'gross_indices':np.where(g)[0].tolist(),'doorway_recall':float((p[y==2]==2).mean())}
        np.savez(ROOT/f'diagnostics/{name}_predictions.npz',probabilities=prob)
    (ROOT/'mask_sparsity_report.json').write_text(json.dumps(report,indent=2));print('SPARSITY RESULT',json.dumps(report),flush=True)
