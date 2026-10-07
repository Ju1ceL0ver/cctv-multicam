"""Foreground detail ablation; held-out whole days, all defined three-class rows.
Detail is a distance cue, not a calibrated depth measurement.
"""
from pathlib import Path
import json,time
import cv2
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parent

def detail_features(crop):
    gray=cv2.cvtColor(crop[:,:,:3],cv2.COLOR_RGB2GRAY).astype(np.float32)/255
    mask=(crop[:,:,3]>127).astype(np.uint8)
    mask=cv2.erode(mask,np.ones((3,3),np.uint8))>0
    ys=np.where(mask)[0]
    top=int(ys.min()) if len(ys) else 0
    height=int(ys.max()-top+1) if len(ys) else 1
    out=[]
    for fraction in [.25,.5,1.]:
        region=mask.copy();region[int(top+height*fraction):]=False
        n=int(region.sum());out.extend([n/gray.size,float(n>=16)])
        if n<16:
            out.extend([0.]*35);continue
        values=gray[region];contrast=float(values.std())
        out.extend([float(values.mean()),contrast])
        for sigma in [0.,.7,1.4,2.8]:
            g=gray if sigma==0 else cv2.GaussianBlur(gray,(0,0),sigma)
            lap=cv2.Laplacian(g,cv2.CV_32F)
            gx=cv2.Sobel(g,cv2.CV_32F,1,0);gy=cv2.Sobel(g,cv2.CV_32F,0,1)
            mag=np.hypot(gx,gy)[region]
            out.extend([float(np.mean(lap[region]**2))/(contrast**2+1e-4),
                        float(mag.mean())/(contrast+.01),float(np.quantile(mag,.9))/(contrast+.01)])
        for step in [1,2,4]:
            for axis in [0,1]:
                shifted=np.roll(gray,step,axis);valid=region&np.roll(region,step,axis)
                if axis==0:valid[:step]=False
                else:valid[:,:step]=False
                diff=np.abs(gray-shifted)[valid]
                out.extend([float(diff.mean())/(contrast+.01) if len(diff) else 0.,
                            float((diff<1/255).mean()) if len(diff) else 0.,
                            float((diff<3/255).mean()) if len(diff) else 0.])
        for sigma in [.7,1.4,2.8]:
            residual=gray-cv2.GaussianBlur(gray,(0,0),sigma)
            out.append(float(np.mean(residual[region]**2))/(contrast**2+1e-4))
    return np.asarray(out,np.float32)

if __name__=='__main__':
    start=time.monotonic();z=np.load(ROOT/'io_cam1.npz');X=z['X']
    D=np.stack([detail_features(x) for x in X]);del X
    assert D.shape==(1628,111) and np.isfinite(D).all(),D.shape
    np.save(ROOT/'diagnostics/pixel_detail_features.npy',D)
    print('detail features',D.shape,'seconds',round(time.monotonic()-start,1),flush=True)
    y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    G=z['G'][:,1:];A=np.column_stack([np.load(ROOT/f'diagnostics/{n}_features.npy') for n in ['rich','head','context']])
    sets={'geometry':G,'geometry_detail':np.column_stack([G,D]),'context_detail':np.column_stack([A,D])}
    probs={k:np.zeros((len(y),3)) for k in sets}
    with threadpool_limits(limits=2):
        for day in np.unique(days):
            tr=valid&(days!=day);te=days==day
            for name,B in sets.items():
                model=ExtraTreesClassifier(n_estimators=200,max_features=.7,random_state=19,n_jobs=2)
                probs[name][te]=model.fit(B[tr],y[tr]).predict_proba(B[te])
            print('detail evaluated',day,flush=True)
    report={}
    for name,p in probs.items():
        pred=p.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y)
        report[name]={'defined':int(valid.sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'doorway_recall':float((pred[y==2]==2).mean())}
        np.savez(ROOT/f'diagnostics/pixel_{name}_predictions.npz',probabilities=p)
    (ROOT/'pixel_detail_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,ensure_ascii=False),flush=True)
