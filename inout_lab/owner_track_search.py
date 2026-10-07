"""Owner threshold + true SAM history up to30s ahead. No pN identity or labels.
All rows retained; missing tracks explicit. Diagnostic buffered classifier only.
"""
from pathlib import Path
import gzip,json,time
import cv2,numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from track_context_features import geometry
from owner_boundary_search import metrics
R=Path(__file__).parent

def features():
    matches=json.loads((R/'track_context_match_report.json').read_text())['matches'];tracks={}
    cal=json.loads((R/'diagnostics/calib_final_source.json').read_text())['cam1'];K=np.array(cal['K'],float);K[:2]*=.5;dist=np.array(cal['dist'],float)
    ends=next(o['points'] for o in json.loads((R/'owner_seams.json').read_text())['objects'] if o['id']=='threshold1')
    def und(p):return cv2.undistortPoints(np.asarray(p,float).reshape(-1,1,2),K,dist,P=K).reshape(-1,2)
    a,b=und(ends);n=np.array([-(b-a)[1],(b-a)[0]]);n/=np.linalg.norm(n)
    if (und([[640,600]])[0]-a)@n<0:n=-n
    for day in ['20260917','20260918','20260919']:
        with gzip.open(R/f'diagnostics/temporal/{day}_sam31.jsonl.gz','rt') as f:
            next(f)
            for line in f:
                tick=json.loads(line)
                for q in tick['p']:tracks.setdefault((day,tick['s'],q['w']),[]).append((tick['t'],geometry(q)))
    converted={}
    for key,rows in tracks.items():
        rows=sorted(rows,key=lambda r:r[0]);t=np.array([r[0] for r in rows]);B=np.stack([r[1] for r in rows]);feet=(und(B[:,4:6]*[1280,720])-a)@n/720;bottom=(und(np.column_stack([B[:,0],B[:,1]+B[:,3]/2])*[1280,720])-a)@n/720
        converted[key]=(t,np.column_stack([B,feet,bottom]))
    output={}
    for future in [0,10,30]:
        rows=[]
        for i in range(1628):
            m=matches.get(str(i));vec=[float(m is not None)]
            for past,ahead in [(3,min(future,3)),(10,min(future,10)),(30,future)]:
                if m:
                    t,B=converted[(m['day'],m['span'],m['track'])];dt=t-m['time'];use=(dt>=-past)&(dt<=ahead+1e-5);b=B[use];d=dt[use]
                else:b=np.empty((0,8));d=np.array([])
                if len(b):
                    # Height is only a reliability cue, never a hard side override.
                    tall=b[:,3]>=np.quantile(b[:,3],.8);high=b[tall];j=np.argmax(b[:,3]);k=np.argmin(np.abs(d))
                    vec.extend([min(len(b)/(12.5*(past+ahead)),1),d[0]/30,d[-1]/30,*b[k],*b.mean(0),*b.std(0),*np.quantile(b,[.1,.5,.9],axis=0).ravel(),*high.mean(0),*b[j],d[j]/30,*b[-1]-b[0]])
                else:vec.extend(np.zeros(76))
            rows.append(vec)
        E=np.array(rows,np.float32);assert E.shape==(1628,229),E.shape;assert np.isfinite(E).all();output[future]=E;np.save(R/f'diagnostics/owner_track_{future}_features.npy',E)
    return output

if __name__=='__main__':
    start=time.perf_counter();E=features();z=np.load(R/'io_cam1.npz');y=z['y'];days=z['day'];v=np.load(R/'diagnostics/original_labels.npy')!=0;A=np.load(R/'diagnostics/aligned_features.npy')[:,:1584];G=np.column_stack([np.load(R/'diagnostics/aligned_geometry.npy'),A[:,:39],A[:,234:360],np.load(R/'diagnostics/owner_boundary_features.npy')]);P={f'{name}_{f}':np.full((len(y),3),np.nan) for name in ['context','geometry'] for f in E}
    with threadpool_limits(limits=2):
        for d in np.unique(days):
            tr=v&(days!=d);te=days==d
            for future,extra in E.items():
                for name,base in [('context',A),('geometry',G)]:
                    B=np.column_stack([base,extra]);model=ExtraTreesClassifier(n_estimators=160,max_features=.7,random_state=19,n_jobs=2) if name=='context' else HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19)
                    P[f'{name}_{future}'][te]=model.fit(B[tr],y[tr]).predict_proba(B[te])
            print('track completed',d,'elapsed',round(time.perf_counter()-start),flush=True)
    report={}
    for name,p in P.items():
        report[name]=metrics(p,y,v,days);np.savez(R/f'diagnostics/owner_track_{name}.npz',probabilities=p,completed_days=np.unique(days));print(name,report[name]['wrong'],report[name]['gross'],report[name]['gross_indices'],flush=True)
    (R/'owner_track_report.json').write_text(json.dumps({'models':report,'seconds':time.perf_counter()-start,'future_seconds':[0,10,30],'live_validated':False},indent=2))
