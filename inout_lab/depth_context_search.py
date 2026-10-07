"""Cached monocular-depth ablation. Whole-day exclusion, all three classes.
Depth values use scene-aligned relative scale, not metric distance. The cache
uses teacher track IDs and student masks: this is not yet a streaming API.
"""
from pathlib import Path
import gzip,json
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier,HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parent

def build_features():
    matches=json.loads((ROOT/'track_context_match_report.json').read_text())['matches']
    tracks={}
    for path in (ROOT/'diagnostics/depth').glob('*.depth.json.gz'):
        day=path.name.split('_')[0]
        for key,values in json.load(gzip.open(path,'rt')).items():
            span,t,track=key.split('|');tracks.setdefault((day,int(span),int(track)),[]).append((float(t),values))
    tracks={key:(np.array([r[0] for r in sorted(rows)]),np.array([[np.nan if v is None else v for v in r[1]] for r in sorted(rows)])) for key,rows in tracks.items()}
    D=np.zeros((1628,42),np.float32);coverage=[]
    for ix,m in matches.items():
        i=int(ix);key=(m['day'],m['span'],m['track'])
        if key not in tracks:continue
        times,values=tracks[key];t=m['time'];j=np.searchsorted(times,t+1e-5,side='right')-1
        if j<0 or t-times[j]>.17:continue
        v=values[j];flags=np.isfinite(v);v=np.nan_to_num(v)
        extra=np.array([v[0]-v[2],v[1]-v[2],v[0]-v[3],v[1]-v[3],v[1]-v[0],v[2]-v[3],v[0]/(abs(v[2])+.01),v[0]/(abs(v[3])+.01)])
        row=[1.,t-times[j],*v,*flags.astype(float),*extra]
        for seconds in [1.,3.]:
            window=values[(times>=t-seconds)&(times<=t+1e-5)]
            for col in range(4):
                vv=window[:,col];vv=vv[np.isfinite(vv)]
                row.extend([float(vv.mean()) if len(vv) else 0.,float(vv.std()) if len(vv) else 0.,float(len(vv)>0)])
        D[i]=row;coverage.append(i)
    assert np.isfinite(D).all()
    np.save(ROOT/'diagnostics/depth_context_features.npy',D)
    return D,coverage

if __name__=='__main__':
    D,covered=build_features();print('depth matched',len(covered),'of1628',flush=True)
    z=np.load(ROOT/'io_cam1.npz');y=z['y'];days=z['day'];valid=np.load(ROOT/'diagnostics/original_labels.npy')!=0
    A=np.column_stack([np.load(ROOT/f'diagnostics/{n}_features.npy') for n in ['rich','head','context']])
    P={k:np.zeros((len(y),3)) for k in ['forest','boost']}
    with threadpool_limits(limits=2):
        for day in np.unique(days):
            tr=valid&(days!=day);te=days==day;B=np.column_stack([A,D])
            models={'forest':ExtraTreesClassifier(n_estimators=160,max_features=.7,random_state=19,n_jobs=2),'boost':HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=10,l2_regularization=3,random_state=19)}
            for name,model in models.items():P[name][te]=model.fit(B[tr],y[tr]).predict_proba(B[te])
            print('depth evaluated',day,flush=True)
    P['ensemble']=(P['forest']+P['boost'])/2;report={}
    for name,p in P.items():
        pred=p.argmax(1);gross=valid&(y<2)&(pred<2)&(pred!=y);covered_valid=valid&(D[:,0]>0)
        report[name]={'defined':int(valid.sum()),'accuracy':float((pred[valid]==y[valid]).mean()),'wrong':int((valid&(pred!=y)).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'depth_covered':int(covered_valid.sum()),'covered_wrong':int((covered_valid&(pred!=y)).sum()),'covered_gross':int((covered_valid&gross).sum()),'doorway_recall':float((pred[y==2]==2).mean()),'hard_cases':{str(i):p[i].tolist() for i in [483,625,789,970,1149]}}
        np.savez(ROOT/f'diagnostics/depth_{name}_predictions.npz',probabilities=p)
    (ROOT/'depth_context_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
