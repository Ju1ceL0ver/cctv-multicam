"""Diagnostic physical reliability branch; all three classes and rows retained.
Train-only visible feet calibrate shape/margin; day-excluded predictions are the
fallback. Teacher IDs are offline: this does not prove streaming ID continuity.
"""
from pathlib import Path
import numpy as np,json,gzip
from door_plane_search import fit_plane
root=Path(__file__).parent

def shape(box,foot):
 width=box[:,2]*1280;height=box[:,3]*720
 return height/np.maximum(width,1), (box[:,1]+box[:,3]/2-foot[:,1])/np.maximum(box[:,3],.001)

def choose_side(times,boxes,feet,score,t,plane,params,buffered):
 use=(times>=t-10)&(times<=t+(3 if buffered else 0));times,boxes,feet,score=times[use],boxes[use],feet[use],score[use]
 if not len(times):return None
 asp,gap=shape(boxes,feet);valid=(asp>=params['aspect_min'])&(gap<=params['gap_max'])&(gap>=-.05)&(score>=params['score_min'])
 if not valid.any():return None
 # Reject shrunk upper-body masks relative to credible nearby full-body views.
 reference=np.quantile(boxes[valid,3],.9);valid&=boxes[:,3]>=reference*params['height_ratio_min']
 coef,bias,norm,_=plane;distance=(feet@coef+bias)/norm;valid&=np.abs(distance)>=params['margin_pixels']
 available=np.where(valid)[0]
 if len(available)<5:return None
 nearest=available[np.argsort(np.abs(times[available]-t),kind='stable')[:5]];sign=distance[nearest]>0
 if not np.all(sign==sign[0]):return None
 return int(sign[0])

if __name__=='__main__':
 z=np.load(root/'io_cam1.npz');g=z['G'][:,1:];y=z['y'];days=z['day'];v=np.load(root/'diagnostics/original_labels.npy')!=0;p=np.load(root/'diagnostics/pose_features.npy');a,b=p[:,45:48],p[:,48:51];ankles=(a[:,:2]+b[:,:2])/2;visible=(a[:,2]>.7)&(b[:,2]>.7)&(np.abs(ankles[:,1]-g[:,7])<.03)&(np.abs(ankles[:,0]-g[:,6])<.05);matches=json.loads((root/'track_context_match_report.json').read_text())['matches'];tracks={}
 for day in ['20260917','20260918','20260919']:
  with gzip.open(root/('diagnostics/temporal/'+day+'_sam31.jsonl.gz'),'rt') as f:
   next(f)
   for line in f:
    tick=json.loads(line)
    for q in tick['p']:
     if q['foot'] is not None:tracks.setdefault((day,tick['s'],q['w']),[]).append((tick['t'],np.asarray(q['box'])*[1,1248/1224,1,1248/1224],np.asarray(q['foot'])/[2176,1224],q['s']))
 tracks={k:(np.array([r[0] for r in rows]),np.stack([r[1] for r in rows]),np.stack([r[2] for r in rows]),np.array([r[3] for r in rows])) for k,rows in tracks.items()};base=(np.load(root/'diagnostics/context_predictions.npz')['probabilities']+np.load(root/'diagnostics/context_boost.npz')['probabilities'])/2;mlp=np.load(root/'diagnostics/expert_mlp_3.0.npz')['probabilities'];sources={'baseline':base,'mlp':mlp,'equal_blend':(base+mlp)/2};calibration={};report={}
 for buffered in [False,True]:
  overrides=np.full(len(y),-1);cal={}
  for day in np.unique(days):
   train=v&visible&(days!=day);plane=fit_plane(g,y,v,visible,days,[day]);boxes=np.column_stack([(g[:,:2]+g[:,2:4])/2,g[:,4:6]]);aspect,gap=shape(boxes,g[:,6:8]);coef,bias,norm,_=plane;disagreement=np.abs((ankles-g[:,6:8])@coef)/norm
   params={'aspect_min':float(np.quantile(aspect[train],.05)),'gap_max':float(np.quantile(gap[train],.95)),'margin_pixels':float(np.quantile(disagreement[train],.9)+5),'score_min':0.,'height_ratio_min':0.}
   scores=[];ratios=[]
   for i in np.where(train)[0]:
    match=matches.get(str(int(i)))
    if match is None:continue
    times,bs,fs,sc=tracks[(match['day'],match['span'],match['track'])];t=match['time'];use=(times>=t-10)&(times<=t+(3 if buffered else 0));asp,gp=shape(bs,fs);credible=use&(asp>=params['aspect_min'])&(gp<=params['gap_max']);nearest=np.argmin(np.abs(times-t));scores.append(sc[nearest])
    if credible.any():ratios.append(bs[nearest,3]/max(.001,np.quantile(bs[credible,3],.9)))
   params['score_min']=float(np.quantile(scores,.05)) if scores else 0.;params['height_ratio_min']=min(1.,float(np.quantile(ratios,.05))) if ratios else .8;cal[str(day)]=params
   for i in np.where(days==day)[0]:
    match=matches.get(str(int(i)))
    if match is None:continue
    state=choose_side(*tracks[(match['day'],match['span'],match['track'])],match['time'],plane,params,buffered)
    if state is not None:overrides[i]=state
  mode='buffered' if buffered else 'causal';calibration[mode]=cal
  for name,P in sources.items():
   pred=P.argmax(1);apply=(pred<2)&(overrides>=0);changed=apply&(pred!=overrides);pred[apply]=overrides[apply];gross=v&(y<2)&(pred<2)&(pred!=y);key=mode+'_'+name;report[key]={'accuracy':float((pred[v]==y[v]).mean()),'wrong':int(((pred!=y)&v).sum()),'gross':int(gross.sum()),'gross_indices':np.where(gross)[0].tolist(),'changed':int((changed&v).sum()),'fixed':int((changed&v&(pred==y)).sum()),'regressed':int((changed&v&(P.argmax(1)==y)).sum()),'doorway_recall':float((pred[y==2]==2).mean()),'future_seconds':3 if buffered else 0};np.savez(root/('diagnostics/reliable_'+key+'.npz'),predictions=pred,physical_side=overrides);print('RELIABLE',key,report[key],flush=True)
 (root/'reliable_foot_report.json').write_text(json.dumps({'results':report,'calibration':calibration},indent=2))
