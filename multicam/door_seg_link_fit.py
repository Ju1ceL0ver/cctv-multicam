"""Fit and replay an experimental association classifier on owner TRAIN caches.

Usage: door_seg_link_fit.py [--cache PATH] [--held-tag TAG] [--held-start N] [--held-end N]
The held training interval is development data; no GPU, event TEST or deployment.
"""
import argparse,json,pickle,time,numpy as np
from pathlib import Path
from dataclasses import replace
from sklearn.ensemble import RandomForestClassifier
import door_seg as S
from door_seg_track import Tracker,Config,centre
from door_seg_link_model import features,LearnedTracker
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cache',type=Path,default=S.ROOT/'data/quality_backups/doorseg_quick')
parser.add_argument('--held-tag',default='door_20260918_18720')
parser.add_argument('--held-start',type=int,default=3500)
parser.add_argument('--held-end',type=int,default=4100)
args=parser.parse_args()
if args.held_end<=args.held_start:parser.error('empty held interval')
r=args.cache;datasets={};xx=[];yy=[];split=[]
for f in sorted(r.glob('door_*_appearance.pkl')):
 d=pickle.load(f.open('rb'))
 if not d['state'].get('done') or d['state'].get('role')!='train':
  raise ValueError('only completed owner TRAIN caches are accepted: %s' % f)
 rows=[];past=[]
 for tick,pp,gg in d['records']:
  up=lambda z:(np.unpackbits(z[0],count=184*320).reshape(184,320).astype(bool),*z[1:])
  p=[up(z) for z in pp if z[2]>=.4];g=[up(z) for z in gg];mm=S._match([z[0] for z in p],[z[0] for z in g]);labels={i:g[j][2] for i,j,_ in mm};rows.append((tick,p,g))
  for ta,pa,la in (past[-3:] if tick%3==0 else []):
   gap=tick-ta
   if gap>3:continue
   for i,a in enumerate(pa):
    if i not in la:continue
    for j,b in enumerate(p):
     if j not in labels:continue
     h=max(16.,min(a[1][3]-a[1][1],b[1][3]-b[1][1]))
     if np.linalg.norm(centre(a[1])-centre(b[1]))>.8*h:continue
     ft=features(a,b,gap)
     if ft[1]>.8 and max(ft[4:6])<.1:continue
     xx.append(ft);yy.append(la[i]==labels[j]);split.append(d['tag']==args.held_tag and (args.held_start<=tick<args.held_end or args.held_start<=ta<args.held_end))
  past.append((tick,p,labels));past=past[-3:]
 datasets[d['tag']]=rows
 print('loaded',d['tag'],'pairs',len(yy),flush=True)
x=np.array(xx);y=np.array(yy);val=np.array(split);print('pairs',len(y),'positive',sum(y),'held',sum(val),flush=True)
if not val.any() or val.all():raise ValueError('need disjoint fit and holdout pairs')
m=RandomForestClassifier(n_estimators=48,max_depth=6,min_samples_leaf=15,class_weight='balanced',random_state=1,n_jobs=2).fit(x[~val],y[~val]);m.n_jobs=1;pr=m.predict_proba(x[val])[:,1]
report={'held_tag':args.held_tag,'held_interval':[args.held_start,args.held_end],'train_tags':list(datasets),'fit_pairs':int((~val).sum()),'pairs':len(y),'positives':int(y.sum()),'held_pairs':int(val.sum()),'pair_validation':{}}
for threshold in (.7,.8,.9):
 q=pr>=threshold;yt=y[val];report['pair_validation'][str(threshold)]={'tp':int((q&yt).sum()),'fp':int((q&~yt).sum()),'fn':int((~q&yt).sum())}
report['replay']={}
for name,make in [('b2',lambda:Tracker()),('learn_07',lambda:LearnedTracker(m,threshold=.7)),('learn_08',lambda:LearnedTracker(m,threshold=.8)),('learn_09',lambda:LearnedTracker(m,threshold=.9))]:
 stats={}
 for tag,rows in datasets.items():
  tr=make();prev={};owners={};last=None;total=correct=wrong=0;t0=time.perf_counter()
  for tick,p,g in rows:
   if last is not None and tick-last!=1:tr=make();prev={};owners={}
   last=tick;linked=tr.update(tick,p);mm=S._match([z[0] for z in p],[z[0] for z in g]);now={g[j][2]:linked[i][0] for i,j,_ in mm}
   if tag==args.held_tag and args.held_start<=tick<args.held_end:
    for who,pid in now.items():
     if who in prev:total+=1;correct+=int(prev[who]==pid)
     wrong+=int(pid in owners and owners[pid]!=who)
   for who,pid in now.items():owners.setdefault(pid,who)
   prev=now
  stats[tag]={'total':total,'continued':correct,'broken':total-correct,'wrong':wrong,'s':time.perf_counter()-t0};print(name,tag,stats[tag],flush=True)
 report['replay'][name]=stats
with (r/'link_model.pkl').open('wb') as f:pickle.dump(m,f)
(r/'link_model_report.json').write_text(json.dumps(report,indent=2));print(report['pair_validation'],flush=True)
