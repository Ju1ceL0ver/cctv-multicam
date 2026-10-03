"""Cross-window continuity from shared raw observations and explicit human decisions.

Uncertain appearance matches remain proposals. Nothing here writes identity GT or
training labels. Node revisions prevent old approvals surviving changed evidence.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from threading import RLock
import numpy as np
from review_store import read_state, review_groups, fingerprint
from storage import read_json, atomic_json, file_lock
ROOT=Path(__file__).resolve().parent


def node_key(clip, label):return clip+':'+label


def _shot(cam, frame, box):
 """Where this person was in one exact frame, so the page can ask for a crop of them
 instead of the whole 2560x1440 picture. Display only: the box is rounded to whole
 pixels and never becomes annotation."""
 return {'cam':cam,'frame':int(frame),'box':[int(round(float(v))) for v in box]}


_node_cache = {}
_cache_lock = RLock()


def _video_nodes(folder, meta, state):
 """Build identity nodes from effective observations, never whole obsolete pieces."""
 from video_annotations import effective_decision
 start=datetime.fromisoformat(meta['start']).timestamp()
 with np.load(folder/'dets_yolo26x-seg.npz') as archive:
  dz={cam:archive[cam] for cam in ('cam1','cam2')}
 ep=folder/'emb_osnet_ain_x1_0_msmt17.npz'
 ez=None
 if ep.exists():
  with np.load(ep) as archive:ez={cam:archive[cam] for cam in ('cam1','cam2')}
 overrides=state['video'].get('observations',{})
 unlabelled={}
 for gi,g in enumerate(review_groups(folder,state)):
  for i in g['pieces']:unlabelled[i]='G%d'%gi
 buckets=defaultdict(list);known_labels=set();seen=set()
 def add(oid,cam,frame,piece,box,label,quality,det=None):
  if label=='?' or quality in ('false_positive','mixed'):return
  if label:
   bucket=label;known_labels.add(label)
  elif piece is not None:
   bucket=unlabelled.get(piece,'Gpiece%d'%piece)
  else:return
  buckets[bucket].append({'id':oid,'cam':cam,'frame':int(frame),'piece':piece,
                          'box':[float(x) for x in box],'det':det})
 for p in state['pieces']:
  cam=p['cam'];piece=p['piece']
  for di in p['dets']:
   oid='det:%s:%d'%(cam,di);seen.add(oid)
   item=overrides.get(oid,{})
   if item.get('deleted'):continue
   row=dz[cam][di];frame=int(round(float(row[0])*25))
   label,quality=effective_decision(state,piece,frame,oid)
   box=item.get('box',row[1:5])
   feature_index=di if np.allclose(box,row[1:5]) else None
   add(oid,cam,frame,piece,box,label,quality,feature_index)
 for oid,item in overrides.items():
  if oid in seen or item.get('deleted'):continue
  cam=item['cam'];frame=item['frame'];piece=item.get('piece')
  label,quality=effective_decision(state,piece,frame,oid)
  # New manual observations and previously orphaned detections have no safe cached descriptor.
  add(oid,cam,frame,piece,item['box'],label,quality)
 state_evidence=fingerprint(state);nodes=[]
 for label,items in buckets.items():
  items.sort(key=lambda x:(x['frame'],x['cam'],x['id']))
  ids=sorted({o['piece'] for o in items if o['piece'] is not None})
  obs={(o['cam'],int(round(start*25))+o['frame'],*(int(round(x)) for x in o['box'])) for o in items}
  first=start+min(o['frame'] for o in items)/25
  last=start+max(o['frame'] for o in items)/25
  feats=[];feature_groups=defaultdict(list);range_groups=defaultdict(list)
  for o in items:
   range_groups[(o['piece'],o['cam'])].append(o['frame'])
   if o['det'] is not None:feature_groups[(o['piece'],o['cam'])].append(o['det'])
  if ez is not None:
   for (piece,cam),indices in feature_groups.items():
    selected=indices[::max(1,len(indices)//16)][:16]
    feats.extend(ez[cam][selected].astype(float))
  feat=np.median(feats,axis=0) if feats else None
  if feat is not None:feat=feat/max(np.linalg.norm(feat),1e-8)
  ranges=[{'piece':piece,'cam':cam,'start_frame':min(frames),'end_frame':max(frames)}
          for (piece,cam),frames in range_groups.items()]
  evidence=hashlib.sha256(json.dumps([state_evidence,label,items,meta],sort_keys=True).encode()).hexdigest()[:16]
  representative=items[len(items)//2]
  nodes.append({'key':node_key(folder.name,label),'clip':folder.name,'label':label,'pieces':ids,
    'first':first,'last':last,'reviewed':label in known_labels,'evidence':evidence,
    'ranges':ranges,'observations':[o['id'] for o in items],
    'representative':{'cam':representative['cam'],'frame':representative['frame'],'observation_id':representative['id'],
                      'box':[int(round(float(v))) for v in representative['box']]},
    'first_shot':_shot(items[0]['cam'],items[0]['frame'],items[0]['box']),
    'last_shot':_shot(items[-1]['cam'],items[-1]['frame'],items[-1]['box']),
    'cams':sorted({o['cam'] for o in items}),
    '_obs':obs,'_feat':feat})
 return nodes


def clip_nodes(folder, meta):
 names=('review_state.json','gt_manual.json','pieces_yolo26x-seg.json','groups_yolo26x-seg.json',
        'meta_yolo26x-seg.json','dets_yolo26x-seg.npz','emb_osnet_ain_x1_0_msmt17.npz')
 signature=tuple((name,(folder/name).stat().st_size,(folder/name).stat().st_mtime_ns)
                 for name in names if (folder/name).exists())
 key=str(folder)
 with _cache_lock:cached=_node_cache.get(key)
 if cached is not None and cached[0]==signature:return cached[1]
 nodes=[]
 state=read_state(folder)
 if state.get('video'):
  nodes=_video_nodes(folder,meta,state)
  with _cache_lock:
   if key not in _node_cache and len(_node_cache)>=64:_node_cache.pop(next(iter(_node_cache)))
   _node_cache[key]=(signature,nodes)
  return nodes
 if not state['pieces']:return []
 human_labels=set(state['labels'].values())-{'?'}
 buckets=defaultdict(list)
 unlabelled={}
 for gi,g in enumerate(review_groups(folder,state)):
  for i in g['pieces']:unlabelled[i]='G%d'%gi
 for p in state['pieces']:
  lab=state['labels'].get(str(p['piece']))
  if lab=='?' or state['quality'].get(str(p['piece'])) in ('false_positive','mixed'):continue
  buckets[lab or unlabelled[p['piece']]].append(p)
 start=datetime.fromisoformat(meta['start']).timestamp()
 ep=folder/'emb_osnet_ain_x1_0_msmt17.npz'
 with np.load(folder/'dets_yolo26x-seg.npz') as archive:
  dz={cam:archive[cam] for cam in ('cam1','cam2')}
 ez=None
 if ep.exists():
  with np.load(ep) as archive:ez={cam:archive[cam] for cam in ('cam1','cam2')}
 for label,pieces in buckets.items():
  obs=set();feats=[];first=1e30;last=-1e30;ids=[];head=None;tail=None
  for p in pieces:
   ids.append(p['piece']);d=dz[p['cam']][p['dets']]
   lo=int(d[:,0].argmin());hi=int(d[:,0].argmax())
   if start+float(d[lo,0])<first:
    first=start+float(d[lo,0]);head=_shot(p['cam'],round(float(d[lo,0])*25),d[lo,1:5])
   if start+float(d[hi,0])>last:
    last=start+float(d[hi,0]);tail=_shot(p['cam'],round(float(d[hi,0])*25),d[hi,1:5])
   for row in d:
    obs.add((p['cam'],int(round((start+float(row[0]))*25)),*(int(round(x)) for x in row[1:5])))
   if ez is not None:
    ii=p['dets'][::max(1,len(p['dets'])//16)]
    feats.extend(ez[p['cam']][ii].astype(float))
  feat=np.median(feats,axis=0) if feats else None
  if feat is not None:feat=feat/max(np.linalg.norm(feat),1e-8)
  evidence=hashlib.sha256(json.dumps([label,pieces,{str(i):state['quality'].get(str(i)) for i in ids},meta],sort_keys=True).encode()).hexdigest()[:16]
  nodes.append({'key':node_key(folder.name,label),'clip':folder.name,'label':label,'pieces':ids,
                'first':first,'last':last,'reviewed':label in human_labels,'evidence':evidence,
                'first_shot':head,'last_shot':tail,
                'cams':sorted({p['cam'] for p in pieces}),
                '_obs':obs,'_feat':feat})
 # One version per clip, bounded number of clips. Never cache the dense embeddings.
 with _cache_lock:
  if key not in _node_cache and len(_node_cache)>=64:_node_cache.pop(next(iter(_node_cache)))
  _node_cache[key]=(signature,nodes)
 return nodes


NEIGHBOUR_WINDOW=120.0   # how far apart two fragments may be and still be asked about
_day_cache={}


def _compute(day, root):
 """The day's grouping, plus every pair of fragments close enough in time to be the same
 person. Both come out of one pass and are cached together: an answer used to rebuild the
 whole day twice, and a page that only reads it rebuilt it again on every request."""
 root=Path(root);nodes=[]
 for folder in sorted((root/'data/raw_clips').glob('c*')):
  meta=read_json(folder/'meta_yolo26x-seg.json',{})
  if meta.get('day')==day:nodes.extend(clip_nodes(folder,meta))
 decisions=read_json(root/'data/day_review'/('%s.json'%day),{'revision':0,'links':[]})
 # Node evidence is a content hash, so this signature changes exactly when an answer would.
 signature=(tuple(sorted((n['key'],n['evidence']) for n in nodes)),
            hashlib.sha256(json.dumps(decisions,sort_keys=True,default=str).encode()).hexdigest())
 cache_key=(day,str(root))
 with _cache_lock:cached=_day_cache.get(cache_key)
 if cached is not None and cached[0]==signature:return cached[1]
 by={n['key']:n for n in nodes};parents={k:k for k in by};members={k:{k} for k in by}
 accepted=[];proposals=[];neighbours=[]
 def find(k):
  while parents[k]!=k:k=parents[k]
  return k
 def merge(a,b,source):
  ra,rb=find(a),find(b)
  if ra==rb:return True
  combo=members[ra]|members[rb]
  # Never combine separately reviewed people from the same clip.
  assigned=defaultdict(set)
  for key in combo:
   n=by[key]
   if n['reviewed']:assigned[n['clip']].add(n['label'])
  if any(len(x)>1 for x in assigned.values()):return False
  for decision in decisions['links']:
   if valid(decision) and decision['decision']=='different' and decision['a'] in combo and decision['b'] in combo:return False
  winner,loser=sorted((ra,rb));parents[loser]=winner;members[winner]=combo
  accepted.append({'a':a,'b':b,'source':source});return True
 def valid(r):
  return r['a'] in by and r['b'] in by and r.get('evidence_a')==by[r['a']]['evidence'] and r.get('evidence_b')==by[r['b']]['evidence']
 valid_decisions={(r['a'],r['b']):r for r in decisions['links'] if valid(r)}
 for r in valid_decisions.values():
  if r['decision']=='same':merge(r['a'],r['b'],'human')
 for i,a in enumerate(nodes):
  for b in nodes[i+1:]:
   gap=max(0,max(a['first'],b['first'])-min(a['last'],b['last']))
   if gap>NEIGHBOUR_WINDOW:continue
   pair=tuple(sorted((a['key'],b['key'])))
   distance=(None if a['_feat'] is None or b['_feat'] is None
             else round(float(1-a['_feat']@b['_feat']),4))
   answered=valid_decisions.get(pair)
   # Kept whatever the pair's fate below: the long-visit queue needs the ones the
   # automatic rules skip -- above all two machine groups inside one ten-minute window,
   # which is where the measurements say a long visit actually breaks.
   neighbours.append({'a':pair[0],'b':pair[1],'gap_s':round(gap,2),'distance':distance,
                      'same_clip':a['clip']==b['clip'],'decision':answered['decision'] if answered else None,
                      'evidence_a':by[pair[0]]['evidence'],'evidence_b':by[pair[1]]['evidence']})
   if a['clip']==b['clip'] or gap>90 or answered is not None:continue
   common=a['_obs']&b['_obs']
   frames=len({(o[0],o[1]) for o in common})
   if frames>=2 and merge(a['key'],b['key'],'shared_raw_frames'):
    continue
   if distance is None:continue
   proposals.append({'a':pair[0],'b':pair[1],'distance':distance,'gap_s':round(gap,2),
                     'evidence_a':by[pair[0]]['evidence'],'evidence_b':by[pair[1]]['evidence']})
 proposals.sort(key=lambda r:(r['distance'],r['gap_s']))
 # At most three alternatives per endpoint keeps the review queue finite.
 count=defaultdict(int);queue=[]
 for r in proposals:
  if find(r['a'])==find(r['b']):continue
  if count[r['a']]>=3 or count[r['b']]>=3:continue
  queue.append(r);count[r['a']]+=1;count[r['b']]+=1
 components=defaultdict(list)
 for n in nodes:components[find(n['key'])].append(n['key'])
 visits=[{'id':day+'-'+hashlib.sha256(k.encode()).hexdigest()[:10],'nodes':sorted(v),
          'first':min(by[x]['first'] for x in v),'last':max(by[x]['last'] for x in v)} for k,v in components.items()]
 public=[{k:v for k,v in n.items() if not k.startswith('_')} for n in nodes]
 result={'day':day,'revision':decisions['revision'],'can_undo':decisions.get('undo_revision') is not None,'nodes':public,'visits':visits,'links':accepted,
         'proposals':queue,'stale_decisions':sum(not valid(r) for r in decisions['links'])}
 atomic_json(root/'data/day_visits'/('%s.json'%day),result)
 with _cache_lock:
  if cache_key not in _day_cache and len(_day_cache)>=8:_day_cache.pop(next(iter(_day_cache)))
  _day_cache[cache_key]=(signature,(result,neighbours))
 return result,neighbours


def build_day(day, root=ROOT):
 return _compute(day,root)[0]


def neighbours(day, root=ROOT):
 """Every pair of fragments within NEIGHBOUR_WINDOW seconds of each other, answered or not."""
 return _compute(day,root)[1]


def decide(day, body, root=ROOT):
 root=Path(root);path=root/'data/day_review'/('%s.json'%day)
 with file_lock(path.with_suffix('.lock')):
  current=read_json(path,{'revision':0,'links':[]})
  if body.get('revision')!=current['revision']:raise ValueError('Связи изменились, обновите список')
  if body.get('action')=='undo':
   undo=current.get('undo_revision')
   if undo is None:raise ValueError('Нет связи для отмены')
   previous=read_json(path.parent/'history'/('%s_%08d.json'%(day,undo)))
   if previous is None:raise ValueError('Нет связи для отмены')
   atomic_json(path.parent/'history'/('%s_%08d.json'%(day,current['revision'])),current)
   previous['revision']=current['revision']+1;previous['undo_revision']=None;atomic_json(path,previous)
   return build_day(day,root)
  fresh=build_day(day,root);by={n['key']:n for n in fresh['nodes']}
  a,b=sorted((body['a'],body['b']))
  if a==b or a not in by or b not in by or body['decision'] not in ('same','different','unsure'):raise ValueError('Неверная связь')
  evidence={body['a']:body.get('evidence_a'),body['b']:body.get('evidence_b')}
  if evidence[a]!=by[a]['evidence'] or evidence[b]!=by[b]['evidence']:raise ValueError('Отрезки изменились; проверьте новую версию')
  if body['decision']=='same':
   members=set()
   for visit in fresh['visits']:
    if a in visit['nodes'] or b in visit['nodes']:members.update(visit['nodes'])
   confirmed=defaultdict(set)
   for key in members:
    if by[key]['reviewed']:confirmed[by[key]['clip']].add(by[key]['label'])
   if any(len(v)>1 for v in confirmed.values()):
    raise ValueError('Связь объединит разные проверенные личности внутри одного клипа. Сначала исправьте их метки.')
   for link in current['links']:
    x,y=link['a'],link['b']
    if (x,y)==(a,b) or link['decision']!='different':continue
    if (x in members and y in members and link.get('evidence_a')==by[x]['evidence']
        and link.get('evidence_b')==by[y]['evidence']):
     raise ValueError('Связь противоречит сохранённому ответу «разные люди». Сначала исправьте этот ответ.')

  atomic_json(path.parent/'history'/('%s_%08d.json'%(day,current['revision'])),current)
  current['links']=[r for r in current['links'] if (r['a'],r['b'])!=(a,b)]
  link={'a':a,'b':b,'decision':body['decision'],'evidence_a':evidence[a],'evidence_b':evidence[b]}
  # An answer given with someone else's proposal on screen is recorded as such.
  if body.get('assisted') in ('claude',):link['assisted']=body['assisted']
  current['links'].append(link)
  current['undo_revision']=current['revision']
  current['revision']+=1;atomic_json(path,current)
 return build_day(day,root)


if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('day');p.add_argument('--root',default=str(ROOT));a=p.parse_args()
 result=build_day(a.day,a.root);print(json.dumps({k:len(result[k]) for k in ('nodes','visits','links','proposals')}))
