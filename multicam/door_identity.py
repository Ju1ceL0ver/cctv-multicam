"""One persistent person ID for door crossings, independent of tracker IDs.

The automatic matcher uses appearance and chronology only. Owner labels are read
only by `bench`, never by `match`. Ambiguous exits remain unresolved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from storage import atomic_json,read_json

ROOT=Path(__file__).resolve().parent


def load_events(day,root=ROOT):
    out=[];path=root/'data/live/events_cam1.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
        try:
            e=json.loads(line)
            if e.get('engine')!='doorseg_c+zones' or not e.get('clock','').replace('-','').startswith(day):continue
            from live_events import key_of
            out.append(dict(e,key=key_of(e),answer_key='%.1f'%e['t']))
        except (ValueError,KeyError,TypeError):continue
    return sorted(out,key=lambda e:(e['t'],e['key']))


def prepare(day,root=ROOT,mask_snapshots=False):
    from door_identity_views import from_legacy,snapshot_views
    events=load_events(day,root);answers=read_json(root/'data/visits'/day/'people_answers.json',{})
    # Evaluation preparation is restricted to the owner's labelled events.
    chosen=[e for e in events if e['answer_key'] in answers and answers[e['answer_key']]!='x']
    if not chosen:raise ValueError('No labelled real crossings for '+day)
    out=root/'data/door_identity'/day;out.mkdir(parents=True,exist_ok=True)
    segmenter=None
    if mask_snapshots:
        if not ('21:00'<=time.strftime('%H:%M') or time.strftime('%H:%M')<'09:30'):
            raise RuntimeError('GPU snapshot preparation is night-only')
        from ultralytics import YOLO
        segmenter=YOLO(r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models\yolo26x-seg.pt')
    prepared=[]
    for e in chosen:
        folder=out/'views'/e['key'];manifest=read_json(folder/'manifest.json',{})
        clip=root/'data/live/clips_seg'/day/(e['key']+'.mp4')
        live=root/'data/live/event_views'/day/e['key']
        if manifest.get('source') in ('tape_mask_v1','live_bank_v1'):pass
        elif (live/'manifest.json').exists():
            # 20.10: immutable best views copied by the live door at the crossing
            import shutil;shutil.copytree(live,folder,dirs_exist_ok=True);manifest=read_json(folder/'manifest.json',{})
        elif clip.exists():manifest=from_legacy(clip,folder,e)
        else:manifest=snapshot_views(e,folder,segmenter)
        prepared.append(dict(e,manifest=str(folder/'manifest.json'),available_views=len(manifest.get('views',[]))))
    cutoff=max(1,int(len(prepared)*.7))
    index=dict(day=day,events=prepared,split=dict(policy='chronological 70/30; choose parameters only on train',
               train_keys=[e['key'] for e in prepared[:cutoff]],test_keys=[e['key'] for e in prepared[cutoff:]]))
    path=out/'index.json'
    previous=read_json(path,{})
    if previous.get('split') and previous['split']!=index['split']:
        raise RuntimeError('Existing identity evaluation split changed; do not silently reuse it')
    atomic_json(path,index)
    print('prepared',len(prepared),'events',sum(e['available_views'] for e in prepared),'views',flush=True)
    return index


def embed(day,root=ROOT):
    import cv2,torch,track_emb
    torch.set_num_threads(int(os.environ.get('RA_ID_THREADS','3')))
    folder=root/'data/door_identity'/day;index=read_json(folder/'index.json',{})
    cache=folder/'features.npz';have=dict(np.load(cache)) if cache.exists() else {}
    todo={}
    for e in index['events']:
        manifest=read_json(e['manifest'],{})
        for view in manifest.get('views',[]):
            path=Path(e['manifest']).parent/view['path'];payload=path.read_bytes();digest=hashlib.sha256(payload).hexdigest()
            if digest!=view['sha256']:raise ValueError('Identity view content changed: '+str(path))
            if 'c_'+digest not in have:todo[digest]=path
    if todo:
        teachers=track_emb.teachers('cpu');items=list(todo.items())
        for start in range(0,len(items),8):
            batch=items[start:start+8];images=[cv2.imread(str(p)) for _,p in batch]
            if any(im is None for im in images):raise ValueError('Unreadable identity crop')
            cloth,shape=teachers(images)
            for (digest,_),c,s in zip(batch,cloth,shape):
                have['c_'+digest]=np.asarray(c,np.float32)/max(float(np.linalg.norm(c)),1e-9)
                have['s_'+digest]=np.asarray(s,np.float32)/max(float(np.linalg.norm(s)),1e-9)
            tmp=cache.with_suffix('.tmp.npz');np.savez_compressed(tmp,**have);os.replace(tmp,cache)
            print('embedded',min(start+8,len(items)),'/',len(items),flush=True)
    return have


def event_vectors(events,features,variant='raw',cloth_weight=.5):
    result={}
    for e in events:
        manifest=read_json(e['manifest'],{});views=manifest.get('views',[])
        if any('variant' in v for v in views):views=[v for v in views if v.get('variant')==variant]
        vectors=[]
        for view in views:
            digest=view['sha256']
            if 'c_'+digest in features and 's_'+digest in features:
                vectors.append(np.r_[np.sqrt(cloth_weight)*features['c_'+digest],np.sqrt(1-cloth_weight)*features['s_'+digest]])
        if vectors:result[e['key']]=np.asarray(vectors)
    return result


def similarity(a,b):
    matrix=a@b.T
    return float((np.median(matrix.max(axis=0))+np.median(matrix.max(axis=1)))/2)


def assign(events,vectors,threshold=.75,margin=.04):
    """Chronological, deterministic IDs. Exit links require a clear appearance match.

    Person prototypes stay anchored to the original entry, so a single lucky
    view cannot progressively absorb a chain of strangers.
    """
    people=[];assignments=[]
    for event in events:
        key=event['key'];v=vectors.get(key);scores=[]
        if v is not None:
            for i,person in enumerate(people):
                if person['anchor'] is None:continue
                # Two different people crossing together cannot share a person ID.
                if abs(event['t']-person['last_t'])<1.0 and str(event.get('w'))!=str(person['last_w']):continue
                if event['kind']=='out' and not person['inside'] and event['t']-person['last_t']>30:continue
                score=similarity(v,person['anchor'])
                scores.append((score,i))
        scores.sort(reverse=True)
        best=scores[0] if scores else (-1,None)
        gap=best[0]-(scores[1][0] if len(scores)>1 else -1)
        accepted=best[1] is not None and best[0]>=threshold and gap>=margin
        pid=best[1] if accepted else None
        status='matched' if accepted else 'pending'
        entry_key=None
        if pid is None and event['kind']=='in':
            pid=len(people);people.append(dict(anchor=v,last_t=event['t'],last_w=event.get('w'),inside=True,last_entry=key))
            status='new' if v is not None else 'missing_views'
        elif pid is not None:
            person=people[pid]
            if event['kind']=='out':entry_key=person['last_entry']
            else:person['last_entry']=key
            person.update(last_t=event['t'],last_w=event.get('w'),inside=event['kind']=='in')
        assignments.append(dict(event=key,t=event['t'],kind=event['kind'],person=None if pid is None else 'P%04d'%(pid+1),
                                entry_event=entry_key,status=status,score=None if not scores else round(best[0],5),
                                margin=None if not scores else round(gap,5)))
    return assignments


def metrics(events,assignments,answers,selected_keys):
    truth={e['key']:answers[e['answer_key']] for e in events};by={a['event']:a for a in assignments}
    eligible=paired=correct=wrong=0;previous=set();clusters={}
    for event in events:
        owner=truth[event['key']];a=by[event['key']]
        if event['key'] in selected_keys and event['kind']=='out':
            eligible+=int(owner in previous)
            if a['entry_event'] is not None:
                paired+=1
                if truth.get(a['entry_event'])==owner:correct+=1
                else:wrong+=1
        if event['kind']=='in':previous.add(owner)
        if a['person'] is not None:clusters.setdefault(a['person'],set()).add(owner)
    return dict(eligible_exits=eligible,paired_exits=paired,correct=correct,wrong=wrong,
                precision=correct/paired if paired else None,recall=correct/eligible if eligible else None,
                mixed_person_ids=sum(len(x)>1 for x in clusters.values()))


def bench(day,root=ROOT,final=False):
    folder=root/'data/door_identity'/day;index=read_json(folder/'index.json',{})
    features=dict(np.load(folder/'features.npz'));events=index['events']
    answers=read_json(root/'data/visits'/day/'people_answers.json',{})
    train_keys=set(index['split']['train_keys']);train=[e for e in events if e['key'] in train_keys]
    chosen_path=folder/'chosen.json'
    if not final:
        results=[]
        for variant in ('raw','masked'):
            for weight in (0.,.5,1.):
                vectors=event_vectors(train,features,variant,weight)
                for threshold in (.6,.65,.7,.75,.8,.85,.9):
                    for margin in (0.,.03,.06,.1):
                        assignment=assign(train,vectors,threshold,margin)
                        score=metrics(train,assignment,answers,train_keys)
                        results.append(dict(variant=variant,cloth_weight=weight,threshold=threshold,margin=margin,
                                            available_events=len(vectors),**score))
        safe=[r for r in results if r['paired_exits']>=3 and r['wrong']==0 and r['mixed_person_ids']==0]
        if not safe:
            atomic_json(folder/'train_results.json',dict(results=results,status='no_safe_candidate'))
            raise RuntimeError('No measured safe identity matcher; keep automatic linking off')
        chosen=max(safe,key=lambda r:(r['correct'],r['available_events'],r['threshold'],r['margin']))
        atomic_json(chosen_path,chosen);atomic_json(folder/'train_results.json',dict(results=results,chosen=chosen))
        print('TRAIN',chosen,flush=True)
        return chosen
    if (folder/'test_result.json').exists():raise RuntimeError('Identity holdout already evaluated; do not tune on it')
    chosen=read_json(chosen_path,None)
    if not chosen:raise ValueError('Freeze a train choice before holdout evaluation')
    vectors=event_vectors(events,features,chosen['variant'],chosen['cloth_weight'])
    assignment=assign(events,vectors,chosen['threshold'],chosen['margin'])
    score=metrics(events,assignment,answers,set(index['split']['test_keys']))
    atomic_json(folder/'test_result.json',dict(config=chosen,score=score,assignments=assignment))
    print('HOLDOUT',score,flush=True)
    return score


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=['prepare','embed','bench','final']);parser.add_argument('day');parser.add_argument('--mask',action='store_true')
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.day,mask_snapshots=args.mask)
    elif args.mode=='embed':embed(args.day)
    else:bench(args.day,final=args.mode=='final')
