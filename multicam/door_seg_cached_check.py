"""One frozen CPU-only comparison of student mask quantization correction.

Usage: door_seg_cached_check.py SOURCE NAME
Reuses SOURCE's exported masks and its original paired baseline result. No
new detector inference, training or TEST threshold search. A new NAME contains
its manifest, incremental scores and complete result; SOURCE is read-only.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.update(RA_DOOR_PIECES='0', RA_SIDE_FINAL='1', RA_DOOR_LINE='1')
ROOT=Path(__file__).resolve().parent


def run(source, name):
    import cv2
    from door_seg_check import score_source
    src=ROOT/'data/micro_door'/source
    original=json.loads((src/'check_results.json').read_text(encoding='utf-8'))
    if not original.get('complete'):
        raise ValueError('the reference comparison is incomplete')
    manifest=original['manifest']
    items=manifest['cohort']
    if set(manifest['paired_tags'])!={x['tag'] for x in items}:
        raise ValueError('this comparison requires a fully paired cohort')
    if any(x['state'].get('role','test')=='train' for x in items):
        raise ValueError('TRAIN is not part of this final event comparison')
    for filename,digest in manifest['config_hashes'].items():
        if hashlib.sha256((ROOT/'data/door_v2'/filename).read_bytes()).hexdigest()!=digest:
            raise ValueError('reference settings changed: '+filename)
    out=ROOT/'data/micro_door'/name
    if out.exists():
        raise FileExistsError('use a new result name')
    out.mkdir(parents=True)
    cv2.setNumThreads(2)
    baseline=manifest['baseline']
    result={'complete':False,'manifest':{**manifest,'detector_source':source,
            'student_side_policy':'h_in + stride-4 cell projection on door-line normal; CPU RGB in uncertain band',
            'made':time.strftime('%Y-%m-%d %H:%M:%S'),'baseline_reused_from':str(src/'check_results.json')},
            'sources':{baseline:original['sources'][baseline]},'paired':{baseline:original['paired'][baseline]}}
    details={};counts={kind:dict(hit=0,pred=0,true=0) for kind in ('in','out')};t0=time.perf_counter()
    for item in items:
        detail=score_source(item,source,stride=manifest['stride'],side_guard=True)
        details[item['tag']]=detail
        for kind in counts:
            for k in counts[kind]:counts[kind][k]+=detail['counts'][kind][k]
        print(item['tag'],detail['counts'],flush=True)
        f1=sum(2*v['hit']/max(1,v['pred']+v['true']) for v in counts.values())/2
        result['sources'][name]=dict(counts=counts,f1=f1,stretches=details)
        (out/'check_results.json').write_text(json.dumps(result,indent=1),encoding='utf-8')
    for filename,digest in manifest['config_hashes'].items():
        if hashlib.sha256((ROOT/'data/door_v2'/filename).read_bytes()).hexdigest()!=digest:
            raise RuntimeError('door settings changed during comparison')
    result['complete']=True
    result['paired'][name]=dict(counts=counts,f1=f1)
    result['cpu_evaluation_seconds']=time.perf_counter()-t0
    result['limitations']='Cached b2 masks. New fast inference and production latency are not tested by this CPU run.'
    (out/'check_results.json').write_text(json.dumps(result,indent=1),encoding='utf-8')
    print('COMPLETE',json.dumps(result['paired']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source');parser.add_argument('name')
    a=parser.parse_args();run(a.source,a.name)
