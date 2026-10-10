"""Frozen door-event comparison on the owner's completed TEST stretches only.

python door_seg_check.py CKPT NAME [--baseline micro1s3c240p]
GPU inference runs once, then both track sources use door_combo.Live on CPU,
the same 1280x720 video, stride 3 and frozen current door settings. No tuning.
Results and exact cohort: data/micro_door/NAME/check_{manifest,results}.json.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def cohort():
    out = []
    for path in sorted((ROOT / 'data/door_side').glob('2026????.json')):
        state = json.loads(path.read_text(encoding='utf-8'))
        for tag, s in state.items():
            if s.get('done') and s.get('role', 'test') != 'train':
                out.append({'day': path.stem, 'tag': tag, 'state': s})
    if not out:
        raise ValueError('no completed held-out stretches')
    return out


def counts(match):
    return {kind: {key: match[kind][key] for key in ('hit', 'pred', 'true')} for kind in ('in', 'out')}


def score_source(item, name, stride=3, side_guard=False):
    import cv2
    import door_combo as C
    import door_compare_video as V
    import door_sam as DS
    import door_side as S
    import door_v2 as D
    tag, day = item['tag'], item['day']
    base = ROOT / 'data/micro_door' / name / tag / 'cam1'
    if not (base / 'report.json').exists():
        raise FileNotFoundError('missing source: %s' % base)
    masks, by_tick, info = V.people(base)
    src_stride = int(info.get('stride', 1))
    by_tick = {k * src_stride: rows for k, rows in by_tick.items()}
    video = ROOT / 'data/sam31_door' / tag / 'cam1/video.mp4'
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    if width <= 0 or height <= 0:
        raise ValueError('unreadable video: %s' % video)
    if side_guard:
        from door_seg_side import Live
        live = Live()
    else:
        live = C.Live()
    person_of, k = {}, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if k % stride == 0 and k in by_tick:
                rgb = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
                for row, pid in by_tick[k]:
                    x1, y1, x2, y2 = masks.rows[row, 4:8].astype(int)
                    crop = masks.crop(row).astype(np.uint8)
                    sx, sy = 1280 / width, 720 / height
                    xx, yy = int(x1 * sx), int(y1 * sy)
                    small = cv2.resize(crop, (max(1, round(crop.shape[1] * sx)), max(1, round(crop.shape[0] * sy))),
                                       interpolation=cv2.INTER_NEAREST)[:720 - yy, :1280 - xx]
                    foot = DS.foot_of(crop)
                    fx, fy = (x1 + foot[0], y1 + foot[1]) if foot else ((x1 + x2) / 2, y2)
                    key = (tag, k, row)
                    live.add(key, k * .08, rgb, (small, xx, yy), [fx / width, fy / height])
                    person_of[key] = pid
            k += 1
        events = live.events(None, person_of)
    finally:
        cap.release()
        pool = getattr(live, '_feat_pool', None)
        if pool:
            pool.shutdown(wait=True)
    state = S.migrate(tag, item['state'])
    truth = S.events(state['sides'], state.get('merge'), state.get('noperson', []))
    truth = [{'kind': e['kind'], 't': e['t']} for e in truth]
    match = D.match(events, truth, 6.0)
    return dict(counts=counts(match), f1=(match['in']['f1'] + match['out']['f1']) / 2,
                events=[{'kind': e['kind'], 't': e['t'], 'w': e['w']} for e in events], truth=truth,
                observed_tracks=len(set(person_of.values())), observations=len(person_of), video_ticks=k,
                inference_s_per_tick=info.get('s_per_live_tick'))


def run(ckpt, name, baseline='micro1s3c240p'):
    import door_seg_run as R
    os.environ['RA_DOOR_PIECES'] = '0'
    os.environ['RA_SIDE_FINAL'] = '1'
    os.environ['RA_DOOR_LINE'] = '1'
    items = cohort()
    paired = [item for item in items if (ROOT / 'data/micro_door' / baseline / item['tag'] / 'cam1/report.json').exists()]
    if not paired:
        raise FileNotFoundError('no held-out stretches with saved baseline')
    frozen = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              (ROOT / 'data/door_v2/side_final.json', ROOT / 'data/door_v2/combo_final.json')}
    print('TEST cohort', len(items), [i['tag'] for i in items], flush=True)
    days = sorted({i['day'] for i in items})
    R.run(ckpt, name, days, ','.join(i['tag'] for i in items))
    base = ROOT / 'data/micro_door' / name
    manifest = dict(ckpt=str(ckpt), baseline=baseline, cohort=items, config_hashes=frozen, tolerance=6.0,
                    paired_tags=[i['tag'] for i in paired], missing_baseline=[i['tag'] for i in items if i not in paired],
                    stride=3, settings='current live door, no tuning', made=time.strftime('%Y-%m-%d %H:%M:%S'))
    (base / 'check_manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    result = {'manifest': manifest, 'sources': {}}
    for src in (name, baseline):
        details = {}
        total = {kind: dict(hit=0, pred=0, true=0) for kind in ('in', 'out')}
        for item in (items if src == name else paired):
            detail = score_source(item, src)
            details[item['tag']] = detail
            for kind in total:
                for key in total[kind]:
                    total[kind][key] += detail['counts'][kind][key]
            print(src, item['tag'], detail['counts'], flush=True)
        fs = [2 * v['hit'] / max(1, v['pred'] + v['true']) for v in total.values()]
        result['sources'][src] = dict(counts=total, f1=sum(fs) / 2, stretches=details)
        (base / 'check_results.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    for filename, digest in frozen.items():
        if hashlib.sha256((ROOT / 'data/door_v2' / filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError('door settings changed during comparison: %s' % filename)
    result['complete'] = True
    result['paired'] = {}
    for src in (name, baseline):
        total = {kind: dict(hit=0, pred=0, true=0) for kind in ('in', 'out')}
        for item in paired:
            cs = result['sources'][src]['stretches'][item['tag']]['counts']
            for kind in total:
                for key in total[kind]:
                    total[kind][key] += cs[kind][key]
        result['paired'][src] = dict(counts=total, f1=sum(2 * v['hit'] / max(1, v['pred'] + v['true'])
                                                        for v in total.values()) / 2)
    (base / 'check_results.json').write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('COMPLETE identical paired cohort', json.dumps(result['paired']), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ckpt')
    p.add_argument('name')
    p.add_argument('--baseline', default='micro1s3c240p')
    args = p.parse_args()
    run(args.ckpt, args.name, args.baseline)
