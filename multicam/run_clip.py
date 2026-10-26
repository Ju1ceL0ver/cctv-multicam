"""Association on one detected clip: people, visits, identity score (if GT exists).

usage: run_clip.py CLIP [TAG]"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera, place
from fusion2 import build, fuse, visits, REG
from clipdata import load
from eval_ids import evaluate

clip = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x-seg'
dets, feats, meta, embs = load(clip, tag)
calib = json.load(open('data/calib_final.json'))
cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
per_cam, stats = build(cams, dets, feats, embs, meta.get('clean'))
segs, people, cross = fuse(per_cam)
items = per_cam['cam1'] + per_cam['cam2']
owner = {s: pid for pid, p in enumerate(people) for s in p['segments']}
label = {'cam1': {}, 'cam2': {}}
for si, s in enumerate(segs):
    for m in s['members']:
        for di in items[m]['det']:
            label[items[m]['cam']][int(di)] = owner[si]
print(clip, tag, stats)
print('people %d, cross-camera merges %d' % (len(people), len(cross)))
V = visits(people)
for v in V:
    if v['inside_s'] >= 2.0 or v['events']:
        print('  P%-3d %6.1f-%6.1f s inside %5.1f s  %-9s %s' % (v['person'], v['first'], v['last'], v['inside_s'], '+'.join(v['cams']), v['events']))
gt_path = os.path.join('data', 'raw_clips', clip, 'gt_identity_%s.json' % tag)
if os.path.exists(gt_path):
    evaluate(label, gt_path)
out = {'clip': clip, 'tag': tag, 'start': meta['start'],
       'people': [{'id': pid, 't': p['t'].round(2).tolist(), 'xy': p['xy'].round(3).tolist(), 'cams': p['cams'],
                   'height_m': None if not np.isfinite(p['h']) else round(float(p['h']), 2)} for pid, p in enumerate(people)],
       'visits': V}
positions = place(cams, REG, dets)
# Keep detection provenance for window seams and visit crops; no tracking decisions change.
for pid, person in enumerate(out['people']):
    sources, vectors, observations = [], [], []
    for member in segs[pid]['members']:
        item = items[member]
        cam = item['cam']
        indices = np.asarray(item['det'], int)
        observations.extend([[round(float(t),3), round(float(xy[0]),3), round(float(xy[1]),3), bool(foot)]
                             for t,xy,foot in zip(item['t'],item['xy'],positions[cam]['vis'][indices])
                             if np.isfinite(xy).all()])
        for j in indices[::max(1, int(round(0.2 / max(float(np.median(np.diff(item['t']))) if len(item['t']) > 1 else 0.2, 0.04))))]:
            row = dets[cam][j]
            sources.append({'cam': cam, 't': round(float(row[0]), 3),
                            'raw_t': round(float(row[0]) - (meta.get('cam1_offset_s', 0) if cam == 'cam1' else 0), 3),
                            'box': row[1:5].round(1).tolist()})
        if embs is not None:
            clean = meta.get('clean', {}).get(cam)
            chosen = indices if clean is None else indices[clean[indices]]
            if len(chosen):
                e = embs[cam][chosen]
                e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-9)
                vectors.append(e.mean(0))
    person['sources'] = sorted(sources, key=lambda q: q['t'])
    person['observations'] = sorted(observations,key=lambda q:q[0])
    person['embedding_model'] = meta.get('embeddings')
    if vectors:
        e = np.mean(vectors, axis=0)
        person['embedding'] = (e / max(float(np.linalg.norm(e)), 1e-9)).round(6).tolist()

json.dump(out, open(os.path.join('data', 'raw_clips', clip, 'people_%s.json' % tag), 'w'))
