"""Which pieces the system believes are one person -- so review is one click per person.

The labelling tool asks about one piece at a time, and a shopper walking the hall for
nine minutes is dozens of pieces. The clustering already has an opinion; writing it down
lets the reviewer confirm a whole person at once and only take apart the ones that are
wrong."""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build, fuse
from clipdata import load

clip = sys.argv[1]
tag = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x-seg'
d = os.path.join('data', 'raw_clips', clip)
pieces = json.load(open(os.path.join(d, 'pieces_%s.json' % tag)))
dets, feats, meta, embs = load(clip, tag)
cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
items = per_cam['cam1'] + per_cam['cam2']
segs, people, _ = fuse(per_cam)

by_first = {}
for p in pieces:
    by_first[(p['cam'], int(p['dets'][0]))] = p['piece']

groups = []
for pid, person in enumerate(people):
    mine = []
    for si in person['segments']:
        for m in segs[si]['members']:
            it = items[m]
            k = (it['cam'], int(it['det'][0]))
            if k in by_first:
                mine.append(by_first[k])
    if not mine:
        continue
    mine = sorted(set(mine))
    ps = [pieces[i] for i in mine]
    groups.append({'person': pid, 'pieces': mine,
                   't0': round(min(p['t0'] for p in ps), 1), 't1': round(max(p['t1'] for p in ps), 1),
                   'cams': sorted({p['cam'] for p in ps}),
                   'dets': int(sum(p['n_dets'] for p in ps))})
placed = {i for g in groups for i in g['pieces']}
for p in pieces:
    if p['piece'] not in placed:      # a piece the clustering left alone is a person of one piece
        groups.append({'person': None, 'pieces': [p['piece']], 't0': p['t0'], 't1': p['t1'],
                       'cams': [p['cam']], 'dets': p['n_dets']})
groups.sort(key=lambda g: g['t0'])
out = os.path.join(d, 'groups_%s.json' % tag)
json.dump(groups, open(out, 'w'), indent=1)
print('%s: %d pieces -> %d people (largest %d pieces)'
      % (clip, len(pieces), len(groups), max(len(g['pieces']) for g in groups) if groups else 0))
