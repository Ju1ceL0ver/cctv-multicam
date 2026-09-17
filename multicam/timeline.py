"""Per real person: which pieces, which ids, and why cross-camera links were missed."""
import sys, os, json, numpy as np
from collections import Counter
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build, fuse, co_distances, app_dist, time_overlap
from clipdata import load
clip = sys.argv[1]; tag = 'yolo26x-seg'
dets, feats, meta, embs = load(clip, tag)
cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
segs, people, _ = fuse(per_cam)
items = per_cam['cam1'] + per_cam['cam2']
gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_%s.json' % tag)))
owner = {s: pid for pid, p in enumerate(people) for s in p['segments']}
ipid = {m: owner[si] for si, s in enumerate(segs) for m in s['members']}
rows = []
for n, it in enumerate(items):
    lab = Counter(gt[it['cam']].get(str(int(i)), '?') for i in it['det']); lab.pop('?', None)
    rows.append((lab.most_common(1)[0][0] if lab else '?', n))
for g in sys.argv[2] if len(sys.argv) > 2 else 'ABC':
    print('=== %s' % g)
    mine = sorted([n for gg, n in rows if gg == g], key=lambda n: items[n]['t'][0])
    for n in mine:
        it = items[n]
        print('   item %2d %s %5.1f-%5.1f P%-2d start (%.1f,%.1f) end (%.1f,%.1f)' % (n, it['cam'], it['t'][0], it['t'][-1], ipid[n], *it['xy'][0], *it['xy'][-1]))
    for i in mine:
        for j in mine:
            if i < j and ipid[i] != ipid[j]:
                a, b = items[i], items[j]
                d = co_distances(a, b)
                print('     unlinked %2d-%2d: %s overlap %.1fs co-frames %d median dist %s share<1m %s app %.2f gap %.1fs' % (
                    i, j, 'cross' if a['cam'] != b['cam'] else 'same', time_overlap(a, b), len(d),
                    '%.2f' % np.median(d) if len(d) else '-', '%.2f' % (d < 1).mean() if len(d) else '-',
                    app_dist(a['app'], b['app']), b['t'][0] - a['t'][-1]))
