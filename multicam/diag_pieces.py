"""Where do real people fragment? Per person: pieces over time with their output ids, and for each
consecutive pair of that person's pieces in time, why association did not join them."""
import sys, os, json, numpy as np
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build, fuse, co_distances, app_dist, time_overlap, distinct_people
from clipdata import load
clip = sys.argv[1]; who = sys.argv[2].split(',')
dets, feats, meta, embs = load(clip)
cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
segs, people, _ = fuse(per_cam)
items = per_cam['cam1'] + per_cam['cam2']
gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')))
owner = {s: pid for pid, p in enumerate(people) for s in p['segments']}
ipid = {m: owner[si] for si, s in enumerate(segs) for m in s['members']}
lab = {}
for n, it in enumerate(items):
    c = Counter(gt[it['cam']].get(str(int(i)), '?') for i in it['det']); c.pop('?', None)
    lab[n] = (c.most_common(1)[0] if c else ('?', 0), sum(c.values()), len(it['det']))
reasons = defaultdict(Counter)
for g in who:
    mine = sorted([n for n in lab if lab[n][0][0] == g], key=lambda n: items[n]['t'][0])
    print('=== %s: %d pieces, ids %s' % (g, len(mine), dict(Counter(ipid[n] for n in mine))))
    for cam in ('cam1', 'cam2'):
        seq = [n for n in mine if items[n]['cam'] == cam]
        cover = sum(items[n]['t'][-1] - items[n]['t'][0] for n in seq)
        print('   %s: %d pieces covering %.0f s, median piece %.1f s' % (cam, len(seq), cover,
              np.median([items[n]['t'][-1] - items[n]['t'][0] for n in seq]) if seq else 0))
        for a, b in zip(seq, seq[1:]):
            A, B = items[a], items[b]
            gap = B['t'][0] - A['t'][-1]
            if ipid[a] == ipid[b]:
                continue
            jump = np.linalg.norm(B['xy'][0] - A['xy'][-1])
            ad = app_dist(A['app'], B['app'])
            r = 'gap>10s' if gap > 10 else ('far %.1fm' % jump if jump > 1.5 * max(gap, 0) + 0.8 else ('colour %.2f' % ad if ad > (1.3 if gap < 3 else 0.9) else 'other'))
            reasons[g][r.split()[0]] += 1
            if len(seq) < 40:
                print('     %s piece %d->%d: gap %.1fs jump %.2fm app %.2f -> %s' % (cam, a, b, gap, jump, ad, r))
    print('   break reasons:', dict(reasons[g]))
