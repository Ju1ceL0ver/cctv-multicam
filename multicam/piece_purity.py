"""Are the tracker's own pieces pure? (If not, association cannot be blamed.)"""
import sys, os, json, numpy as np
from collections import Counter
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build
from clipdata import load
clip = sys.argv[1]
dets, feats, meta, embs = load(clip)
gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')))
cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
per_cam, stats = build(cams, dets, feats, embs, meta.get('clean'))
print(stats)
rows = []
for it in per_cam['cam1'] + per_cam['cam2']:
    c = Counter(gt[it['cam']].get(str(int(i)), '?') for i in it['det']); c.pop('?', None)
    if not c:
        continue
    lab, n = c.most_common(1)[0]
    rows.append((lab, n / sum(c.values()), sum(c.values()), it))
pure = [r for r in rows if r[1] >= 0.95]
print('pieces with identity: %d | pure (>=95%%): %d | impure: %d' % (len(rows), len(pure), len(rows) - len(pure)))
for lab, share, n, it in sorted(rows, key=lambda r: r[1])[:10]:
    print('   impure piece %s %.0f-%.0f s share %.2f (%d dets) majority %s' % (it['cam'], it['t'][0], it['t'][-1], share, n, lab))
cov = {}
for lab, share, n, it in rows:
    cov.setdefault(lab, []).append(it['t'][-1] - it['t'][0])
for lab in sorted(cov, key=lambda l: -sum(cov[l]))[:6]:
    print('   %s: %d pieces, total %.0f s, median piece %.1f s' % (lab, len(cov[lab]), sum(cov[lab]), np.median(cov[lab])))
