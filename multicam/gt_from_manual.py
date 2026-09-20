"""Turn the manual piece labels from the web tool into detection-level ground truth.
'?' pieces (two people in one box) are dropped, as they cannot score anything."""
import sys, os, json
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT)
clip = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x-seg'
d = os.path.join('data', 'raw_clips', clip)
from review_store import read_state
state = read_state(d)
pieces, man = state['pieces'], state['labels']
gt = {'cam1': {}, 'cam2': {}}
used = 0
for p in pieces:
    lab = man.get(str(p['piece']))
    if not lab or lab == '?' or state['quality'].get(str(p['piece'])) in ('false_positive', 'mixed'):
        continue
    used += 1
    for di in p['dets']:
        gt[p['cam']][str(di)] = lab
out = os.path.join(d, 'gt_identity_%s.json' % tag)
json.dump(gt, open(out, 'w'))
from collections import Counter
print('%s: %d/%d pieces labelled -> %s; people %s' % (clip, used, len(pieces), {c: len(v) for c, v in gt.items()},
      dict(Counter(list(gt['cam1'].values()) + list(gt['cam2'].values())).most_common(10))))
