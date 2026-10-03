"""Clean maps from the keep/delete network's scores (score_all.py) and the owner's own answers (they always win).
person: score >= DELETE -> deleted (the place becomes background, 0); CLEAN <= score < DELETE -> ignored (value IGNORE_VALUE: no loss
on it at all); score < CLEAN -> kept. People under MIN_PX (300 px of the full frame) are deleted, loose bits of a kept mask under MIN_PX too.
Writes data/sam31_stills_clean/<frame>.png (same size as the draft), data/keep2/decisions.json (counts) and puts CHECK random people of the
clean set first in the owner's /keep2 queue, to measure how much bad is left inside it.
usage: decide.py [DELETE=0.85] [CLEAN=0.01] [CHECK=200] [OUT=sam31_stills_clean]"""
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import keep2 as K2
import keepnet as K

IGNORE_VALUE = 254
ROOT = K.ROOT


def main(delete=0.85, clean=0.01, check=200):
    t0 = time.time()
    from storage import atomic_json
    A = json.load(open(ROOT / 'data' / 'keep2' / 'all_scores.json'))
    S, L = K2.samples(ROOT), K2.labels(ROOT)
    own = {}                                                    # (frame, value) -> action by the owner
    for sid, v in L.items():
        s = S.get(sid)
        if s:
            own[(s['frame'], s['value'])] = 'keep' if v['label'] == 1 else 'delete' if v['label'] in K2.DELETE else 'ignore'
    out = ROOT / 'data' / (sys.argv[4] if len(sys.argv) > 4 else 'sam31_stills_clean')
    out.mkdir(exist_ok=True)
    cnt = {'keep': 0, 'delete': 0, 'ignore': 0, 'small': 0, 'own': 0}
    kept = []
    for k, (fid, people) in enumerate(sorted(A.items())):
        p = ROOT / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % fid)
        lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if lab is None:
            continue
        lab = lab[..., 0] if lab.ndim == 3 else lab
        new = lab.copy()
        big = {int(v) for v in people}
        for v in np.unique(lab):
            if v and int(v) not in big:                        # under MIN_PX: the size rule
                new[lab == v] = 0
                cnt['small'] += 1
        for v, rec in people.items():
            v = int(v)
            act = own.get((fid, v))
            if act:
                cnt['own'] += 1
            else:
                act = 'delete' if rec[0] >= delete else 'ignore' if rec[0] >= clean else 'keep'
            cnt[act] += 1
            if act == 'delete':
                new[lab == v] = 0
            elif act == 'ignore':
                new[lab == v] = IGNORE_VALUE
            else:
                kept.append((fid, v, rec))
                m = (lab == v).astype(np.uint8)                # fragments: the main body stays; a bit stays only if >= 300 px of the full frame
                n, cc, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)       # and within 300 px of it (his own masks: the largest gap is 255 px)
                if n > 2:
                    f = lab.shape[1] / 2176
                    limit = K.MIN_FULL * lab.shape[0] * lab.shape[1] / (2176 * 1224)
                    main = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
                    dist = cv2.distanceTransform((cc != main).astype(np.uint8), cv2.DIST_L2, 3)
                    for c in range(1, n):
                        if c != main and (st[c, cv2.CC_STAT_AREA] < limit or dist[cc == c].min() > 300 * f):
                            new[cc == c] = 0
                            cnt['fragments'] = cnt.get('fragments', 0) + 1
        cv2.imwrite(str(out / ('%s.png' % fid)), new)
        if k % 2000 == 0:
            print('%d frames, %.0f s' % (k, time.time() - t0), flush=True)
    # his check: random kept people he has not answered, first in his queue
    random.seed(7)
    pool = [x for x in kept if (x[0], x[1]) not in own]
    pick = random.sample(pool, min(check, len(pool)))
    pri = K2.priority(ROOT)
    pri = {i: r for i, r in pri.items()}
    for j, (fid, v, rec) in enumerate(pick):
        sid = '%s_p%d' % (fid, v)
        lab = K2._map(ROOT / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % fid))
        pp = {a: (b, c, d) for a, b, c, d in K2.people(lab)}
        if v not in pp:
            continue
        area, box, foot = pp[v]
        S[sid] = {'frame': fid, 'value': v, 'day': fid[:8], 'cam': 'cam2' if '_cam2_' in fid else 'cam1', 'box': box, 'foot': foot, 'area': area,
                  'area_full': int(area * K2.SCALE), 'top': foot[1] < K2.TOP * K2.H, 'rl': 'check'}
        pri[sid] = -1000 + j
    atomic_json(K2.folder(ROOT) / 'samples.json', S)
    json.dump(pri, open(K2.folder(ROOT) / 'priority.json', 'w'))
    json.dump({'delete': delete, 'clean': clean, 'counts': cnt, 'check': len(pick), 'seconds': round(time.time() - t0)}, open(K2.folder(ROOT) / 'decisions.json', 'w'), indent=1)
    print(cnt, 'check', len(pick), '%.0f s' % (time.time() - t0), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(float(a[0]) if a else 0.85, float(a[1]) if len(a) > 1 else 0.01, int(a[2]) if len(a) > 2 else 200)
