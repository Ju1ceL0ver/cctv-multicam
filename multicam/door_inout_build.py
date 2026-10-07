"""A /inout batch from the door stretches (06.10.2026): people at the edge of the shop floor by the door, where the
inside/outside classifier errs, so that the owner's answers retrain it where it matters.

From the teacher's tracks of the door stretches (data/sam31_door, camera 1): one look a second, every person whose
feet are within NEAR_PX of the floor's edge (on either side); the classifier's (door_v2.io_model) doubt makes the
order -- the least sure first -- plus some sure ones as a check; at most PER_PIECE looks of one track piece, MIN_GAP s
apart. Each look becomes a 'draft' the /inout page already reads: the frame (1280 x 720) in
data/seg_datasets/door_inout/images/, the persons' label map in .../drafts/; the sample goes into data/inout/
samples.json with door=True (they are asked first).

usage: door_inout_build.py [N] [DAYS...]  (CPU)"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'seg_datasets' / 'door_inout'
NEAR_PX = 120
PER_PIECE, MIN_GAP = 2, 3.0


def main(n=600, days=('20260917', '20260918', '20260919')):
    import cv2
    import door_sam as DS
    import door_v2 as D
    import inout
    import inout_train as IT
    import sam31_reid as R
    from storage import atomic_json, file_lock
    n = int(n)
    rng = random.Random(1)
    io = D.io_model()
    dist = inout.floor_distance('cam1')
    cands = []
    for day in days:
        for a, b in D.stretches(day):
            tag = DS.tag_of(day, a)
            base = ROOT / 'data' / 'sam31_door' / tag / 'cam1'
            if not (base / 'report.json').exists():
                continue
            info = json.load(open(base / 'info.json'))
            M = R.Masks(base / 'chunks.npz')
            owned, _ = R.link_seams(M, {int(s): int(sh) for s, e, sh in info['sessions']})
            person = {int(p): v for p, v in json.load(open(base / 'report.json'))['person_of_piece'].items()}
            static = DS.static_people(M, owned, person, info['ticks'])
            for p, rs in owned.items():
                if (person.get(int(p)) or 0) in static:
                    continue
                last = -1e9
                for r in sorted(rs, key=lambda r: M.rows[r, 1]):
                    k = int(M.rows[r, 1])
                    if k % 12 or k * D.TICK - last < 1.0:            # a look a second
                        continue
                    x1, y1, x2, y2 = (M.rows[r, 4:8] * [1280 / 2176, 720 / 1224, 1280 / 2176, 720 / 1224]).astype(int)
                    if y2 - y1 < 40:
                        continue
                    f = DS.foot_of(M.crop(r))
                    fx = int((M.rows[r, 4] + (f[0] if f else (M.rows[r, 6] - M.rows[r, 4]) / 2)) * 1280 / 2176)
                    fy = int((M.rows[r, 5] + (f[1] if f else M.rows[r, 7] - M.rows[r, 5])) * 720 / 1224)
                    d = float(dist[min(719, max(0, fy)), min(1279, max(0, fx))])
                    if abs(d) > NEAR_PX:
                        continue
                    s = {'cam': 'cam1', 'box': [int(x1), int(y1), int(x2), int(y2)], 'foot': [fx, fy], 'floor_px': round(d, 1)}
                    pr = io.predict_proba(np.array([IT.geometry(s)], np.float32))[0]
                    cands.append({'tag': tag, 'day': day, 'k': k, 'r': int(r), 'piece': int(p), 't': k * D.TICK, 'doubt': float(1 - pr.max()), **s})
                    last = k * D.TICK
    # the least sure first, a sure fifth as a check; per piece at most PER_PIECE, MIN_GAP apart
    # 07.10: skip people already asked (same stretch, within MIN_GAP s, boxes overlapping) -- a second batch asks new ones
    old = {}
    for o in inout.samples(str(ROOT)).values():
        if o.get('door') and '_door' in o.get('frame', ''):
            fr = o['frame']
            old.setdefault((o['day'], fr.split('_door')[1].split('_')[0]), []).append((int(fr.split('_k')[1][:5]) * D.TICK, o['box']))

    def iou(a, b):
        w = max(0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - w * h
        return w * h / u if u > 0 else 0
    n0 = len(cands)
    cands = [c for c in cands if not any(abs(c['t'] - t) < MIN_GAP and iou(c['box'], b) > 0.3
                                         for t, b in old.get((c['day'], c['tag'].split('_')[2]), []))]
    print('already asked, skipped', n0 - len(cands), flush=True)
    cands.sort(key=lambda c: -c['doubt'])
    sure = [c for c in cands if c['doubt'] < 0.1]
    rng.shuffle(sure)
    pick, per = [], {}
    for c in cands[:] + sure:
        key = (c['tag'], c['piece'])
        got = per.setdefault(key, [])
        if len(got) >= PER_PIECE or any(abs(c['t'] - t) < MIN_GAP for t in got):
            continue
        if c['doubt'] < 0.1 and sum(x['doubt'] < 0.1 for x in pick) >= n // 5:
            continue
        got.append(c['t']); pick.append(c)
        if len(pick) >= n:
            break
    (OUT / 'images').mkdir(parents=True, exist_ok=True)
    (OUT / 'drafts').mkdir(parents=True, exist_ok=True)
    caps = {}
    new = {}
    for c in sorted(pick, key=lambda c: (c['tag'], c['k'])):
        base = ROOT / 'data' / 'sam31_door' / c['tag'] / 'cam1'
        cap = caps.get(c['tag'])
        if cap is None:
            for v in caps.values():
                v.release()
            caps = {c['tag']: cv2.VideoCapture(str(base / 'video.mp4'))}
            cap = caps[c['tag']]
        cap.set(cv2.CAP_PROP_POS_FRAMES, c['k'])
        ok, f = cap.read()
        if not ok:
            continue
        ident = '%s_door%s_k%05d_cam1' % (c['day'], c['tag'].split('_')[2], c['k'])
        cv2.imwrite(str(OUT / 'images' / (ident + '.jpg')), cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
        M = R.Masks(base / 'chunks.npz')
        lab = np.zeros((1224, 2176), np.uint8)
        rows = [r for r in range(len(M.rows)) if int(M.rows[r, 1]) == c['k']]
        val = {}
        for i, r in enumerate(sorted(rows, key=lambda r: M.rows[r, 7])):        # the nearer (lower) drawn last
            x1, y1, x2, y2 = M.rows[r, 4:8].astype(int)
            lab[y1:y2, x1:x2][M.crop(r)] = i + 1
            val[r] = i + 1
        cv2.imwrite(str(OUT / 'drafts' / (ident + '.png')), cv2.resize(lab, (1280, 720), interpolation=cv2.INTER_NEAREST))
        v = val.get(c['r'])
        if v is None:
            continue
        new['%s_p%d' % (ident, v)] = {'frame': ident, 'value': v, 'day': c['day'], 'cam': 'cam1', 'box': c['box'], 'foot': c['foot'],
                                      'floor_px': c['floor_px'], 'near': abs(c['floor_px']) < inout.NEAR, 'door': True,
                                      'doubt': round(c['doubt'], 3)}
    f = inout.folder(str(ROOT))
    with file_lock(str(f / 'samples.json') + '.lock'):
        ss = inout.samples(str(ROOT))
        ss.update(new)
        atomic_json(f / 'samples.json', ss)
    print('candidates', len(cands), 'picked', len(pick), 'added', len(new), 'doubtful (<0.9 sure)', sum(c['doubt'] >= 0.1 for c in pick), flush=True)


if __name__ == '__main__':
    a = sys.argv[1:]
    main(*(a[:1]), **({'days': a[1:]} if len(a) > 1 else {}))
