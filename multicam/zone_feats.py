"""13.10.2026: per-observation zone features of every /doorside stretch (SAM 3.1 outlines at 1280, every 3rd tick),
fast (on crops), for rule experiments off the machine. -> data/door_v2/zone_feats.json.gz
row: tag, role, k, piece, root(owner person), truth, x1,y1,x2,y2, fx,fy (lowest pixel), n, band, shop, door, gallery
(pixels in each owner zone), legs (share of the legs box covered by another mask)
plus per tag: the owner's events."""
import gzip, json, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_side as S

Z = json.load(open(ROOT / 'data/door_v2/door_lines_owner.json', encoding='utf-8'))['lines']
NAMES = ['band', 'shop', 'door', 'gallery']
zm = {}
for n, z in zip(NAMES, Z):
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(m, [np.array(z['pts'], np.int32)], 1)
    zm[n] = m.astype(bool)


def raster(poly, box):
    x1, y1, x2, y2 = box
    m = np.zeros((y2 - y1 + 1, x2 - x1 + 1), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32) - [x1, y1]], 1)
    return m.astype(bool)


def truth_at(xs, t):
    xs = sorted(xs)
    s = xs[0][1]
    for tt, ss in xs:
        if tt <= t + 1e-6:
            s = ss
    return s


out = {'zones': Z, 'tags': {}}
for f in sorted((ROOT / 'data/door_side').glob('2026*.json')):
    if 'history' in f.name:
        continue
    for tag, st in json.load(open(f, encoding='utf-8')).items():
        if not isinstance(st, dict) or not st.get('sides'):
            continue
        st = S.migrate(tag, st)
        merge, cuts, nop = st.get('merge', {}), st.get('cuts', {}), set(st.get('noperson', []))
        joined = {}
        for p, xs in st['sides'].items():
            joined.setdefault(S.root_of(merge, p), []).extend(xs)
        ev = S.events(st['sides'], merge, list(nop))
        ol = S.outlines(tag)
        rows = []
        for ks in sorted(ol, key=int):
            k = int(ks)
            if k % 3:
                continue
            t = k / S.FPS
            items = []
            for piece, _, poly in ol[ks]:
                a = np.array(poly)
                box = [int(max(0, a[:, 0].min())), int(max(0, a[:, 1].min())), int(min(1279, a[:, 0].max())), int(min(719, a[:, 1].max()))]
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                m = raster(poly, box)
                if m.sum() >= 30:
                    items.append((piece, box, m, poly))
            for i, (piece, (x1, y1, x2, y2), m, poly) in enumerate(items):
                ys, xs = np.nonzero(m)
                j = np.argmax(ys)
                fx, fy = int(xs[j] + x1), int(ys[j] + y1)
                zc = [int((m & zm[z][y1:y2 + 1, x1:x2 + 1]).sum()) for z in NAMES]
                h = y2 - y1 + 1
                L = [max(0, x1 - 5), min(719, y2 - int(.2 * h)), min(1279, x2 + 5), min(719, y2 + int(.5 * h))]
                cov = 0.0
                for jj, (_, b, mm, pp) in enumerate(items):
                    if jj == i or b[2] < L[0] or b[0] > L[2] or b[3] < L[1] or b[1] > L[3]:
                        continue
                    lm = np.zeros((L[3] - L[1] + 1, L[2] - L[0] + 1), np.uint8)
                    cv2.fillPoly(lm, [np.array(pp, np.int32) - [L[0], L[1]]], 1)
                    cov = max(cov, float(lm.mean()))
                r = S.root_of(merge, S.eff(cuts, piece, t))
                tru = truth_at(joined[r], t) if (r in joined and r not in nop) else ('noperson' if r in nop else None)
                rows.append([k, str(piece), r, tru, x1, y1, x2, y2, fx, fy, int(m.sum())] + zc + [round(cov, 3)])
        out['tags'][tag] = {'role': st.get('role', 'test'), 'rows': rows, 'events': ev, 'ticks': max(int(x) for x in ol) if ol else 0}
        print(tag, len(rows), flush=True)
json.dump(out, gzip.open(ROOT / 'data/door_v2/zone_feats.json.gz', 'wt'))
print('done')
