"""13.10.2026, the owner: a small tree on zone features -> inside/outside per observation, then read its rules and make
them deterministic. Features (SAM 3.1 outlines at 1280, every 3rd tick): share of the mask in each owner zone, the zone
of the mask's lowest pixel, height, bottom y, whether another person's mask covers where this one's legs would be, and
causal track memory (shop pixels seen earlier on this track, door zone passed since). Truth: the owner's /doorside sides.
Fit on 'train' stretches only; 'test' scored once.  -> data/door_v2/zone_tree.json"""
import json, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_side as S
from sklearn.tree import DecisionTreeClassifier, export_text

Z = json.load(open(ROOT / 'data/door_v2/door_lines_owner.json', encoding='utf-8'))['lines']
NAMES = ['band', 'shop', 'door', 'gallery']
zm = {}
for n, z in zip(NAMES, Z):
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(m, [np.array(z['pts'], np.int32)], 1)
    zm[n] = m.astype(bool)
FEAT = ['f_band', 'f_shop', 'f_door', 'f_gallery', 'shop_px', 'foot_band', 'foot_shop', 'foot_door', 'foot_gallery',
        'h', 'bottom_y', 'legs_covered', 'touch_other', 'seen_shop_before', 'door_since_shop', 'seen_gallery_before']


def mask_of(poly):
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    return m.astype(bool)


def truth_at(xs, t):
    xs = sorted(xs)
    s = xs[0][1]
    for tt, ss in xs:
        if tt <= t + 1e-6:
            s = ss
    return s


rows = []
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
        ol = S.outlines(tag)
        mem = {}                                            # track -> causal memory
        for k in sorted(int(x) for x in ol):
            if k % 3:
                continue
            t = k / S.FPS
            items = [(piece, mask_of(poly)) for piece, _, poly in ol[str(k)]]
            items = [(p_, m) for p_, m in items if m.sum() >= 30]
            for i, (piece, m) in enumerate(items):
                ys, xs = np.nonzero(m)
                y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
                h = y2 - y1 + 1
                j = np.argmax(ys)
                fx, fy = xs[j], ys[j]
                foot = next((n for n in ('door', 'shop', 'band', 'gallery') if zm[n][fy, fx]), 'none')
                legs = np.zeros((720, 1280), bool)                  # where the legs would be: below the box by 0.5 h
                legs[min(719, y2 - int(.2 * h)):min(720, y2 + int(.5 * h) + 1), max(0, x1 - 5):min(1280, x2 + 6)] = True
                others = [mm for jj, (_, mm) in enumerate(items) if jj != i]
                cov = max((float((mm & legs).sum()) / max(1, legs.sum()) for mm in others), default=0.0)
                touch = float(any((cv2.dilate(mm.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0)[m].any() for mm in others))
                n = m.sum()
                g = mem.setdefault(piece, {'shop': 0, 'door_since': 0, 'gal': 0})
                shop_px = int((m & zm['shop']).sum())
                feat = [float((m & zm[z]).sum()) / n for z in NAMES] + [shop_px] + \
                       [float(foot == z) for z in ('band', 'shop', 'door', 'gallery')] + \
                       [h, fy, cov, touch, float(g['shop'] > 0), float(g['door_since']), float(g['gal'] > 0)]
                r = S.root_of(merge, S.eff(cuts, piece, t))
                if r not in nop and r in joined:
                    rows.append(dict(x=feat, y=int(truth_at(joined[r], t) == 'in'), role=st.get('role', 'test'), tag=tag, t=t))
                if shop_px >= 20:
                    g['shop'] += 1; g['door_since'] = 0
                if (m & zm['door']).sum() >= 20 and g['shop']:
                    g['door_since'] = 1
                if foot == 'gallery':
                    g['gal'] += 1
X = np.array([r['x'] for r in rows]); y = np.array([r['y'] for r in rows]); tr = np.array([r['role'] == 'train' for r in rows])
out = {'n_train': int(tr.sum()), 'n_test': int((~tr).sum())}
for depth in (2, 3, 4, 5):
    clf = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=50, class_weight='balanced', random_state=0).fit(X[tr], y[tr])
    p = clf.predict(X)
    def acc(msk):
        return dict(acc=round(float((p[msk] == y[msk]).mean()), 4), in_as_out=int(((p == 0) & (y == 1) & msk).sum()),
                    out_as_in=int(((p == 1) & (y == 0) & msk).sum()), n=int(msk.sum()))
    out['depth%d' % depth] = dict(train=acc(tr), test=acc(~tr), rules=export_text(clf, feature_names=FEAT, decimals=3),
                                  importance={f: round(float(v), 3) for f, v in zip(FEAT, clf.feature_importances_) if v > .005})
json.dump(out, open(ROOT / 'data/door_v2/zone_tree.json', 'w'), ensure_ascii=False, indent=1)
for d in (2, 3, 4, 5):
    o = out['depth%d' % d]
    print('depth', d, 'train', o['train'], 'test', o['test'], o['importance'])
print(out['depth4']['rules'])
