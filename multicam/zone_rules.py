"""13.10.2026, the owner's zones on camera 1 (/doorside K/G -> data/door_v2/door_lines_owner.json): how true are simple
zone rules against the owner's /doorside sides, per observation (SAM 3.1 tracks, every 3rd tick, outlines at 1280).
usage: zone_rules.py -> data/door_v2/zone_rules.json"""
import json, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_side as S

Z = json.load(open(ROOT / 'data/door_v2/door_lines_owner.json', encoding='utf-8'))['lines']
NAMES = ['band', 'shop', 'door', 'gallery']           # 1 orange band by the glass, 2 shop, 3 doorway, 4 far gallery
zm = {}
for n, z in zip(NAMES, Z):
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(m, [np.array(z['pts'], np.int32)], 1)
    zm[n] = m.astype(bool)


def stats(poly):
    m = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 1)
    m = m.astype(bool)
    if m.sum() < 30:
        return None
    ys, xs = np.nonzero(m)
    i = np.argmax(ys)
    foot = (int(xs[i]), int(ys[i]))
    where = next((n for n in ('door', 'shop', 'band', 'gallery') if zm[n][foot[1], foot[0]]), 'none')
    return dict(shop=int((m & zm['shop']).sum()), door=int((m & zm['door']).sum()), foot=where, h=int(ys.max() - ys.min()))


def truth_at(xs, t):
    xs = sorted(xs)
    s = xs[0][1]
    for tt, ss in xs:
        if tt <= t + 1e-6:
            s = ss
    return s


out = {'rows': []}
for f in sorted((ROOT / 'data/door_side').glob('2026*.json')):
    if 'history' in f.name:
        continue
    day = f.stem
    for tag, st in json.load(open(f, encoding='utf-8')).items():
        if not isinstance(st, dict) or not st.get('sides'):
            continue
        st = S.migrate(tag, st)
        merge, cuts, nop = st.get('merge', {}), st.get('cuts', {}), set(st.get('noperson', []))
        joined = {}
        for p, xs in st['sides'].items():
            joined.setdefault(S.root_of(merge, p), []).extend(xs)
        ol = S.outlines(tag)
        for k, items in ol.items():
            k = int(k)
            if k % 3:
                continue
            t = k / S.FPS
            for piece, _, poly in items:
                r = S.root_of(merge, S.eff(cuts, piece, t))
                if r in nop or r not in joined:
                    continue
                s = stats(poly)
                if s:
                    out['rows'].append(dict(s, tag=tag, role=st.get('role', 'test'), p=r, t=round(t, 2), truth=truth_at(joined[r], t)))
rows = out['rows']
def summ(rs):
    res = {}
    for name, f in (('shop_px>=1', lambda r: r['shop'] >= 1), ('shop_px>=20', lambda r: r['shop'] >= 20)):
        a = [r for r in rs if f(r)]
        res[name] = dict(fires=len(a), fires_but_outside=sum(r['truth'] == 'out' for r in a),
                         inside_total=sum(r['truth'] == 'in' for r in rs), inside_caught=sum(r['truth'] == 'in' for r in a))
    nf = [r for r in rs if r['shop'] < 1]
    by = {}
    for r in nf:
        by.setdefault('%s/%s' % (r['foot'], r['truth']), 0)
        by['%s/%s' % (r['foot'], r['truth'])] += 1
    res['no_shop_px_by_foot_zone/truth'] = by
    return res
out['train'] = summ([r for r in rows if r['role'] == 'train'])
out['test'] = summ([r for r in rows if r['role'] != 'train'])
bad = [r for r in rows if r['shop'] >= 1 and r['truth'] == 'out']
out['bad_examples'] = sorted(bad, key=lambda r: -r['shop'])[:40]
json.dump(out, open(ROOT / 'data/door_v2/zone_rules.json', 'w'), ensure_ascii=False)
print(json.dumps({k: out[k] for k in ('train', 'test')}, ensure_ascii=False, indent=1))
print(json.dumps(out['bad_examples'][:15], ensure_ascii=False))
