"""The learned door rule + the owner's two-class inside/outside model (07.10.2026), on SAM 3.1's door tracks of 17-19.09.

1. door_learn.evaluate on the three days (live features only): every rule candidate gets its out-of-day probability
   (learn_preds_<day>.json; the previous files are kept as *_before0710.json).
2. The binary model's events (door_binary_count.events, a fixed setting conf=3, thr=0.9 -- not tuned on any day).
3. One candidate list: the rule's candidates, plus binary events with no rule candidate of the same direction within
   JOIN s. Features: the rule's p, distance to the nearest binary event of the same direction, how many there are
   within 6 s, the opposite direction's, and whether the candidate is binary-only.
4. A second level (boosting) fitted on two days, the threshold from those two days, scored on the third; next to it
   the rule alone and the binary model alone with the same protocol. -> data/door_v2/fuse_binary.json"""
import json
import os
import shutil
import sys
from pathlib import Path

for k in ('RA_DOOR_NOCNN', 'RA_DOOR_NODEPTH', 'RA_DOOR_NOLK', 'RA_DOOR_NOTAP', 'RA_DOOR_STITCH'):
    os.environ.setdefault(k, '1')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'
DAYS = ('20260917', '20260918', '20260919')
JOIN = 4.0


def rule_preds():
    import door_learn as L
    for d in DAYS:
        f = OUT / ('learn_preds_%s.json' % d)
        if f.exists() and not (OUT / ('learn_preds_%s_before0710.json' % d)).exists():
            shutil.copy(f, OUT / ('learn_preds_%s_before0710.json' % d))
    days = [L.load_day(d, OUT / ('%s_sam31.jsonl.gz' % d)) for d in DAYS]
    L.evaluate(days)
    return {d: json.load(open(OUT / ('learn_preds_%s.json' % d))) for d in DAYS}


def feats(c, bins):
    same = [abs(b['t'] - c['t']) for b in bins if b['kind'] == c['kind']]
    opp = [abs(b['t'] - c['t']) for b in bins if b['kind'] != c['kind']]
    return [c['p'], min(same + [10.0]), sum(s <= 6 for s in same), min(opp + [10.0]), float(c['bin_only'])]


def nms(cs, p, thr, gap=4.0):
    keep = []
    for i in np.argsort(-p):
        if p[i] < thr:
            break
        if any(cs[j]['kind'] == cs[i]['kind'] and abs(cs[j]['t'] - cs[i]['t']) <= gap for j in keep):
            continue
        keep.append(i)
    return keep


def main():
    import door_binary_count as B
    import door_learn as L
    import door_v2 as D
    from sklearn.ensemble import HistGradientBoostingClassifier
    R = rule_preds()
    data = {}
    for d in DAYS:
        bins = [dict(e, t=e['t'] + L.SHIFT) for e in B.day_events(d, 3, 0.9)[0]]
        cs = [dict(kind=c['kind'], t=c['t'], w=c['w'], p=c['p'], bin_only=0) for c in R[d]]
        for b in bins:
            if not any(c['kind'] == b['kind'] and abs(c['t'] - b['t']) <= JOIN for c in cs):
                cs.append(dict(kind=b['kind'], t=b['t'], w=b['w'], p=0.0, bin_only=1))
        spans = [(st['a'] + L.SHIFT, st['end'] + L.SHIFT) for st in json.load(open(OUT / ('binary_%s.json' % d))).values()]
        inside = lambda t: any(a <= t <= b for a, b in spans)
        truth = [t for t in D.truth(d, True) if inside(t['t'])]
        y = L.label([(None, c['kind'], c['t'], c['w']) for c in cs], truth)
        X = np.array([feats(c, bins) for c in cs], np.float32)
        data[d] = dict(cs=cs, X=X, y=y, truth=truth, bins=bins)

    def f1(d, keep):
        m = D.match([data[d]['cs'][i] for i in keep], data[d]['truth'], L.TOL)
        return m, (m['in']['f1'] + m['out']['f1']) / 2

    rep = {}
    for test in DAYS:
        train = [d for d in DAYS if d != test]
        Xt = np.concatenate([data[d]['X'][data[d]['y'] >= 0] for d in train])
        yt = np.concatenate([data[d]['y'][data[d]['y'] >= 0] for d in train])
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=8, random_state=0).fit(Xt, yt)
        scores = {
            'rule': lambda d: data[d]['X'][:, 0] * (1 - data[d]['X'][:, 4]),
            'binary': lambda d: np.array([1.0 if (c['bin_only'] or x[1] <= JOIN) else 0.0 for c, x in zip(data[d]['cs'], data[d]['X'])]),
            'fused': lambda d: clf.predict_proba(data[d]['X'])[:, 1],
        }
        rep[test] = {}
        for name, sc in scores.items():
            best = max(np.arange(0.05, 0.96, 0.05), key=lambda th: np.mean([f1(d, nms(data[d]['cs'], sc(d), th))[1] for d in train]))
            m, f = f1(test, nms(data[test]['cs'], sc(test), best))
            rep[test][name] = {'f1': round(f, 3), 'thr': round(float(best), 2),
                               'in': [m['in'][k] for k in ('precision', 'recall', 'f1')],
                               'out': [m['out'][k] for k in ('precision', 'recall', 'f1')]}
        print(test, json.dumps(rep[test]), flush=True)
    json.dump(rep, open(OUT / 'fuse_binary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
