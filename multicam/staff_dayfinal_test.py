"""Does the role get better by the end of the day (09.10.2026)? On the owner's /staff answers, per day, against the role
model alone (each day by a model of the other days, as /staff checks it):
  online -- a view is compared only with the gallery seeded before it (views the model is sure of, p >= SEED_P), as the
            live door does during the day;
  final  -- the same with the whole day's gallery (the 21:00 pass), the view's own minutes (+-GAP s) left out, so a
            customer's other views of the same visit do not vote for themselves;
  final2 -- final once more: the views called staff by the first pass AND p >= 0.5 seed the gallery too.
Staff when the model OR the gallery says so; the gallery threshold is chosen on the other days.
-> data/staff/dayfinal_test.json      usage: staff_dayfinal_test.py   (CPU)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'staff'
SEED_P = float(os.environ.get('RA_GAL_P', '0.7'))
GAP = float(os.environ.get('RA_GAL_GAP', '300'))


def main():
    from sklearn.neural_network import MLPClassifier
    import staff
    from staff_gallery_test import time_of
    L = json.load(open(OUT / 'labels.json', encoding='utf-8'))
    M = json.load(open(OUT / 'people.json', encoding='utf-8'))
    z = np.load(OUT / 'masked_emb.npz')
    at = {str(i): k for k, i in enumerate(z['ids'])}
    nrm = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    V = np.concatenate([nrm(z['cloth'].astype(np.float32)), nrm(z['shape'].astype(np.float32))], 1) / np.sqrt(2)
    F = staff.feats(str(ROOT))
    ids = [i for i, v in L.items() if v['label'] in (1, 2) and i in at and i in M and i in F['row'] and time_of(M[i]) is not None]
    day = np.array([M[i]['day'] for i in ids])
    y = np.array([L[i]['label'] == 2 for i in ids])
    t = np.array([time_of(M[i]) for i in ids])
    X = V[[at[i] for i in ids]]
    Xm = F['X'][[F['row'][i] for i in ids]]
    days = sorted(set(day))
    pm = np.zeros(len(ids))
    for d in days:
        te = day == d
        pm[te] = MLPClassifier((256, 64), max_iter=400, early_stopping=True, random_state=0).fit(Xm[~te], y[~te]).predict_proba(Xm[te])[:, 1]

    def sims(seed, causal):
        """mean of the top-3 cosines to the day's seeds (NaN when fewer than 5 usable)"""
        out = np.full(len(ids), np.nan)
        for d in days:
            idx = np.where(day == d)[0]
            S = X[idx] @ X[idx].T
            for a, i in enumerate(idx):
                ok = seed[idx] & (np.abs(t[idx] - t[i]) > GAP)
                if causal:
                    ok &= t[idx] < t[i]
                if ok.sum() < 5:
                    continue
                s = np.sort(S[a, ok])[-3:]
                out[i] = s.mean()
        return out

    def decide(sim):
        """threshold for the gallery by the other days; staff = model >= 0.5 OR sim >= thr"""
        q = np.zeros(len(ids), bool)
        thr = {}
        for d in days:
            tr = day != d
            best = (-1, 1.0)
            for th in np.arange(0.40, 0.90, 0.01):
                qq = (pm[tr] >= 0.5) | (np.nan_to_num(sim[tr], nan=-1) >= th)
                acc = np.mean(qq == y[tr])
                if acc > best[0]:
                    best = (acc, th)
            thr[d] = round(float(best[1]), 2)
            te = day == d
            q[te] = (pm[te] >= 0.5) | (np.nan_to_num(sim[te], nan=-1) >= best[1])
        return q, thr

    def score(q):
        return {'acc': round(float(np.mean(q == y)), 4), 'staff_as_customer': int((~q & y).sum()),
                'customer_as_staff': int((q & ~y).sum())}
    res = {'n': len(ids), 'staff': int(y.sum()), 'days': len(days), 'seed_p': SEED_P, 'gap_s': GAP}
    res['model'] = score(pm >= 0.5)
    seed = pm >= SEED_P
    s_on = sims(seed, True)
    q_on, th_on = decide(s_on)
    res['online'] = dict(score(q_on), thr=th_on)
    s_fin = sims(seed, False)
    q_fin, th_fin = decide(s_fin)
    res['final'] = dict(score(q_fin), thr=th_fin)
    seed2 = seed | (q_fin & (pm >= 0.5))
    s2 = sims(seed2, False)
    q2, th2 = decide(s2)
    res['final2'] = dict(score(q2), thr=th2)
    for k in ('model', 'online', 'final', 'final2'):
        print(k, json.dumps({a: b for a, b in res[k].items() if a != 'thr'}), flush=True)
    json.dump(res, open(OUT / 'dayfinal_test.json', 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
