"""Staff by a gallery of the day (08.10.2026), the way people counters exclude staff without uniforms: a few known staff
views of the day (they wear the same clothes all day), every other person of that day compared with them by appearance.
On the owner's /staff answers: per day, the K earliest staff views are the gallery, the rest of the day is scored; the
threshold is chosen on the other days. Next to it the role model (cross-day MLP, as /staff shows) and both together.
-> data/staff/gallery_test.json

usage: staff_gallery_test.py   (CPU)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'staff'


def time_of(meta):
    """seconds of the day for draft frames 'DAY_cam1_HHMMSS_NNNN_SEC'; None for the counter's snapshots"""
    f = meta.get('frame')
    if not f:
        return None
    p = f.split('_')
    try:
        hh, mm, ss = int(p[2][:2]), int(p[2][2:4]), int(p[2][4:6])
        return hh * 3600 + mm * 60 + ss + 900 * 0 + float(p[4])
    except (ValueError, IndexError):
        return None


def main():
    from sklearn.metrics import roc_auc_score
    from sklearn.neural_network import MLPClassifier
    import staff
    L = json.load(open(OUT / 'labels.json', encoding='utf-8'))
    M = json.load(open(OUT / 'people.json', encoding='utf-8'))
    z = np.load(OUT / 'masked_emb.npz')
    at = {str(i): k for k, i in enumerate(z['ids'])}
    nrm = lambda a: a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    V = np.concatenate([nrm(z['cloth'].astype(np.float32)), nrm(z['shape'].astype(np.float32))], 1) / np.sqrt(2)
    F = staff.feats(str(ROOT))
    ids = [i for i, v in L.items() if v['label'] in (1, 2) and i in at and i in M and i in F['row']]
    day = np.array([M[i]['day'] for i in ids])
    y = np.array([L[i]['label'] == 2 for i in ids])
    t = np.array([time_of(M[i]) if time_of(M[i]) is not None else 1e9 for i in ids])
    X = V[[at[i] for i in ids]]
    Xm = F['X'][[F['row'][i] for i in ids]]
    days = sorted(set(day))
    # the role model, each day by a model of the other days (as /staff's own check)
    pm = np.zeros(len(ids))
    for d in days:
        te = day == d
        pm[te] = MLPClassifier((256, 64), max_iter=400, early_stopping=True, random_state=0).fit(Xm[~te], y[~te]).predict_proba(Xm[te])[:, 1]
    res = {'n': len(ids), 'staff': int(y.sum()), 'days': len(days)}
    import os
    mode = os.environ.get('RA_GAL_SEED', 'owner')     # owner: the owner's staff answers; model: the role model's p >= SEED_P; first: anyone
    seed_p = float(os.environ.get('RA_GAL_P', '0.7'))
    res['seed'] = mode
    for K in (1, 3, 5, 10, 20):
        sims = np.full(len(ids), np.nan)
        enrolled = np.zeros(len(ids), bool)
        for d in days:
            idx = np.where(day == d)[0]
            st = idx[y[idx]] if mode == 'owner' else (idx[pm[idx] >= seed_p] if mode == 'model' else idx)
            if len(st) < K + 2:
                continue
            g = st[np.argsort(t[st], kind='stable')[:K]]   # the K earliest staff views of the day
            enrolled[g] = True
            rest = [i for i in idx if i not in set(g)]
            S = X[rest] @ X[g].T
            sims[rest] = np.sort(S, 1)[:, -min(3, K):].mean(1)
        ok = ~np.isnan(sims) & ~enrolled
        if ok.sum() < 50:
            continue
        auc = roc_auc_score(y[ok], sims[ok])
        # threshold by the other days, scored on each day
        hit = tot = s_c = c_s = 0
        hit2 = s_c2 = c_s2 = 0
        for d in days:
            te = ok & (day == d)
            tr = ok & (day != d)
            if not te.any() or not tr.any():
                continue
            ths = np.unique(np.round(sims[tr], 3))
            accs = [(np.mean((sims[tr] >= th) == y[tr]), th) for th in ths[::max(1, len(ths) // 200)]]
            th = max(accs)[1]
            q = sims[te] >= th
            hit += int((q == y[te]).sum()); tot += int(te.sum())
            s_c += int((~q & y[te]).sum()); c_s += int((q & ~y[te]).sum())
            q2 = q | (pm[te] >= 0.5)                      # staff when the gallery OR the model says so
            hit2 += int((q2 == y[te]).sum()); s_c2 += int((~q2 & y[te]).sum()); c_s2 += int((q2 & ~y[te]).sum())
        qm = pm[ok] >= 0.5
        res['K=%d' % K] = {'scored': int(ok.sum()), 'auc': round(float(auc), 3),
                           'gallery_acc': round(hit / max(1, tot), 4), 'staff_as_customer': s_c, 'customer_as_staff': c_s,
                           'model_acc_same_views': round(float(np.mean(qm == y[ok])), 4),
                           'model_staff_as_customer': int((~qm & y[ok]).sum()), 'model_customer_as_staff': int((qm & ~y[ok]).sum()),
                           'gallery_or_model_acc': round(hit2 / max(1, tot), 4), 'or_staff_as_customer': s_c2, 'or_customer_as_staff': c_s2}
        print('K=%d' % K, json.dumps(res['K=%d' % K]), flush=True)
    json.dump(res, open(OUT / ('gallery_test_%s.json' % mode), 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main()
