"""The role model learns from the owner's /liveevents answers, day after day (09.10.2026).

Every live event the owner answered about its role -- 1 'right' (the shown role is right) or 4 'role wrong' (it is the
other one) -- gives one labelled person: the event's tracks, and up to KEEP of their best views kept by the live door
(data/live/role_views/<run>/views.json, door_role.Bank.dump). Their ReID vectors (the same teachers and masked crops as
the live role) join the model's training set (data/staff/refit_guarded_20261008/training.npz) as samples of that day,
and the same recipe (SVM C10 gamma .5, threshold .45 -- staff_refit.py) is fitted again.

The new model replaces data/staff/current_role.pkl only when, on the owner's live days, it is not worse than the current
one, each day counted by a model that did not see that day (leave one live day out). The previous model is kept next to
the report. Nothing happens with fewer than MIN_PEOPLE answered people.

usage (venv_rfdetr, the card for the vectors, a few minutes): staff_live_learn.py [--dry]
-> data/staff/live_learn/<stamp>/report.json (+ previous_current_role.pkl), data/staff/live_learn/vectors.npz (cache)"""
import hashlib
import json
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIVE = ROOT / 'data' / 'live'
STAFF = ROOT / 'data' / 'staff'
OUT = STAFF / 'live_learn'
BASE = STAFF / 'refit_guarded_20261008' / 'training.npz'
RECIPE = dict(kind='svm', C=10, gamma=.5)
THRESHOLD = .45
KEEP = 8
MIN_PEOPLE = 10


def answered(day, live=None):
    """{tuple(track keys): staff bool} -- the owner's word on the role of the day's events; a person answered both ways
    is left out"""
    import live_events as LE
    if live is not None:
        LE.LIVE, LE.REVIEW = live, live / 'review'
    st = LE.load(day)
    said = {}
    for e in LE.events(day):
        v = (st.get(LE.key_of(e)) or {}).get('v')
        if v not in (1, 4) or not e.get('tracks') or e.get('role') not in ('staff', 'customer'):
            continue
        staff = (e['role'] == 'staff') == (v == 1)
        keys = tuple(e['tracks'])
        said.setdefault(keys, set()).add(staff)
    return {k: v.pop() for k, v in said.items() if len(v) == 1}


def day_views(day, live=None):
    """track key -> [(score, path, box_1280)] over every live run of the day"""
    out = {}
    for d in sorted(((live or LIVE) / 'role_views').glob(day + '_*')):
        f = d / 'views.json'
        if not f.exists():
            continue
        for k, v in json.load(open(f, encoding='utf-8')).items():
            out.setdefault(k, []).extend((float(s), str(d / name), [float(x) for x in b]) for s, name, b in v)
    return out


def samples(days, live=None):
    """[(day, person index, staff, path, box)] -- up to KEEP best views per answered person"""
    rows, n_people = [], 0
    for day in days:
        views = day_views(day, live)
        for keys, staff in answered(day, live).items():
            v = sorted((x for k in keys for x in views.get(k, [])), key=lambda z: -z[0])[:KEEP]
            v = [x for x in v if os.path.exists(x[1])]
            if not v:
                continue
            for _, path, box in v:
                rows.append((day, n_people, bool(staff), path, box))
            n_people += 1
    return rows, n_people


def vectors(paths, embed=None):
    """(cloth, shape) of each view, cached by path in OUT/vectors.npz"""
    import cv2
    cache = OUT / 'vectors.npz'
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        have = {p: (c, s) for p, c, s in zip(z['paths'], z['cloth'], z['shape'])}
    new = [p for p in dict.fromkeys(paths) if p not in have]
    if new:
        if embed is None:
            import track_emb as T
            embed = T.teachers()
        for k in range(0, len(new), 64):
            part = new[k:k + 64]
            crops = [cv2.imread(p) for p in part]
            ok = [i for i, c in enumerate(crops) if c is not None]
            v1, v2 = embed([crops[i] for i in ok])
            for j, i in enumerate(ok):
                have[part[i]] = (np.asarray(v1[j], np.float32), np.asarray(v2[j], np.float32))
        OUT.mkdir(parents=True, exist_ok=True)
        ps = sorted(have)
        tmp = OUT / 'vectors.tmp.npz'
        np.savez_compressed(tmp, paths=np.array(ps), cloth=np.stack([have[p][0] for p in ps]), shape=np.stack([have[p][1] for p in ps]))
        os.replace(tmp, cache)
    keep = [p for p in paths if p in have]
    return keep, np.stack([have[p][0] for p in keep]) if keep else None, np.stack([have[p][1] for p in keep]) if keep else None


def fit(c, s, boxes, cam, y):
    from staff_model import Features, RoleModel, StaffClassifier, geometry, norm
    f = Features().fit(c, s)
    x = f.transform(c, s, geometry(boxes, cam))
    raw = np.concatenate([norm(c), norm(s)], 1) / np.sqrt(2)
    return StaffClassifier(f, [RoleModel(RECIPE).fit(x, y, raw)], [1.0], THRESHOLD)


def score(y, q):
    return {'n': int(len(y)), 'right': int((q == y).sum()), 'accuracy': round(float((q == y).mean()), 4) if len(y) else None,
            'staff_as_customer': int((~q & y).sum()), 'customer_as_staff': int((q & ~y).sum())}


def main(dry=False, live=None, base=BASE, current=None, embed=None, out_root=None):
    import staff_model
    from storage import atomic_json
    t0 = time.time()
    live = live or LIVE
    days = sorted({p.stem for p in (live / 'review').glob('20??????.json')}) if (live / 'review').exists() else []
    rows, n_people = samples(days, live)
    out_root = out_root or OUT
    rep = {'days': days, 'people': n_people, 'views': len(rows), 'recipe': RECIPE, 'threshold': THRESHOLD}
    if n_people < MIN_PEOPLE:
        rep['decision'] = 'not enough answered people (%d < %d)' % (n_people, MIN_PEOPLE)
        print(json.dumps(rep), flush=True)
        return rep
    paths, c, s = vectors([r[3] for r in rows], embed)
    at = {p: i for i, p in enumerate(paths)}
    rows = [r for r in rows if r[3] in at]
    L_c = c[[at[r[3]] for r in rows]]; L_s = s[[at[r[3]] for r in rows]]
    L_b = np.array([r[4] for r in rows], np.float32); L_y = np.array([r[2] for r in rows]); L_d = np.array([r[0] for r in rows])
    L_cam = np.array(['cam1'] * len(rows))
    z = np.load(base)
    cur = current or staff_model.load_current(ROOT)
    # honest check: each live day by a model that did not see it, the current model on the same views
    old_q = cur.predict(L_c, L_s, L_b, L_cam)
    new_q = np.zeros(len(rows), bool)
    for d in sorted(set(L_d)):
        tr = L_d != d
        m = fit(np.concatenate([z['cloth'], L_c[tr]]), np.concatenate([z['shape'], L_s[tr]]),
                np.concatenate([z['boxes'], L_b[tr]]), np.concatenate([z['cam'], L_cam[tr]]), np.concatenate([z['y'], L_y[tr]]))
        new_q[~tr] = m.predict(L_c[~tr], L_s[~tr], L_b[~tr], L_cam[~tr])
    rep['current_on_live'] = score(L_y, old_q)
    rep['refit_on_live_lodo'] = score(L_y, new_q)
    rep['by_day'] = {d: {'current': score(L_y[L_d == d], old_q[L_d == d]), 'refit': score(L_y[L_d == d], new_q[L_d == d])} for d in sorted(set(L_d))}
    better = rep['refit_on_live_lodo']['right'] >= rep['current_on_live']['right']
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out = out_root / stamp
    out.mkdir(parents=True, exist_ok=True)
    if better and not dry:
        m = fit(np.concatenate([z['cloth'], L_c]), np.concatenate([z['shape'], L_s]), np.concatenate([z['boxes'], L_b]),
                np.concatenate([z['cam'], L_cam]), np.concatenate([z['y'], L_y]))
        cur_path = STAFF / 'current_role.pkl' if current is None else out / 'current_role.pkl'
        if cur_path.exists():
            shutil.copyfile(cur_path, out / 'previous_current_role.pkl')
        tmp = cur_path.with_suffix('.pkl.tmp')
        with tmp.open('wb') as h:
            pickle.dump(m, h)
        os.replace(tmp, cur_path)
        with cur_path.open('rb') as h:                      # the saved model answers as the fitted one
            again = pickle.load(h)
        assert np.array_equal(again.predict(L_c, L_s, L_b, L_cam), m.predict(L_c, L_s, L_b, L_cam))
        rep['decision'] = 'replaced'
        rep['current_model_sha256'] = hashlib.sha256(cur_path.read_bytes()).hexdigest()
        if current is None:
            atomic_json(STAFF / 'current_role.json', {'path': str(cur_path), 'sha256': rep['current_model_sha256'],
                                                     'report': str(out / 'report.json'), 'live_days': days,
                                                     'threshold': THRESHOLD, 'base': str(base)})
    else:
        rep['decision'] = 'kept the current model' + (' (dry run)' if dry else ' (the refit was worse on the live days)')
    rep['seconds'] = round(time.time() - t0, 1)
    atomic_json(out / 'report.json', rep)
    print(json.dumps({k: rep[k] for k in ('people', 'views', 'current_on_live', 'refit_on_live_lodo', 'decision')}), flush=True)
    return rep


if __name__ == '__main__':
    main(dry='--dry' in sys.argv)
