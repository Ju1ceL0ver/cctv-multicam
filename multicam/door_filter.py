"""Which of the live counter's door events are real customer passages: a fast filter in front of the CRM.

The owner answered every counter event of 17, 18 and 19.09 in /door: a customer went in or out, nobody
passed (lingering at the sample stand on the door line, the same person out and back), or it was
staff. Real = a customer in or out. The filter sees only what the counter knows at the moment of the
event plus the events before it: where the feet are (the stand stands on the line), how sure the
detector and the role classifier are, and whether the same person (global_id / track) already crossed
in the last seconds or minutes, and how busy the door is. A gradient boosting, checked day by day:
trained on two days, scored on the third.

usage: door_filter.py            -> data/logs/door_filter.json (per-day precision/recall vs the counter)"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DAYS = ('20260917', '20260918', '20260919')


def truth(day, root=ROOT):
    """{event_id: 1 real customer passage, 0 not}, from the owner's /door answers."""
    import door_review
    st = door_review.labels(day, root)
    roles = st.get('roles', {})
    out = {}
    for eid, v in st['labels'].items():
        kind = v.get('kind')
        role = v.get('role') or roles.get(v.get('person') or '', None)
        out[eid] = int(kind in ('in', 'out') and role != 'staff')
    return out


def features(events):
    """One row per event, from the event itself and the ones before it (causal: usable live)."""
    rows = []
    t = np.array([e['unix_ms'] / 1000.0 for e in events])
    for k, e in enumerate(events):
        prev = [j for j in range(k) if t[k] - t[j] <= 300]
        same_g = [j for j in prev if events[j].get('global_id') == e.get('global_id')]
        same_t = [j for j in prev if events[j].get('track_id') == e.get('track_id')]
        last_g = min((t[k] - t[j] for j in same_g), default=999.0)
        last_opp = min((t[k] - t[j] for j in same_g if events[j].get('event') != e.get('event')), default=999.0)
        near = [j for j in prev if t[k] - t[j] <= 30]
        h = e.get('box_y2', 0) - e.get('box_y1', 0)
        rows.append([
            1.0 if e.get('event') == 'entry' else 0.0,
            {'customer': 0, 'guest': 0, 'staff': 1, 'employee': 1, 'passerby': 2}.get(e.get('role'), 3),
            float(bool(e.get('role_locked'))), float(e.get('staff_probability') or 0), float(e.get('detection_confidence') or 0),
            float(e.get('classifier_samples') or 0), float(e.get('foot_x') or 0), float(e.get('foot_y') or 0),
            float(h), float(e.get('box_x2', 0) - e.get('box_x1', 0)), float(bool(e.get('unmatched'))),
            min(last_g, 999.0), min(last_opp, 999.0), float(len(same_g)), float(len(same_t)), float(len(near)),
            float(len([j for j in prev if t[k] - t[j] <= 120])),
            (t[k] % 86400) / 3600.0,
        ])
    return np.array(rows, np.float32)


NAMES = ['entry', 'role', 'role_locked', 'staff_prob', 'det_conf', 'role_samples', 'foot_x', 'foot_y', 'box_h', 'box_w',
         'unmatched', 's_since_same_person', 's_since_same_person_opposite', 'same_person_5min', 'same_track_5min',
         'events_30s', 'events_120s', 'hour']


def dataset(root=ROOT):
    import door_review
    data = {}
    for day in DAYS:
        ev = door_review.day_events(day, root)
        tr = truth(day, root)
        ev = [e for e in ev if e['event_id'] in tr]
        data[day] = (ev, features(ev), np.array([tr[e['event_id']] for e in ev]))
    return data


def evaluate(root=ROOT):
    from sklearn.ensemble import HistGradientBoostingClassifier
    data = dataset(root)
    rep = {}
    for test in DAYS:
        X = np.concatenate([data[d][1] for d in DAYS if d != test]); y = np.concatenate([data[d][2] for d in DAYS if d != test])
        ev, Xt, yt = data[test]
        clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0).fit(X, y)
        p = clf.predict_proba(Xt)[:, 1]
        counter = np.array([e.get('role') not in ('staff', 'employee', 'passerby') for e in ev])     # what goes to the CRM as a guest
        res = {'events': len(yt), 'real': int(yt.sum()),
               'counter_as_guest': {'sent': int(counter.sum()), 'precision': round(float(yt[counter].mean()), 3) if counter.any() else None,
                                    'recall': round(float(counter[yt == 1].mean()), 3)}}
        for thr in (0.3, 0.4, 0.5, 0.6):
            keep = p >= thr
            res['filter_%.1f' % thr] = {'sent': int(keep.sum()), 'precision': round(float(yt[keep].mean()), 3) if keep.any() else None,
                                        'recall': round(float(keep[yt == 1].mean()), 3)}
        rep[test] = res
    X = np.concatenate([data[d][1] for d in DAYS]); y = np.concatenate([data[d][2] for d in DAYS])
    from sklearn.inspection import permutation_importance
    clf = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300, l2_regularization=1.0, random_state=0).fit(X, y)
    imp = permutation_importance(clf, X, y, n_repeats=5, random_state=0).importances_mean
    rep['importance'] = {NAMES[i]: round(float(imp[i]), 4) for i in np.argsort(-imp)[:8]}
    return rep


if __name__ == '__main__':
    os.chdir(ROOT)
    rep = evaluate()
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'door_filter.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))
