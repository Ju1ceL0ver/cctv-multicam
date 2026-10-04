"""Two trackers, one door: fuse the door rule's events on the model's tracks and on SAM 3.1's tracks (04.10.2026).

1. door_learn.py runs on each track source (its held-out-day scores are kept as learn_preds_<day>_<src>.json);
2. events of both sources of one direction within JOIN_S are one event; its numbers: each source's best score and
   count there, whether both saw it, the time between them;
3. a second-level classifier, trained on two days, scores the third; the threshold comes from the training days.
Also printed for reference: each source alone, "both must agree", "either is enough", the mean score.

usage: door_fuse.py [--skip '20260918 15:06-15:29']  -> data/door_v2/fuse.out, fuse.json"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
D = ROOT / 'data' / 'door_v2'
DAYS = ('20260917', '20260918', '20260919')
RUNS = {'model': {'20260917': '20260917_v2_m_clips_best_q', '20260918': '20260918_v2_m_clips_best', '20260919': '20260919_v2_m_clips_best_p'},
        'sam': {d: '%s_sam31' % d for d in DAYS}}
JOIN_S = 4.0


def scores(src, env):
    args = ['%s=%s' % (d, D / (n + '.jsonl.gz')) for d, n in RUNS[src].items()]
    r = subprocess.run([sys.executable, 'door_learn.py'] + args, cwd=str(ROOT), capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(r.stderr[-2000:])
    for d in DAYS:
        shutil.copy(D / ('learn_preds_%s.json' % d), D / ('learn_preds_%s_%s.json' % (d, src)))
    return r.stdout


def events(day):
    """Fused events of a day: [(features, kind, t)]. A model event and a SAM event of one direction are one event
    when they pair up one to one within JOIN_S (Hungarian on the time gap); within one source nothing is merged --
    two people entering together are two events in each source."""
    from scipy.optimize import linear_sum_assignment
    out = []
    for kind in ('in', 'out'):
        A = [p for p in json.load(open(D / ('learn_preds_%s_model.json' % day))) if p['kind'] == kind]
        B = [p for p in json.load(open(D / ('learn_preds_%s_sam.json' % day))) if p['kind'] == kind]
        pairs = []
        if A and B:
            C = np.array([[abs(a['t'] - b['t']) for b in B] for a in A])
            r, c = linear_sum_assignment(np.where(C <= JOIN_S, C, 1e6))
            pairs = [(i, j) for i, j in zip(r, c) if C[i, j] <= JOIN_S]
        ia, ib = {i for i, _ in pairs}, {j for _, j in pairs}
        near = lambda t, L: float(sum(abs(x['t'] - t) <= JOIN_S for x in L))
        def feat(a, b):
            t = a['t'] if a and (not b or a['p'] >= b['p']) else b['t']
            return ({'model_p': a['p'] if a else 0.0, 'sam_p': b['p'] if b else 0.0, 'both': float(bool(a and b)),
                     'dt': abs(a['t'] - b['t']) if a and b else -1.0, 'model_near': near(t, A), 'sam_near': near(t, B),
                     'out': float(kind == 'out')}, kind, t)
        out += [feat(A[i], B[j]) for i, j in pairs]
        out += [feat(A[i], None) for i in range(len(A)) if i not in ia]
        out += [feat(None, B[j]) for j in range(len(B)) if j not in ib]
    return out


def main():
    import door_learn as L
    import door_v2 as D2
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='-1')
    rep, lines = {}, []
    if os.environ.get('RA_FUSE_REUSE') != '1':          # 1: the per-source scores of the last run are kept
        lines.append(scores('model', env))
        lines.append(scores('sam', env))
    errors = []
    days = {}
    for day in DAYS:
        _, spans = D2._read(D / (RUNS['model'][day] + '.jsonl.gz'))
        skip = L.skipped(day)
        oo = lambda t, sk=skip: any(a - 10 <= t <= b + 10 for a, b in sk)
        inside = lambda t, sp=spans, o=oo: any(a <= t <= b for a, b in sp) and not o(t)
        truth = [t for t in D2.truth(day, True) if inside(t['t'])]
        ev = [e for e in events(day) if not oo(e[2])]
        cands = [(f, k, t, 0) for f, k, t in ev]
        days[day] = {'ev': ev, 'truth': truth, 'y': L.label(cands, truth)}
    names = sorted(days[DAYS[0]]['ev'][0][0])
    X = {d: np.array([[f[n] for n in names] for f, _, _ in v['ev']], np.float32) for d, v in days.items()}
    res = {}
    for test in DAYS:
        train = [d for d in DAYS if d != test]
        Xt = np.concatenate([X[d][days[d]['y'] >= 0] for d in train]); yt = np.concatenate([days[d]['y'][days[d]['y'] >= 0] for d in train])
        clf = L.model().fit(Xt, yt)
        # threshold: each training day scored by a model of the other training day
        best_thr, best_f = 0.5, -1
        oof = []
        for d in train:
            o = [e for e in train if e != d][0]
            m = L.model().fit(X[o][days[o]['y'] >= 0], days[o]['y'][days[o]['y'] >= 0])
            oof.append((d, m.predict_proba(X[d])[:, 1]))
        for thr in np.arange(0.05, 0.96, 0.05):
            fs = []
            for d, p in oof:
                pred = [{'kind': k, 't': t, 'w': 0} for (f, k, t), pp in zip(days[d]['ev'], p) if pp >= thr]
                mm = D2.match(pred, days[d]['truth'], L.TOL)
                fs.append((mm['in']['f1'] + mm['out']['f1']) / 2)
            if np.mean(fs) > best_f:
                best_f, best_thr = float(np.mean(fs)), float(thr)
        p = clf.predict_proba(X[test])[:, 1]
        ev, tr = days[test]['ev'], days[test]['truth']
        rules = {'fused': [pp >= best_thr for pp in p],
                 'both agree': [f['both'] > 0 and f['model_p'] >= 0.2 and f['sam_p'] >= 0.2 for f, _, _ in ev],
                 'either': [max(f['model_p'], f['sam_p']) >= 0.35 for f, _, _ in ev],
                 'mean score': [(f['model_p'] + f['sam_p']) / 2 >= 0.25 for f, _, _ in ev]}
        res[test] = {'threshold': round(best_thr, 2)}
        for name, keep in rules.items():
            pred = [{'kind': k, 't': t, 'w': 0, 'p': float(pp), 'f': f} for (f, k, t), kk, pp in zip(ev, keep, p) if kk]
            mm = D2.match(pred, tr, L.TOL)
            if name == 'fused':
                from scipy.optimize import linear_sum_assignment
                for kind in ('in', 'out'):
                    P = [x for x in pred if x['kind'] == kind]; T = [x for x in tr if x['kind'] == kind]
                    hp, ht = set(), set()
                    if P and T:
                        C = np.array([[abs(a['t'] - b['t']) for b in T] for a in P])
                        rr, cc = linear_sum_assignment(np.where(C <= L.TOL, C, 1e6))
                        for i, j in zip(rr, cc):
                            if C[i, j] <= L.TOL:
                                hp.add(i); ht.add(j)
                    neutral = [x['t'] for x in tr if x['kind'] == 'any']
                    errors += [{'day': test, 'what': 'false', 'kind': kind, 't': x['t'], 'w': None, 'p': round(x['p'], 3),
                                'model_p': x['f']['model_p'], 'sam_p': x['f']['sam_p']} for i, x in enumerate(P)
                               if i not in hp and not any(abs(x['t'] - n) <= L.TOL for n in neutral)]
                    errors += [{'day': test, 'what': 'missed', 'kind': kind, 't': round(x['t'], 2), 'w': None, 'event': x.get('id')}
                               for j, x in enumerate(T) if j not in ht]
            res[test][name] = {k: [mm[k]['precision'], mm[k]['recall'], mm[k]['false'], mm[k]['true'] - mm[k]['hit']] for k in ('in', 'out')}
            lines.append('%s %-11s in P%.2f R%.2f (%d false, %d missed) | out P%.2f R%.2f (%d false, %d missed)' % (
                test, name, mm['in']['precision'], mm['in']['recall'], mm['in']['false'], mm['in']['true'] - mm['in']['hit'],
                mm['out']['precision'], mm['out']['recall'], mm['out']['false'], mm['out']['true'] - mm['out']['hit']))
    json.dump(res, open(D / 'fuse.json', 'w'), indent=1)
    json.dump(errors, open(D / 'fuse_errors.json', 'w'), indent=1)
    with open(D / 'fuse.out', 'a', encoding='utf-8') as f:
        f.write('\n== skip %s\n' % os.environ.get('RA_DOOR_SKIP', '-') + '\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
