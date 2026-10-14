"""a big experiment queue on doorseg_c tracks, CPU only. queue.json (re-read before every experiment): list of
{"id", "tracker": {max_gap, max_cost, ambiguity, birth, cont}, "engine": "rules"|"lab4", "cfg": {...}}.
Every result -> results.jsonl (train F1 for choosing, test F1 for reporting). Rows per tracker setting are cached."""
import hashlib, json, os, pickle, sys, time, traceback
import numpy as np
import lab2 as L, lab4 as Q, rules as RR          # before the truth: lab4 builds its model from the teacher rows
exec(open('rescore.py').read().split('def drop_gray')[0])
from door_seg_track import Config
from sklearn.neighbors import KNeighborsRegressor

def dg(tag, ev): return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
TAGS = [t for t in d['tags'] if os.path.exists(os.path.join(os.environ['ZL_CACHE'], t + '.npz'))]
# expected single-person width at a head position, from the teacher's clear people (legs uncovered) on train stretches
X, Y = [], []
for t, v in d['tags'].items():
    if v['role'] != 'train': continue
    for r in v['rows']:
        if r[RR.LEGS] < .05 and r[RR.TRU] in ('in', 'out'):
            X.append([(r[RR.X1] + r[RR.X2]) / 2, r[RR.Y1]]); Y.append(r[RR.X2] - r[RR.X1])
WK = KNeighborsRegressor(25).fit(np.array(X), np.array(Y))
RR.WIDTH = lambda r: float(WK.predict([[(r[RR.X1] + r[RR.X2]) / 2, r[RR.Y1]]])[0])
RC = os.environ.get('ZL_ROWCACHE', 'rowcache')      # 14.10: one per predictions cache
os.makedirs(RC, exist_ok=True)

def rows_for(tr):
    key = hashlib.md5(json.dumps(tr, sort_keys=True).encode()).hexdigest()[:10]
    p = os.path.join(RC, key + '.pkl')
    if os.path.exists(p):
        return pickle.load(open(p, 'rb'))
    cfg = Config(**{k: v for k, v in tr.items() if k in ('max_gap', 'max_cost', 'ambiguity', 'max_step', 'centre_gate', 'gap_growth')})
    out = {t: L.rows_of(L.track(L.load(t), cfg, birth=tr.get('birth', .4), cont=tr.get('cont', .4))) for t in TAGS}
    pickle.dump(out, open(p + '.tmp', 'wb')); os.replace(p + '.tmp', p)
    return out

done = set()
if os.path.exists('results.jsonl'):
    for line in open('results.jsonl'):
        try: done.add(json.loads(line)['id'])
        except Exception: pass
while True:
    q = json.load(open('queue.json'))
    todo = [x for x in q if x['id'] not in done]
    if not todo:
        if os.path.exists('STOP'): break
        time.sleep(30); continue
    x = todo[0]
    t0 = time.time()
    try:
        rows = rows_for(x.get('tracker', {}))
        res = {}
        for role in ('train', 'test'):
            S = {'in': [0, 0, 0], 'out': [0, 0, 0]}
            for t in TAGS:
                if d['tags'][t]['role'] != role: continue
                if x['engine'] == 'lab4':
                    ev, _ = Q.events(rows[t], dict(Q.BASE4, **x['cfg']))
                else:
                    ev, _ = RR.run_tag({'rows': rows[t]}, dict(RR.BASE, **x['cfg']), tag_name=t)
                m = R.match(dg(t, ev), dg(t, truth[t]))
                for k in S:
                    for i in range(3): S[k][i] += m[k][i]
            res[role] = round(sum(2 * S[k][0] / max(1, S[k][1] + S[k][2]) for k in S) / 2, 4)
            res[role + '_counts'] = S
        rec = dict(x, **res, sec=round(time.time() - t0, 1))
    except Exception as e:
        rec = dict(x, error=traceback.format_exc()[-600:])
    with open('results.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    done.add(x['id'])
    print(x['id'], rec.get('train'), rec.get('test'), rec.get('error', '')[:200], flush=True)
