"""14.10 night: door events of many variants on one predictions cache, scored on the owner's /doorside truth.
usage (from zone_lab, ZL_CACHE / ZL_ROWCACHE set): night_sweep.py variants.json out.json
variants: [{"id", "tracker": {...}, "cfg": {...}}]; cfg goes over lab4.BASE4. Choose by train, report test.
Counts per kind are (right, said, truth); F1 = 2 right / (said + truth)."""
import json, sys
src = open('exp_queue.py').read().split('done = set()')[0]
exec(src)
V = json.load(open(sys.argv[1])); out = []
for x in V:
    try:
        rows = rows_for(x.get('tracker', {}))
        res = {}
        for role in ('train', 'test'):
            r_ = s_ = t_ = 0
            for t in TAGS:
                if d['tags'][t]['role'] != role: continue
                ev, _ = Q.events(rows[t], dict(Q.BASE4, **x['cfg']))
                m = R.match(dg(t, ev), dg(t, truth[t]))
                for k in ('in', 'out'):
                    r_ += m[k][0]; s_ += m[k][1]; t_ += m[k][2]
            res[role] = [round(2 * r_ / max(1, s_ + t_), 4), r_, s_, t_]
        out.append(dict(x, **res)); print(x['id'], res, flush=True)
    except Exception as exc:
        out.append(dict(x, error=str(exc)[:300])); print(x['id'], 'failed', exc, flush=True)
    json.dump(out, open(sys.argv[2], 'w'), indent=0)
