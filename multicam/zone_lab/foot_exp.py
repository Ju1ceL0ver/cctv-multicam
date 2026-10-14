"""14.10: passer-by filter (foot in the shop after an entry / before an exit) on the owner's /doorside truth.
Same scoring as exp_queue.py; live tracker setting (max_gap 25). Prints train and test F1 per variant."""
import json, os, sys
import numpy as np
src = open('exp_queue.py').read().split('done = set()')[0]
exec(src)
LIVE = dict(SM=0., thr=.6, inh_overlap=.3, door_s=2., pair_s=3.)
rows = rows_for({'max_gap': 25})
variants = [('live', {})] + [('foot%.1f_n%d%s' % (w, n, '_gal' if g else ''), dict(foot_s=w, foot_n=n, foot_gal=g))
                              for w in (1.5, 2.5, 3.5) for n in (1, 2, 3) for g in (False, True)]
if len(sys.argv) > 1 and sys.argv[1] == 'rules':
    variants = [('live', {})] + [('rules_first_shop%d' % n, dict(rules_first=n)) for n in (1, 5, 20, 50)]
if len(sys.argv) > 1 and sys.argv[1] == 'veto':
    variants = [('live', {})] + [('veto_%s%s' % (m, '' if not w else '_foot%.1f' % w), dict(shop_veto=m, **({'foot_s': w, 'foot_n': 1} if w else {})))
                                  for m in ('none', 'out') for w in (0, 1.5, 2.5)]
out = []
for name, extra in variants:
    res = {}
    for role in ('train', 'test'):
        S = {'in': [0, 0, 0], 'out': [0, 0, 0]}
        for t in TAGS:
            if d['tags'][t]['role'] != role: continue
            ev, _ = Q.events(rows[t], dict(Q.BASE4, **LIVE, **extra))
            m = R.match(dg(t, ev), dg(t, truth[t]))
            for k in S:
                for i in range(3): S[k][i] += m[k][i]
        tp = sum(S[k][0] for k in S); fp = sum(S[k][1] for k in S); fn = sum(S[k][2] for k in S)
        res[role] = (round(2 * tp / max(1, fp + fn), 3), tp, fp, fn)    # counts are (right, said, truth)
    out.append((name, res)); print(name, res, flush=True)
json.dump(out, open('foot_exp%s.json' % ('_' + sys.argv[1] if len(sys.argv) > 1 else ''), 'w'))
