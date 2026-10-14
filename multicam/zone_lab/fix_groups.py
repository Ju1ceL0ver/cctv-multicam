import json
exec(open('rescore.py').read().split('def drop_gray')[0])
import lab2 as L, lab4 as Q, rules as RR
from door_seg_track import Config
def dg(tag, ev): return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
tags = ['door_20260917_08691', 'door_20260918_04917', 'door_20260918_06803', 'door_20260918_13748', 'door_20260918_15209', 'door_20260918_15826',
        'door_20260919_00095', 'door_20260919_23237', 'door_20260919_24209', 'door_20260919_24505', 'door_20260919_27663']
trtags = [t for t, v in d['tags'].items() if v['role'] == 'train']
BASE = dict(RR.BASE, shop=1, first=0, occl_legs=.6, occl_h=.6, conf=1, pair_s=3)
VAR = {'as_was': {}, 'no_inherit_number': dict(inherit_person=False), 'inherit_unique': dict(inherit_unique=True, inherit_person=False),
       'no_inherit': dict(inherit_s=0.0), 'no_inherit+no_pair': dict(inherit_s=0.0, pair_s=0)}
rows = {}
for t in tags + trtags:
    rows[t] = L.rows_of(L.track(L.load(t), Config(max_gap=25), birth=.4, cont=.25))
for name, kv in VAR.items():
    for role, ts in (('train', trtags), ('test', tags)):
        S = {'in': [0, 0, 0], 'out': [0, 0, 0]}
        for t in ts:
            ev, _ = RR.run_tag({'rows': rows[t]}, dict(BASE, **kv), tag_name=t)
            m = R.match(dg(t, ev), dg(t, truth[t]))
            for k in S:
                for i in range(3): S[k][i] += m[k][i]
        print('%-20s %s F1 %.3f %s' % (name, role, sum(2 * S[k][0] / max(1, S[k][1] + S[k][2]) for k in S) / 2, S), flush=True)
