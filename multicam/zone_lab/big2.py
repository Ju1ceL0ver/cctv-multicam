"""per tag: tracker variant -> events of three engines; nothing big kept in memory"""
import json, sys, gc
import eval3 as E, lab2 as L, lab3 as T3, lab4 as Q, rules as R
from door_seg_track import Config
RULES = dict(R.BASE, shop=1, first=0, occl_legs=.6, occl_h=.6, conf=1, pair_s=3)
ENG = {'rules': RULES, 'lab3_W3': T3.BASE3, 'lab3_W2': dict(T3.BASE3, W=2.), 'lab3_W4': dict(T3.BASE3, W=4.),
       'lab4': dict(Q.BASE4, SM=0., thr=.7)}
out = {}
for gap, cont in [(int(sys.argv[1]), float(sys.argv[2]))]:
    for tag in E.tags:
        tr = L.track(L.load(tag), Config(max_gap=gap), birth=.4, cont=cont)
        rows = L.rows_of(tr)
        for name, c in ENG.items():
            ev = R.run_tag({'rows': rows}, c)[0] if name == 'rules' else (T3.events(tr, c)[0] if name.startswith('lab3') else Q.events(rows, c)[0])
            out.setdefault('%d_%s_%s' % (gap, cont, name), {})[tag] = [{'kind': e['kind'], 't': float(e['t'])} for e in ev]
        del tr, rows; gc.collect()
    print('done', gap, cont, flush=True)
    json.dump(out, open('big2_events_%s_%s.json' % (sys.argv[1], sys.argv[2]), 'w'))
