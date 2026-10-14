import itertools, json, os, sys, pickle
import numpy as np
exec(open('rescore.py').read().split('def drop_gray')[0])
import lab2 as L, lab3 as T3
from door_seg_track import Config
def drop_gray(tag, ev):
    return [e for e in ev if not any(abs(e['t'] - g) <= 6 for g in gray.get(tag, []))]
tags = [t for t in d['tags'] if os.path.exists(os.path.join(os.environ.get('ZL_CACHE', 'dsc_cache'), '%s.npz' % t))]
_tr = {}
def tracks(tag, gap, cont):
    k = (tag, gap, cont)
    if k not in _tr:
        _tr[k] = L.track(L.load(tag), Config(max_gap=gap), birth=.4, cont=cont)
    return _tr[k]
def score(cfg, gap, cont, roles=('train',), detail=False):
    S = {'in': [0, 0, 0], 'out': [0, 0, 0]}
    for tag in tags:
        if d['tags'][tag]['role'] not in roles:
            continue
        ev, _ = T3.events(tracks(tag, gap, cont), cfg)
        m = R.match(drop_gray(tag, ev), drop_gray(tag, truth[tag]))
        if detail and (m['in'][0] < max(m['in'][1:]) or m['out'][0] < max(m['out'][1:])):
            print('  ', tag, m)
        for k in S:
            for i in range(3): S[k][i] += m[k][i]
    f = sum(2 * S[k][0] / max(1, S[k][1] + S[k][2]) for k in S) / 2
    return round(f, 3), S
if __name__ == '__main__':
    print(len(tags), 'stretches with cache:', sorted(tags))
