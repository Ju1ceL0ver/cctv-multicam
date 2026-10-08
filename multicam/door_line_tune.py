"""The shop line with hysteresis (08.10.2026): the side of a person is 'inside' once its mask goes h_in px past the line,
'outside' once the whole mask stays h_out px short of it, unchanged in between; a side change needs `conf` such votes
in a row. Fitted on the /doorside stretches of the training day(s) (role 'train'), told on the others.
Needs the depth runs: door_binary_count.py predict DAY with RA_LINE_ONLY=1 RA_LINE_DEPTH=1 RA_BIN_VER=ldpc
RA_DOOR_PIECES=1 RA_POS_VER=pc (positions as for lopc). -> data/door_v2/line_tune.json"""
import itertools
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('RA_BIN_SRC', 'micro1s3')
os.environ.setdefault('RA_BIN_VER', 'ldpc')
os.environ.setdefault('RA_POS_VER', 'pc')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def stretches(day):
    import door_learn as L
    import door_side as S
    import door_v2 as D
    st = S.load(day)
    spans, truth, role = [], [], {}
    for tag, s in st.items():
        if not s.get('done'):
            continue
        s = S.migrate(tag, s)
        a = float(tag.split('_')[2])
        a = next((x for x, y in D.stretches(day) if int(round(x)) == int(a)), a)
        info = json.load(open(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'info.json'))
        spans.append((a + L.SHIFT, a + int(info['ticks']) * 0.08 + L.SHIFT, s.get('role', 'test')))
        truth += [{'kind': e['kind'], 't': a + e['t'] + L.SHIFT, 'role': s.get('role', 'test')}
                  for e in S.events(s['sides'], s.get('merge'), s.get('noperson', []))]
    return spans, truth


def main():
    import door_combo as C
    import door_learn as L
    import door_v2 as D
    f_ = json.load(open(ROOT / 'data' / 'door_v2' / 'combo_final.json'))
    u, hz = np.array(f_['u']), np.array(f_['zone'])
    days = {}
    for day in C.DAYS:
        if not (ROOT / 'data' / 'door_v2' / ('binary_%s%s.json' % (C.SUF, day))).exists():
            continue
        dd = C.Day(day)
        raw = {tag: [list(o) for o in st['obs']] for tag, st in dd.S.items()}
        days[day] = (dd, raw) + stretches(day)

    def run(h_in, h_out, conf, gate=True):
        """{role: [hit_in, pred_in, true_in, hit_out, pred_out, true_out]} over all days"""
        tot = {}
        for day, (dd, raw, spans, truth) in days.items():
            for tag, st in dd.S.items():
                st['obs'] = [[t, pid, 1.0 if d >= h_in else (0.0 if d <= -h_out else 0.5)] for t, pid, d in raw[tag]]
            dd.cache = {}
            ev = dd.binary_events(conf, 0.95, tuple(f_['stitch']))
            B = C.gate(ev, u, hz, f_['move'], f_['rad']) if gate else ev
            for role in ('train', 'test'):
                sp = [(x, y) for x, y, r in spans if r == role]
                inside = lambda t: any(x - 1 <= t <= y + 1 for x, y in sp)
                tr = [t for t in truth if t['role'] == role]
                m = D.match([e for e in B if inside(e['t'])], tr, L.TOL)
                v = tot.setdefault(role, [0] * 6)
                for i, (k, part) in enumerate(itertools.product(('in', 'out'), ('hit', 'pred', 'true'))):
                    v[i] += m[k][part]
        return tot

    def f1(v):
        fi = 2 * v[0] / max(1, v[1] + v[2])
        fo = 2 * v[3] / max(1, v[4] + v[5])
        return round((fi + fo) / 2, 3)

    grid = list(itertools.product((0, 5, 10, 20, 30, 50), (0, 5, 10, 20, 30, 50), (1, 2, 3, 4, 6), (True, False)))
    res = []
    for h_in, h_out, conf, g in grid:
        tot = run(h_in, h_out, conf, g)
        res.append({'h_in': h_in, 'h_out': h_out, 'conf': conf, 'gate': g, 'train': tot.get('train'), 'test': tot.get('test'),
                    'f1_train': f1(tot['train']) if 'train' in tot else None, 'f1_test': f1(tot['test']) if 'test' in tot else None})
    base = next(r for r in res if r['h_in'] == 0 and r['h_out'] == 0 and r['conf'] == f_['conf'] and r['gate'])
    best = max(res, key=lambda r: (r['f1_train'] or 0, -r['h_in'] - r['h_out']))
    print('now (no hysteresis, conf %d, gate):' % f_['conf'], json.dumps(base), flush=True)
    print('best on train:', json.dumps(best), flush=True)
    top = sorted(res, key=lambda r: -(r['f1_train'] or 0))[:10]
    for r in top:
        print('  ', json.dumps(r), flush=True)
    json.dump({'now': base, 'best': best, 'all': res}, open(ROOT / 'data' / 'door_v2' / 'line_tune.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
