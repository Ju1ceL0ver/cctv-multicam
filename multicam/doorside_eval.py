"""The owner's /doorside marks (stretches marked 'done') as the test (07.10.2026): entries and exits of the door rule,
the owner's binary model (+ movement + zone), their combination on agreement and the old /door answers, on the same
stretches, against his per-person sides. Tracks: the small SAM (micro1s3) and SAM 3.1. -> data/door_side/eval.json"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('RA_BIN_SRC', 'micro1s3')
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import door_combo as C
    import door_learn as L
    import door_side as S
    import door_v2 as D
    rep = {}
    for f in sorted((ROOT / 'data' / 'door_side').glob('2026????.json')):
        day = f.stem
        st = S.load(day)
        done = {t: S.migrate(t, s) for t, s in st.items() if s.get('done') and s.get('role', 'test') == os.environ.get('RA_SIDE_ROLE', s.get('role', 'test'))}
        if not done:
            continue
        spans, truth = [], []
        for tag, s in done.items():
            a = float(tag.split('_')[2])
            a = next((x for x, y in D.stretches(day) if int(round(x)) == int(a)), a)
            info = json.load(open(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'info.json'))
            spans.append((a + L.SHIFT, a + int(info['ticks']) * 0.08 + L.SHIFT))
            truth += [{'kind': e['kind'], 't': a + e['t'] + L.SHIFT} for e in S.events(s['sides'], s.get('merge'), s.get('noperson', []))]
        inside = lambda t: any(x - 1 <= t <= y + 1 for x, y in spans)
        out = {'truth_in': sum(t['kind'] == 'in' for t in truth), 'truth_out': sum(t['kind'] == 'out' for t in truth)}
        for src in ('micro1s3', 'sam31'):
            C.SRC, C.POS_SUF = src, ('' if src == 'sam31' else src + '_') + os.environ.get('RA_POS_VER', '')
            C.SUF = ('' if src == 'sam31' else src + '_') + os.environ.get('RA_BIN_VER', '')
            if not (ROOT / 'data' / 'door_v2' / ('binary_%s%s.json' % (C.SUF, day))).exists():
                continue
            dd = C.Day(day)
            f_ = json.load(open(ROOT / 'data' / 'door_v2' / 'combo_final.json'))
            u, hz = np.array(f_['u']), np.array(f_['zone'])
            B = C.gate(dd.binary_events(f_['conf'], f_['thr'], tuple(f_['stitch'])), u, hz, f_['move'], f_['rad'])
            variants = {'combination (agreement)': C.combine(B, dd.rule, f_['lo'], f_['hi']),
                        'owner model alone': B,
                        'door rule alone (p>=0.5)': [r for r in dd.rule if r['p'] >= 0.5]}
            for name, ev in variants.items():
                m = D.match([e for e in ev if inside(e['t'])], truth, L.TOL)
                out['%s / %s' % (src, name)] = {'in': [m['in'][k] for k in ('hit', 'pred', 'true')], 'out': [m['out'][k] for k in ('hit', 'pred', 'true')],
                                                'f1': round((m['in']['f1'] + m['out']['f1']) / 2, 3)}
        old = [t for t in D.truth(day, True) if inside(t['t']) and t['kind'] in ('in', 'out')]
        m = D.match(old, truth, L.TOL)
        out['old /door answers'] = {'in': [m['in'][k] for k in ('hit', 'pred', 'true')], 'out': [m['out'][k] for k in ('hit', 'pred', 'true')],
                                    'f1': round((m['in']['f1'] + m['out']['f1']) / 2, 3)}
        rep[day] = dict(out, stretches=sorted(done))
        if os.environ.get('RA_SIDE_ROLE'):
            pass
        for k, v in out.items():
            print(day, k, v, flush=True)
    json.dump(rep, open(ROOT / 'data' / 'door_side' / ('eval%s.json' % os.environ.get('RA_BIN_VER', '')), 'w'), indent=1)


if __name__ == '__main__':
    main()
