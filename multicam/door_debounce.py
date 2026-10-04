"""The owner's rule (04.10.2026): a track's side changes only when the /inout classifier has said the new side steadily for a
while; the crossing is the moment the new side began. No learning beyond two numbers picked on the other days.

Per track: the side at every tick from io_mix (the crop + place classifiers; doorway counted as outside, or as
nobody's); the state flips when, in the last W seconds, at least FRAC of the ticks said the other side (and the
window spans at least W*0.8 s of the track); the event time is the first tick of the new side inside that window.

usage: door_debounce.py  -> data/door_v2/debounce.out"""
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
D = ROOT / 'data' / 'door_v2'
RUNS = {'model': {'20260917': '20260917_v2_m_clips_best_q', '20260918': '20260918_v2_m_clips_best', '20260919': '20260919_v2_m_clips_best_p'},
        'sam': {d: '%s_sam31' % d for d in ('20260917', '20260918', '20260919')}}


def events(by, W, frac, door_out):
    out = []
    for w, seq in by.items():
        ts = np.array([t for t, _ in seq])
        z = np.array([int(np.argmax(q.get('io_mix', q['io']))) for _, q in seq])
        side = np.where(z == 1, 1, np.where(z == 0, 0, 0 if door_out else -1))     # -1: doorway, nobody's
        state = None
        j0 = 0
        for i in range(len(ts)):
            while ts[i] - ts[j0] > W:
                j0 += 1
            win = side[j0:i + 1]
            win = win[win >= 0]
            if ts[i] - ts[j0] < 0.8 * W or len(win) < 3:
                continue
            ones = (win == 1).mean()
            maj = 1 if ones >= frac else 0 if (1 - ones) >= frac else None
            if maj is None:
                continue
            if state is None:
                state = maj
            elif maj != state:
                k = next(k for k in range(j0, i + 1) if side[k] == maj)      # the moment the new side began
                out.append({'w': w, 'kind': 'in' if maj == 1 else 'out', 't': float(ts[k])})
                state = maj
    return out


def main():
    import door_learn as L
    import door_v2 as Dv
    skip_env = os.environ.get('RA_DOOR_SKIP', '')
    lines = ['== skip %s' % (skip_env or '-')]
    grid = list(itertools.product((1.5, 2.0, 3.0, 4.0), (0.6, 0.7, 0.8, 0.9), (True, False)))
    for src, runs in RUNS.items():
        data = {}
        for day, n in runs.items():
            path = D / (n + '.jsonl.gz')
            ticks, spans = Dv._read(path)
            Dv.add_io(ticks, path)
            by = {}
            for r in ticks:
                for q in r['p']:
                    by.setdefault(q['w'], []).append((r['t'], q))
            skip = L.skipped(day)
            oo = lambda t, sk=skip: any(a - 10 <= t <= b + 10 for a, b in sk)
            inside = lambda t, sp=spans, o=oo: any(a <= t <= b for a, b in sp) and not o(t)
            data[day] = (by, [t for t in Dv.truth(day, True) if inside(t['t'])], oo)
        score = {}
        for day, (by, tr, oo) in data.items():
            for W, frac, dout in grid:
                ev = [dict(e, t=e['t'] + L.SHIFT) for e in events(by, W, frac, dout)]
                ev = [e for e in ev if not oo(e['t'])]
                m = Dv.match(ev, tr, L.TOL)
                score[(day, W, frac, dout)] = m
        for test in runs:
            train = [d for d in runs if d != test]
            f1 = lambda k: np.mean([(score[(d,) + k]['in']['f1'] + score[(d,) + k]['out']['f1']) / 2 for d in train])
            best = max(grid, key=f1)
            m = score[(test,) + best]
            lines.append('%-5s %s W=%.1fs frac=%.1f doorway=%s | in P%.2f R%.2f (%d false, %d missed) | out P%.2f R%.2f (%d false, %d missed)' % (
                src, test, best[0], best[1], 'out' if best[2] else 'none', m['in']['precision'], m['in']['recall'], m['in']['false'],
                m['in']['true'] - m['in']['hit'], m['out']['precision'], m['out']['recall'], m['out']['false'], m['out']['true'] - m['out']['hit']))
        for W in (2.0, 3.0):                                     # the owner's own numbers, for reference
            for test in runs:
                m = score[(test, W, 0.7, True)]
                lines.append('%-5s %s fixed W=%.0fs frac=0.7 doorway=out | in P%.2f R%.2f | out P%.2f R%.2f' % (
                    src, test, W, m['in']['precision'], m['in']['recall'], m['out']['precision'], m['out']['recall']))
    with open(D / 'debounce.out', 'a', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
