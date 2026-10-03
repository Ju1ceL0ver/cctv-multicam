"""Pick the door rule on one day, report it on another (03.10.2026).

usage: door_v2_final.py TUNE_RUN.jsonl.gz TEST_RUN.jsonl.gz  -> data/door_v2/final_<test day>.json"""
import gzip
import itertools
import json
import sys
from pathlib import Path

import door_v2 as D

ROOT = Path(__file__).resolve().parent


def load(path):
    ticks, spans = D._read(path)
    D.add_io(ticks, path)
    day = json.loads(gzip.open(path, 'rt').readline())['day']
    inside = lambda t: any(a <= t <= b for a, b in spans)
    return day, ticks, spans, inside, [t for t in D.truth(day, True) if inside(t['t'])]


def main(tune, test, tol=6.0):
    day_a, ta, _, _, tra = load(tune)
    grid = []
    for how, mi, mo, mm in itertools.product(('io', 'io_cnn', 'io_mix', 'zone_hd'), (0.5, 1.0, 1.5, 2.0, 3.0), (0.5, 1.0, 2.0), (0, 60, 120, 200)):
        m = D.match(D.crossings(ta, how, mi, mo, mm), tra, tol)
        grid.append({'how': how, 'min_in': mi, 'min_out': mo, 'min_move': mm, 'f1': (m['in']['f1'] + m['out']['f1']) / 2, 'in': m['in'], 'out': m['out']})
    grid.sort(key=lambda g: -g['f1'])
    best = grid[0]
    day_b, tb, spans_b, inside_b, trb = load(test)
    rep = {'tune_day': day_a, 'test_day': day_b, 'tol_s': tol, 'chosen': {k: best[k] for k in ('how', 'min_in', 'min_out', 'min_move')},
           'tune_f1': round(best['f1'], 3), 'tune_top5': [{k: g[k] for k in ('how', 'min_in', 'min_out', 'min_move', 'f1')} for g in grid[:5]],
           'test_minutes': round(sum(b - a for a, b in spans_b) / 60, 1)}
    c = best
    for tl in (3.0, 6.0):
        cs = D.crossings(tb, c['how'], c['min_in'], c['min_out'], c['min_move'])
        rep['model_tol%d' % tl] = D.match(cs, trb, tl)
        rep['model_visits_tol%d' % tl] = D.match(D.undither(cs), [t for t in D.truth(day_b, False) if inside_b(t['t'])], tl)
    # the best single 'how' each, for the picture
    for how in ('io', 'io_cnn', 'io_mix', 'zone_hd'):
        g = next(g for g in grid if g['how'] == how)
        rep['test_best_of_' + how] = D.match(D.crossings(tb, how, g['min_in'], g['min_out'], g['min_move']), trb, tol)
    rep['counter'] = D.match([x for x in D.counter(day_b) if inside_b(x['t'])], trb, 1.0)
    rep['counter_visits'] = D.match([x for x in D.counter(day_b) if inside_b(x['t'])], [t for t in D.truth(day_b, False) if inside_b(t['t'])], 1.0)
    json.dump(rep, open(ROOT / 'data' / 'door_v2' / ('final_%s.json' % day_b), 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
