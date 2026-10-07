"""07.10: the door on one held-out day for several runs with one rule fitted on other days (live features only).
usage: door_cmp_day.py TEST_DAY RULE_SRC TRAIN_DAY,TRAIN_DAY RUN [RUN ...] -> data/door_v2/cmp_<day>.json"""
import json, os, sys
from pathlib import Path
for k in ('RA_DOOR_NOCNN', 'RA_DOOR_NODEPTH', 'RA_DOOR_NOLK', 'RA_DOOR_NOTAP', 'RA_DOOR_STITCH'):
    os.environ.setdefault(k, '1')
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import door_rule as DR
import door_v2 as D
OUT = ROOT / 'data' / 'door_v2'


def main(test, src, train, *runs):
    name = '%s_cmp%s' % (src, train.replace(',', '').replace('2026', ''))
    rule = DR.fit(name, ['%s=%s' % (d, OUT / ('%s_%s.jsonl.gz' % (d, src))) for d in train.split(',')])
    rep = {'test': test, 'rule': name, 'threshold': rule['thr'], 'runs': {}}
    for r in runs:
        ticks, spans = D._read(OUT / ('%s_%s.jsonl.gz' % (test, r)))
        D.add_io(ticks)
        ev = [dict(e, t=e['t'] + rule['shift']) for e in DR.apply(ticks, rule)]
        inside = lambda t: any(a <= t <= b for a, b in spans)
        truth = [t for t in D.truth(test, True) if inside(t['t'])]
        m = D.match(ev, truth, 6.0)
        rep['runs'][r] = {'in': m['in'], 'out': m['out'], 'f1': round((m['in']['f1'] + m['out']['f1']) / 2, 3),
                          'truth': len(truth), 'events': len(ev)}
        print(r, json.dumps(rep['runs'][r]), flush=True)
    json.dump(rep, open(OUT / ('cmp_%s.json' % test), 'w'), indent=1)


if __name__ == '__main__':
    main(*sys.argv[1:])
