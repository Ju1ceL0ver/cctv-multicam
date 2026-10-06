"""A saved door rule (door_rule.py) on any run of a day, against the owner's truth (answers + /doormark marks)
(06.10.2026). Only the stretches the run covers; with ONLY_DONE=1 only the stretches the owner marked as complete on
/doormark.

usage: door_eval_rule.py RULE DAY=RUN.jsonl.gz [DAY=RUN ...]  -> printed P/R per day and direction, and
       data/door_v2/eval_<rule>_<run>.json with every event and miss"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(rule_name, args):
    import door_learn as L
    import door_mark as DM
    import door_rule as DR
    import door_v2 as D
    rule = DR.load(rule_name)
    out = {}
    for a in args:
        day, path = a.split('=', 1)
        ticks, spans = D._read(path)
        D.add_io(ticks, path)
        done = set(DM.load(day)['done'])
        import door_sam as DS
        if os.environ.get('ONLY_DONE') == '1':
            spans = [(x, y) for x, y in spans if DS.tag_of(day, x) in done]
        inside = lambda t: any(x - 1 <= t <= y + 1 for x, y in spans)
        skip = L.skipped(day)
        out_of = lambda t: any(x - 10 <= t <= y + 10 for x, y in skip)
        ev = [dict(e, t=e['t'] + rule['shift']) for e in DR.apply(ticks, rule)]
        ev = [e for e in ev if inside(e['t'] - rule['shift']) and not out_of(e['t'])]
        truth = [t for t in D.truth(day, True) if inside(t['t'] - L.SHIFT) and not out_of(t['t'])]
        m = D.match(ev, truth, L.TOL)
        r = {k: {'P': round(m[k]['precision'], 3), 'R': round(m[k]['recall'], 3), 'true': m[k]['true'], 'pred': m[k]['pred'] if 'pred' in m[k] else None,
                 'false': m[k]['false'], 'missed': m[k]['true'] - m[k]['hit']} for k in ('in', 'out')}
        out[day] = r
        print('%s %s stretches %d | in P%.2f R%.2f (truth %d, false %d, missed %d) | out P%.2f R%.2f (truth %d, false %d, missed %d)' % (
            day, Path(path).name, len(spans), r['in']['P'], r['in']['R'], r['in']['true'], r['in']['false'], r['in']['missed'],
            r['out']['P'], r['out']['R'], r['out']['true'], r['out']['false'], r['out']['missed']), flush=True)
    name = '_'.join(Path(a.split('=', 1)[1]).name.replace('.jsonl.gz', '') for a in args)
    json.dump(out, open(ROOT / 'data' / 'door_v2' / ('eval_%s_%s.json' % (rule_name, name)), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
