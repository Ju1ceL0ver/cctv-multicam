"""21.10 owner: the door count as the CRM gets it now (every raw event) against the count by visit
(identity_<day>.json, field `count`), side by side before the CRM is switched.

With the owner's /visitpairs answers (data/visits/<day>/people_answers.json) each disagreement gets a verdict:
an event he marked X should not count; a real one should, unless it repeats the same person's previous event of
the same kind (the door counting one crossing twice).
-> data/door_identity/<day>/compare.json   usage: door_count_compare.py DAY"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def compare(day, root=ROOT):
    from storage import atomic_json, read_json
    ident = {x['key']: x for x in read_json(root / 'data' / 'live' / 'records' / ('identity_%s.json' % day), [])}
    answers = read_json(root / 'data' / 'visits' / day / 'people_answers.json', {})
    staff = set(answers.get('_staff', []))
    events = []
    for line in open(root / 'data' / 'live' / 'events_cam1.jsonl', encoding='utf-8'):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if e.get('engine') == 'doorseg_c+zones' and e['clock'].replace('-', '').startswith(day):
            events.append(e)
    events.sort(key=lambda e: e['t'])
    last_kind = {}                        # owner's person -> kind of their previous real event
    rows, score = [], {'raw': [0, 0], 'visit': [0, 0], 'labelled': 0}   # [right, wrong]
    for e in events:
        key = '%s_%s_%.1f' % (e['kind'], e['clock'][11:19].replace(':', ''), float(e['t']))
        x = ident.get(key)
        visit = x.get('count') if x else None
        a = answers.get('%.1f' % e['t'])
        truth = None
        if a == 'x':
            truth = False
        elif a is not None:
            truth = last_kind.get(a) != e['kind']      # a second in (or out) in a row is the same crossing again
            last_kind[a] = e['kind']
        row = dict(key=key, clock=e['clock'][11:], kind=e['kind'], raw=True, visit=visit, status=x and x.get('status'),
                   owner=a, staff=a in staff if a not in (None, 'x') else None, truth=truth)
        if truth is not None:
            score['labelled'] += 1
            score['raw'][0 if truth else 1] += 1
            if visit is not None:
                score['visit'][0 if visit == truth else 1] += 1
        rows.append(row)
    diff = [r for r in rows if r['visit'] is not True]
    out = dict(day=day, events=len(rows), raw_count={k: sum(r['kind'] == k for r in rows) for k in ('in', 'out')},
               visit_count={k: sum(r['kind'] == k and r['visit'] is True for r in rows) for k in ('in', 'out')},
               undecided=sum(r['visit'] is None for r in rows), score=score, differences=diff)
    (root / 'data' / 'door_identity' / day).mkdir(parents=True, exist_ok=True)
    atomic_json(root / 'data' / 'door_identity' / day / 'compare.json', out)
    return out


if __name__ == '__main__':
    r = compare(sys.argv[1])
    print(json.dumps({k: v for k, v in r.items() if k != 'differences'}, ensure_ascii=False))
