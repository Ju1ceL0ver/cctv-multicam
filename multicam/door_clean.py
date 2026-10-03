"""The owner's door answers of a day, tidied by one rule, and the clean set of whole visits.

Two things the owner cannot see from one crossing at a time, fixed the same way every day:
  * somebody who goes out and comes back within MERGE_GAP was only stepping into the
    gallery: both answers become 'none' and the visit goes on (18.09: 4-16 s, 19.09: 11-38 s);
  * visits that lack one end are not whole visits and stay out of the clean set, as do
    stretches named in `--skip HH:MM-HH:MM` (a crowd that could not be told apart).

Every changed answer keeps the owner's version under `owner`, with `fixed: "claude"` and why.
The clean set -- customers with an entry and an exit -- goes to data/door_review/<day>_clean.json.

usage: door_clean.py DAY [--skip HH:MM-HH:MM ...] [--dry]"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MERGE_GAP = 60.0


def tidy(day, root=ROOT, skip=(), dry=False, log=print):
    import door_review as d
    from storage import atomic_json, file_lock
    events = {e['event_id']: e for e in d.day_events(day, root)}
    path = d.store_path(day, root)
    changes = []
    with file_lock(str(path) + '.lock'):
        state = d.labels(day, root)
        lab, roles = state['labels'], state.get('roles', {})
        by = {}
        for eid, v in lab.items():
            if v['kind'] in ('in', 'out') and v.get('person') not in (None, 'unseen'):
                by.setdefault(v['person'], []).append(eid)
        for who, ks in by.items():
            ks.sort(key=lambda k: events[k]['unix_ms'])
            for a, b in zip(ks, ks[1:]):
                gap = (events[b]['unix_ms'] - events[a]['unix_ms']) / 1000
                if lab[a]['kind'] == 'out' and lab[b]['kind'] == 'in' and gap <= MERGE_GAP:
                    for eid in (a, b):
                        before = dict(lab[eid])
                        lab[eid] = {'kind': 'none', 'person': None, 'at': datetime.now().isoformat(timespec='seconds'),
                                    'fixed': 'claude', 'why': 'вышел и вернулся за %.0f с — визит продолжается' % gap,
                                    'owner': before}
                        changes.append((events[eid]['time_local'][11:19], before['kind'], 'none', gap))
        if changes and not dry:
            state['revision'] = int(state.get('revision', 0)) + 1
            atomic_json(path, state)
            with open(str(path).replace('.json', '.history.jsonl'), 'a', encoding='utf-8') as f:
                f.write(json.dumps({'at': time.time(), 'fixed_by': 'claude', 'rule': 'merge<=%ds' % MERGE_GAP,
                                    'changes': changes, 'revision': state['revision']}, ensure_ascii=False) + '\n')
    for c in changes:
        log('  %s %s -> %s (%.0f s)' % c)
    visits = d.visits(events, lab, roles)
    windows = [tuple(w.split('-')) for w in skip]
    clean, dropped = [], []
    for v in visits:
        if v['staff']:
            continue
        if not (v['entry'] and v['exit']):
            dropped.append(dict(v, reason='нет входа или выхода')); continue
        hit = [w for w in windows if w[0] <= v['entry_time'][:5] <= w[1] or w[0] <= v['exit_time'][:5] <= w[1]]
        if hit:
            dropped.append(dict(v, reason='исключено: %s–%s' % hit[0])); continue
        clean.append(v)
    out = {'day': day, 'revision': state['revision'], 'made': datetime.now().isoformat(timespec='seconds'),
           'rule': 'покупатели с входом и выходом; выход и возврат за ≤%d с склеены; исключено: %s'
                   % (MERGE_GAP, ', '.join(skip) or 'ничего'),
           'visits': clean, 'dropped': dropped,
           'staff_visits': sum(1 for v in visits if v['staff'])}
    if not dry:
        atomic_json(Path(root) / 'data' / 'door_review' / ('%s_clean.json' % day), out)
    log('%s: %d changes, customer visits %d, clean %d, dropped %d, staff visits %d' % (
        day, len(changes), len(clean) + len(dropped), len(clean), len(dropped), out['staff_visits']))
    return out


if __name__ == '__main__':
    os.chdir(ROOT)
    args = sys.argv[1:]
    skip = [args[k + 1] for k, a in enumerate(args) if a == '--skip']
    tidy(args[0], ROOT, skip=skip, dry='--dry' in args)
