"""Visits from the live counter's door crossings: who came in, when, and when they left.

The live service (retail_analytics) writes one row per line crossing. Measured on
18.09: 210 crossings, most of them noise -- people choosing samples at the stand
that sits on the door line swing back and forth across it (one family: ~55
crossings in 11 minutes), a saleswoman crossed 22 times and was called a customer
on most entries, and only 25 of 96 customer entries ever got an exit with the same
id. The count of crossings is not the count of visits.

This module turns crossings into visits without the tracker's ids:
  1. every crossing gets an appearance vector (OSNet-AIN on its snapshot crop);
  2. crossings are assigned online to people of the day by appearance -- the id of
     the live tracker is only a hint, it breaks at the door;
  3. a person's crossings are collapsed: a stretch inside begins with the first IN
     after being out for longer than OUTSIDE_GAP and ends with the last OUT that is
     not followed by an IN within OUTSIDE_GAP. Swinging at the stand disappears;
  4. whoever makes STAFF_VISITS separate visits a day, or whom the classifier calls
     staff on most crossings, is staff.

Pure functions over event dicts; `python door_visits.py EVENTS.jsonl EMB.npy DAY`
prints the day."""
import json
import sys
from datetime import datetime

import numpy as np

MATCH = 0.30          # cosine distance: below this a crossing belongs to a known person
TOPK = 3              # distance to a person = mean of the k closest crossings of theirs
OUTSIDE_GAP = 180.0   # s out of the shop before a new IN starts a new visit
SAME_MOMENT = 1.0     # s: two crossings this close by different tracks are two people
STAFF_VISITS = 3
STAFF_SHARE = 0.5
MIN_VISIT = 20.0      # s: shorter stretches inside are the door, not a visit


def when(e):
    return datetime.fromisoformat(e['time_local']).timestamp()


def assign_people(events, emb, match=MATCH):
    """Online: each crossing joins the closest person of the day or starts one."""
    people = []            # dict(idx=[...], vecs=[...])
    owner = []
    for i, e in enumerate(events):
        v = emb[i]
        t = when(e)
        best, bd = None, match
        for p, pr in enumerate(people):
            # the same person cannot cross twice at the same moment under two track ids
            if any(abs(when(events[j]) - t) < SAME_MOMENT and events[j]['track_id'] != e['track_id']
                   for j in pr['idx'][-4:]):
                continue
            d = np.sort(1 - np.asarray(pr['vecs']) @ v)[:TOPK].mean()
            if d < bd:
                best, bd = p, d
        if best is None:
            people.append({'idx': [], 'vecs': []})
            best = len(people) - 1
        people[best]['idx'].append(i)
        people[best]['vecs'].append(v)
        owner.append(best)
    return people, owner


def stretches(events, idx, gap=OUTSIDE_GAP):
    """Collapse one person's crossings into stretches inside: (entry_i, exit_i).

    Either end may be None: an exit without a seen entry (came in before the
    counter looked) or an entry still open."""
    out = []
    cur_in, last_out = None, None
    for i in idx:
        e = events[i]
        if e['event'] == 'entry':
            if cur_in is None:
                if last_out is not None and when(e) - when(events[last_out]) < gap and out:
                    # stepped out and back within the gap: the previous visit goes on
                    cur_in = out.pop()[0]
                else:
                    cur_in = i
            last_out = None
        else:
            if cur_in is None and last_out is None and not out:
                out.append((None, i))           # already inside when first seen
                last_out = i
                continue
            if cur_in is None and last_out is not None:
                # OUT after OUT: the later one is the real exit
                ent = out.pop()[0] if out else None
                out.append((ent, i)); last_out = i
                continue
            if cur_in is None:
                out.append((None, i)); last_out = i
                continue
            out.append((cur_in, i)); cur_in = None; last_out = i
    if cur_in is not None:
        out.append((cur_in, None))
    return out


def visits(events, emb, match=MATCH, gap=OUTSIDE_GAP):
    events = list(events)
    order = sorted(range(len(events)), key=lambda i: when(events[i]))
    events = [events[i] for i in order]
    emb = np.asarray(emb)[order]
    people, _ = assign_people(events, emb, match)
    out = []
    for p, pr in enumerate(people):
        st = stretches(events, pr['idx'], gap)
        roles = [events[i]['role'] for i in pr['idx']]
        staff_share = sum(r == 'staff' for r in roles) / len(roles)
        staff = len(st) >= STAFF_VISITS or staff_share >= STAFF_SHARE
        for a, b in st:
            ta = when(events[a]) if a is not None else None
            tb = when(events[b]) if b is not None else None
            if ta is not None and tb is not None and tb - ta < MIN_VISIT:
                continue
            out.append({'person': p, 'staff': bool(staff),
                        'entry': events[a]['time_local'] if a is not None else None,
                        'exit': events[b]['time_local'] if b is not None else None,
                        'entry_event': events[a]['event_id'] if a is not None else None,
                        'exit_event': events[b]['event_id'] if b is not None else None,
                        'seconds': None if ta is None or tb is None else round(tb - ta, 1),
                        'crossings': len(pr['idx'])})
    out.sort(key=lambda v: v['entry'] or v['exit'])
    return out, people, events


if __name__ == '__main__':
    evs = [json.loads(l) for l in open(sys.argv[1], encoding='utf-8')]
    emb = np.load(sys.argv[2])
    day = sys.argv[3]
    keep = [i for i, e in enumerate(evs) if e['time_local'].startswith(day)]
    V, people, _ = visits([evs[i] for i in keep], emb[keep])
    cust = [v for v in V if not v['staff']]
    print('%d crossings -> %d people, %d visits (%d customer, %d staff), customer with both ends %d'
          % (len(keep), len(people), len(V), len(cust), len(V) - len(cust),
             sum(1 for v in cust if v['entry'] and v['exit'])))
    for v in V:
        print('P%-3d %-5s %s -> %s  %s s  (%d crossings)' % (
            v['person'], 'staff' if v['staff'] else '', (v['entry'] or '-')[11:19], (v['exit'] or '-')[11:19],
            v['seconds'], v['crossings']))
