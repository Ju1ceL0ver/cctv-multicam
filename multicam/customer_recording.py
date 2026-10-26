"""Owner's recording policy: confirmed customer entry, 60 s empty grace, 45 min idle.

Replayed from immutable door events and latest roles, so late role decisions and
keeper restarts cannot double-count an entry. Staff and unknown roles never start capture.
"""
from datetime import datetime
from pathlib import Path
import json


def state(events, now):
    count, last, empty_at = 0, None, None
    seen = set()
    for event in sorted(events, key=lambda e: e['t']):
        t = float(event['t'])
        key = (event.get('kind'), round(t,1))
        if t > now or key in seen or event.get('role') != 'customer':
            continue
        seen.add(key)
        if last is not None and t-last >= 2700:
            count = 0; empty_at = None
        before = count
        if event.get('kind') == 'in':
            count += 1; empty_at = None
        elif event.get('kind') == 'out':
            count = max(0,count-1)
            if before and not count:
                empty_at = t
        else:
            continue
        last = t
    stale = last is None or now-last >= 2700
    active = not stale and (count > 0 or empty_at is not None and now-empty_at < 60)
    return {'record': bool(active), 'customers': 0 if stale else count,
            'last_customer_event': last, 'empty_at': empty_at,
            'reason': 'idle_timeout' if stale else 'customer_inside' if count else 'empty_grace' if active else 'empty'}


def load_state(root, now=None):
    from live_events import key_of
    now = now or datetime.now()
    day = now.strftime('%Y%m%d'); root=Path(root)
    updates={}; events=[]
    path=root/'data/live/records'/('roles_%s.jsonl'%day)
    for line in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
        try:
            r=json.loads(line); updates[r['key']]=r
        except (ValueError,KeyError): pass
    path=root/'data/live/events_cam1.jsonl'
    for line in path.read_text(encoding='utf-8').splitlines() if path.exists() else []:
        try:
            e=json.loads(line)
            if e.get('engine')!='doorseg_c+zones' or not e.get('clock','').replace('-','').startswith(day): continue
            events.append(dict(e,role=updates.get(key_of(e),{}).get('role') or e.get('role')))
        except (ValueError,KeyError,TypeError): pass
    return state(events,now.timestamp())
