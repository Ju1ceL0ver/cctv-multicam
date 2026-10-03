"""The door, labelled by the owner: which crossings of the live counter were real.

The live counter (retail_analytics) writes one row per crossing of its door line. Seen
on 18.09, most rows are not crossings: the sample stand stands on the line and people
choosing laminate swing back and forth over it, a saleswoman is called a customer on
most of her entries. Nobody knows today how many visits a day there really were.

Here the owner answers, per row, what really happened -- a customer came in, a customer
went out, nobody crossed, a member of staff crossed, the same crossing again -- and for
a customer going out, which of the customers he saw come in it was. That is the first
ground truth of whole visits: entry and exit of the same person.

Stored in data/door_review/<day>.json, one answer per event, with a revision; every
change is also appended to data/door_review/<day>.history.jsonl."""
import json
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path

from storage import atomic_json, file_lock, read_json

KINDS = ('in', 'out', 'none', 'staff', 'dup', 'unsure')   # 'staff' alone: the old answer, no direction, no person
ROLES = ('staff', 'customer')
_cache = {}
_lock = threading.Lock()


def live_dir(root):
    own = os.environ.get('RA_LIVE_DIR')
    if own:
        return Path(own)
    return Path(root).resolve().parent / 'retail_analytics' / 'runs' / 'live'


def _all_events(root):
    path = live_dir(root) / 'entrance_events.jsonl'
    stamp = path.stat().st_mtime_ns
    with _lock:
        hit = _cache.get(str(path))
        if hit and hit[0] == stamp:
            return hit[1]
    rows = []
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue            # the live service may be halfway through a line
    with _lock:
        _cache[str(path)] = (stamp, rows)
    return rows


def day_events(day, root):
    iso = '%s-%s-%s' % (day[:4], day[4:6], day[6:])
    rows = [e for e in _all_events(root) if str(e.get('time_local', '')).startswith(iso)]
    rows.sort(key=lambda e: e['time_local'])
    return rows


def days(root):
    seen = sorted({str(e.get('time_local', ''))[:10].replace('-', '') for e in _all_events(root)})
    return [d for d in seen if re.fullmatch(r'\d{8}', d)]


def store_path(day, root):
    return Path(root) / 'data' / 'door_review' / ('%s.json' % day)


def labels(day, root):
    return read_json(store_path(day, root), None) or {'day': day, 'revision': 0, 'labels': {}}


def film_start(day, root):
    """Start of the day's film on the unix clock, or None when the film is not made."""
    side = Path(root) / 'data' / 'day_movie' / day / 'cam1.json'
    mp4 = side.with_suffix('.mp4')
    record = read_json(side, None)
    if not record or not mp4.exists():
        return None
    return float(record['start'])


def overview(day, root):
    start = film_start(day, root)
    rows = []
    for e in day_events(day, root):
        unix = e['unix_ms'] / 1000.0
        rows.append({'id': e['event_id'], 'time': e['time_local'][11:19], 'unix': unix,
                     'event': e['event'], 'role': e.get('role'), 'gid': e.get('global_id'),
                     'box': [e.get('box_x1'), e.get('box_y1'), e.get('box_x2'), e.get('box_y2')],
                     'film': None if start is None else round(unix - start, 2)})
    return {'day': day, 'events': rows, 'film': start is not None, 'review': labels(day, root)}


def _when(e):
    return e['unix_ms'] / 1000.0


def visits(events, lab, roles=None):
    """Visits from the answers, by person, the same rule every time.

    A person is the entry that first brought them in (`person` of an entry is its own id,
    or the id of that first entry when they came back). Walking through their answers in
    time: an entry opens a visit, a second entry while one is open changes nothing (the
    first one stands), and an exit ends the visit -- the latest exit wins, so a person
    marked going out twice is out at the later one. An exit of somebody whose entry was not
    seen is a visit without a start."""
    by, staff = {}, set()
    for eid, v in lab.items():
        e = events.get(eid)
        if e is None or v.get('kind') not in ('in', 'out'):
            continue
        who = v.get('person') or eid
        if who == 'unseen' or who not in lab:
            who = 'unseen:' + eid          # nobody to attach it to: a visit of its own
        if v.get('role') == 'staff' or (roles or {}).get(who) == 'staff':
            staff.add(who)
        by.setdefault(who, []).append((_when(e), v['kind'], eid))
    out = []
    for who, rows in by.items():
        rows.sort()
        cur = None
        for t, kind, eid in rows:
            if kind == 'in':
                if cur is not None and cur['exit'] is None:
                    continue                 # already inside: the first entry stands
                cur = {'person': who, 'entry': eid, 'exit': None}
                out.append(cur)
            else:
                if cur is None:
                    cur = {'person': who, 'entry': None, 'exit': eid}
                    out.append(cur)
                else:
                    cur['exit'] = eid        # the later exit wins
    for v in out:
        a = events.get(v['entry']) if v['entry'] else None
        b = events.get(v['exit']) if v['exit'] else None
        v['entry_time'] = a['time_local'][11:19] if a else None
        v['exit_time'] = b['time_local'][11:19] if b else None
        v['seconds'] = round(_when(b) - _when(a), 1) if a and b else None
        v['staff'] = v['person'] in staff
    out.sort(key=lambda v: v['entry_time'] or v['exit_time'])
    return out


def answer(day, root, body):
    """One answer about one event. `kind` None removes the answer (undo).

    `person` says who: for an entry, its own id (somebody new) or the first entry of a
    person who has come back; for an exit, the first entry of the person going out, or
    'unseen' when their entry is not among the answers."""
    event = str(body.get('event', ''))
    kind = body.get('kind')
    person = body.get('person')
    role = body.get('role')
    if role is not None and (role not in ROLES or kind not in ('in', 'out')):
        raise ValueError('Роль указывается только у входа и выхода: staff или customer')
    known = {e['event_id']: e for e in day_events(day, root)}
    if event not in known:
        raise ValueError('Нет такого события за этот день')
    if kind is not None and kind not in KINDS:
        raise ValueError('Неизвестный ответ')
    if person is not None and kind not in ('in', 'out'):
        raise ValueError('Человек указывается только у входа и выхода')
    path = store_path(day, root)
    with file_lock(str(path) + '.lock'):
        state = labels(day, root)
        lab = state['labels']
        if kind == 'in' and person in (None, event):
            person = event
        elif kind in ('in', 'out') and person != 'unseen' and person is not None:
            first = lab.get(person)
            if person not in known or not first or first.get('kind') != 'in' or first.get('person', person) != person:
                raise ValueError('Этот человек не отмечен входящим')
            if _when(known[person]) >= _when(known[event]):
                raise ValueError('Человек не может выйти или вернуться раньше, чем вошёл')
        elif kind == 'out':
            person = person or 'unseen'
        before = lab.get(event)
        if kind is None:
            lab.pop(event, None)
        else:
            lab[event] = {'kind': kind, 'person': person, 'at': datetime.now().isoformat(timespec='seconds')}
            if role is not None:
                if person == 'unseen':
                    lab[event]['role'] = role          # nobody to hold it: the exit itself says who left
                else:
                    state.setdefault('roles', {})[person] = role
        state['revision'] = int(state.get('revision', 0)) + 1
        atomic_json(path, state)
        with open(str(path).replace('.json', '.history.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': time.time(), 'event': event, 'before': before,
                                'after': lab.get(event), 'revision': state['revision']},
                               ensure_ascii=False) + '\n')
    return {'revision': state['revision'], 'label': lab.get(event)}


def set_role(day, root, person, role):
    """Say once that a person is staff (or a customer after all); every visit of theirs follows."""
    if role not in ROLES:
        raise ValueError('Роль: staff или customer')
    path = store_path(day, root)
    with file_lock(str(path) + '.lock'):
        state = labels(day, root)
        first = state['labels'].get(person)
        if not first or first.get('kind') != 'in' or (first.get('person') or person) != person:
            raise ValueError('Этот человек не отмечен входящим')
        before = state.setdefault('roles', {}).get(person)
        state['roles'][person] = role
        state['revision'] = int(state.get('revision', 0)) + 1
        atomic_json(path, state)
        with open(str(path).replace('.json', '.history.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': time.time(), 'person': person, 'role_before': before, 'role': role,
                                'revision': state['revision']}, ensure_ascii=False) + '\n')
    return {'revision': state['revision'], 'roles': state['roles']}


def clear(day, root):
    """Start the day over: the answers go to the history, the store is emptied."""
    path = store_path(day, root)
    with file_lock(str(path) + '.lock'):
        state = labels(day, root)
        old, old_roles = state.get('labels', {}), state.get('roles', {})
        state = {'day': day, 'revision': int(state.get('revision', 0)) + 1, 'labels': {}}
        atomic_json(path, state)
        with open(str(path).replace('.json', '.history.jsonl'), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': time.time(), 'cleared': old, 'cleared_roles': old_roles, 'revision': state['revision']},
                               ensure_ascii=False) + '\n')
    return state


def summary(day, root):
    """What the answers say about the day, next to what the counter said."""
    ev = {e['event_id']: e for e in day_events(day, root)}
    state = labels(day, root)
    lab = state['labels']
    kinds = {}
    for v in lab.values():
        kinds[v['kind']] = kinds.get(v['kind'], 0) + 1
    counter_in = sum(1 for e in ev.values() if e['event'] == 'entry' and e.get('role') == 'customer')
    V = visits(ev, lab, state.get('roles'))
    cust = [v for v in V if not v['staff']]
    return {'events': len(ev), 'answered': len(lab), 'kinds': kinds,
            'counter_customer_entries': counter_in, 'visits': len(cust),
            'visits_whole': sum(1 for v in cust if v['entry'] and v['exit']),
            'staff_visits': len(V) - len(cust), 'list': V}


def snapshot(day, root, event, what):
    """Path of the live counter's own picture of the crossing, only from its snapshot folder."""
    for e in day_events(day, root):
        if e['event_id'] == event:
            path = Path(e['snapshot_crop' if what == 'crop' else 'snapshot_full'])
            base = (live_dir(root) / 'snapshots').resolve()
            try:
                path.resolve().relative_to(base)
            except ValueError:
                return None, e
            return (path if path.exists() else None), e
    return None, None


def zoom(path, box, width=900):
    """The full picture cut around the person: the counter draws every track on it."""
    import cv2
    import numpy as np
    im = cv2.imread(str(path))
    if im is None:
        return None
    H, W = im.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = max(x2 - x1, 40), max(y2 - y1, 80)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = max(2.2 * bw, 0.8 * bh * 16 / 9), max(0.9 * bh, 1.1 * bw * 9 / 16)
    a, b = int(max(0, cx - hw)), int(min(W, cx + hw))
    c, d = int(max(0, cy - hh)), int(min(H, cy + hh))
    cut = im[c:d, a:b]
    if cut.size == 0:
        cut = im
    if cut.shape[1] > width:
        cut = cv2.resize(cut, (width, int(cut.shape[0] * width / cut.shape[1])), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode('.jpg', cut, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return bytes(buf) if ok else None


def register(app, root):
    from flask import Response, abort, jsonify, render_template, request, send_file
    here = root if callable(root) else (lambda: root)

    def valid_day(day):
        if not re.fullmatch(r'\d{8}', day):
            abort(400)

    @app.get('/door')
    def door_page():
        return render_template('door.html')

    @app.get('/api/door')
    def door_days():
        return jsonify({'days': days(here())})

    @app.get('/api/door/<day>')
    def door_day(day):
        valid_day(day)
        return jsonify(overview(day, here()))

    @app.get('/api/door/<day>/summary')
    def door_summary(day):
        valid_day(day)
        return jsonify(summary(day, here()))

    @app.post('/api/door/<day>')
    def door_answer(day):
        valid_day(day)
        try:
            return jsonify(answer(day, here(), request.get_json(force=True) or {}))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503

    @app.post('/api/door/<day>/role')
    def door_role(day):
        valid_day(day)
        body = request.get_json(force=True) or {}
        try:
            return jsonify(set_role(day, here(), str(body.get('person', '')), body.get('role')))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        except TimeoutError as exc:
            return jsonify({'error': str(exc)}), 503

    @app.get('/api/door/<day>/snap/<event>/<what>')
    def door_snap(day, event, what):
        valid_day(day)
        if what not in ('crop', 'full', 'zoom') or not re.fullmatch(r'[0-9a-f]{8,64}', event):
            abort(404)
        path, e = snapshot(day, here(), event, what)
        if path is None:
            abort(404)
        if what == 'zoom':
            body = zoom(path, [e['box_x1'], e['box_y1'], e['box_x2'], e['box_y2']])
            if body is None:
                abort(404)
            r = Response(body, mimetype='image/jpeg')
        else:
            r = send_file(path, mimetype='image/jpeg', conditional=True)
        r.headers['Cache-Control'] = 'private, max-age=86400'
        return r
