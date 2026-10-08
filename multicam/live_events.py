"""/liveevents -- the owner checks the live door's events (08.10.2026).

Every entry/exit of data/live/events_cam1.jsonl of a day with its clip (data/live/clips/<day>/, door_clips.py: +-10 s
around it, nothing drawn on it) -- one key per event:
  1 right   2 nobody crossed (false)   3 wrong direction (it was the other way)   4 wrong role   0 can't tell
Answers go to data/live/review/<day>.json (+ .history.jsonl), never into the events; the day's summary counts them.
Missed crossings are found on the random-minute and dispute clips (listed below the events)."""
import json
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / 'data' / 'live'
REVIEW = LIVE / 'review'
VERDICTS = {1: 'верно', 2: 'никто не прошёл', 3: 'наоборот (вход/выход)', 4: 'роль неверна', 0: 'не понять'}


def key_of(e):
    return '%s_%s_%.1f' % (e['kind'], e.get('clock', '')[11:19].replace(':', ''), float(e['t']))


def events(day):
    p = LIVE / 'events_cam1.jsonl'
    if not p.exists():
        return []
    want = '%s-%s-%s' % (day[:4], day[4:6], day[6:])
    out = []
    for line in p.read_text(encoding='utf-8').splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if str(e.get('clock', '')).startswith(want):
            out.append(e)
    return sorted(out, key=lambda e: e['t'])


def clips(day):
    out = []
    for j in sorted((LIVE / 'clips' / day).glob('*.json')):
        try:
            c = json.load(open(j))
        except ValueError:
            continue
        if (j.with_suffix('.mp4')).exists():
            out.append({'name': j.stem, 't0': c['t0'], 't1': c['t1'], 'why': c.get('why', [])})
    return out


def load(day):
    p = REVIEW / ('%s.json' % day)
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def answer(day, key, verdict, note=None):
    from storage import atomic_json
    if verdict is not None and verdict not in VERDICTS:
        raise ValueError('verdict')
    REVIEW.mkdir(parents=True, exist_ok=True)
    st = load(day)
    if verdict is None:
        st.pop(key, None)
    else:
        st[key] = {'v': verdict, 'note': note, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
    atomic_json(REVIEW / ('%s.json' % day), st)
    with open(REVIEW / ('%s.history.jsonl' % day), 'a', encoding='utf-8') as f:
        f.write(json.dumps({'key': key, 'v': verdict, 'note': note, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}, ensure_ascii=False) + '\n')
    return st


def summary(day, evs, st):
    s = {'in': sum(e['kind'] == 'in' for e in evs), 'out': sum(e['kind'] == 'out' for e in evs),
         'staff': sum(e.get('role') == 'staff' for e in evs), 'checked': 0, 'right': 0}
    for e in evs:
        v = st.get(key_of(e), {}).get('v')
        if v is not None and v != 0:
            s['checked'] += 1
            s['right'] += v == 1
    return s


def register(app, root=None):
    from flask import abort, jsonify, render_template, request, send_file

    def valid(day):
        if not re.fullmatch(r'\d{8}', day):
            abort(400)

    @app.get('/liveevents')
    def liveevents_page():
        return render_template('liveevents.html', verdicts=VERDICTS)

    @app.get('/api/live/events/<day>')
    def liveevents_day(day):
        valid(day)
        evs, st, cs = events(day), load(day), clips(day)
        nf = LIVE / 'night' / ('%s.json' % day)            # the night teacher (door_night.py)
        night = json.load(open(nf)) if nf.exists() else None
        by_key = {r['live_key']: r['by'] for r in (night or {}).get('events', []) if r.get('live_key')}
        out = []
        for e in evs:
            c = next((c for c in cs if c['t0'] - 0.5 <= e['t'] <= c['t1'] + 0.5), None)
            out.append({'key': key_of(e), 'kind': e['kind'], 'clock': e.get('clock', '')[11:19], 'role': e.get('role'),
                        'p_staff': e.get('p_staff'), 'tracks': e.get('tracks'), 'verdict': st.get(key_of(e)),
                        'clip': c['name'] if c else None, 'offset': round(e['t'] - c['t0'], 2) if c else None,
                        'teacher': by_key.get(key_of(e)) if night else None})
        others = [c for c in cs if 'event' not in c['why']]
        days = sorted({p.name for p in (LIVE / 'clips').glob('20??????')}, reverse=True) if (LIVE / 'clips').exists() else []
        missed = []
        for r in (night or {}).get('events', []):
            if r['by'] == 'teacher':
                c = next((c for c in cs if c['name'] == r.get('clip')), None)
                missed.append({'kind': r['kind'], 'clock': r.get('clock'), 'clip': r.get('clip'),
                               'offset': round(r['t'] - c['t0'], 2) if c else None})
        return jsonify({'day': day, 'events': out, 'summary': summary(day, evs, st), 'other_clips': others, 'days': days,
                        'missed': missed, 'night': (night or {}).get('summary')})

    @app.get('/api/live/clip/<day>/<name>')
    def liveevents_clip(day, name):
        valid(day)
        if not re.fullmatch(r'\d{6}_[a-z]+', name):
            abort(400)
        p = LIVE / 'clips' / day / (name + '.mp4')
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype='video/mp4', conditional=True)

    @app.post('/api/live/events/<day>')
    def liveevents_answer(day):
        valid(day)
        b = request.get_json(force=True) or {}
        try:
            v = b.get('verdict')
            st = answer(day, str(b['key']), None if v is None else int(v), b.get('note'))
        except (KeyError, ValueError) as exc:
            return jsonify({'error': str(exc)}), 400
        return jsonify({'ok': True, 'summary': summary(day, events(day), st)})
