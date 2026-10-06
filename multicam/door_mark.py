"""The owner marks every crossing of the door stretches himself, on the video: /doormark (06.10.2026).

His /door answers cover only the live counter's events, so a crossing the counter never wrote is in no truth, and a
model finding it looks false. Here each door stretch (data/sam31_door, camera 1, +-25 s around the counter's events)
plays with frame stepping; a key at the right frame marks an entry or an exit (customer or staff). His earlier
answers are drawn on the stretch's timeline so as not to mark them twice; a stretch marked 'done' is complete truth.

marks -> data/door_mark/<day>.json {'marks': [{id, tag, film, kind, staff, at}], 'done': [tag, ...]}, every change
also in <day>.history.jsonl. door_v2.truth() adds the marks (film + door_learn.SHIFT, the counter's clock) unless an
answer of the same direction is within 2 s.

The page's video is a light copy of the stretch (1280 x 720, made on first request, kept in data/door_mark/video)."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_mark'
DOOR = ROOT / 'data' / 'sam31_door'
KINDS = ('in', 'out', 'staff_in', 'staff_out')


def load(day):
    p = OUT / ('%s.json' % day)
    return json.load(open(p, encoding='utf-8')) if p.exists() else {'marks': [], 'done': []}


def save(day, state, change):
    from storage import atomic_json
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / ('%s.json' % day), state)
    with open(OUT / ('%s.history.jsonl' % day), 'a', encoding='utf-8') as f:
        f.write(json.dumps(dict(change, at=time.strftime('%Y-%m-%dT%H:%M:%S')), ensure_ascii=False) + '\n')


def stretches(day):
    """[{tag, a, seconds, answers: [{t (s from the stretch start), kind, staff}]}] of the day's door stretches."""
    import door_learn as L
    import door_sam as DS
    import door_v2 as D
    truth = D.truth(day, True)
    out = []
    for a, b in D.stretches(day):
        tag = DS.tag_of(day, a)
        if not (DOOR / tag / 'cam1' / 'video.mp4').exists():
            continue
        ans = [{'t': round(t['t'] - L.SHIFT - a, 2), 'kind': t['kind'], 'staff': t.get('staff', False)}
               for t in truth if a - 2 <= t['t'] - L.SHIFT <= b + 2 and not str(t.get('id', '')).startswith('mark:')]
        out.append({'tag': tag, 'a': a, 'seconds': round(b - a, 1), 'answers': ans})
    return out


def video(tag):
    """The light copy of a stretch's video, made once."""
    import day_proxy
    src = DOOR / tag / 'cam1' / 'video.mp4'
    dst = OUT / 'video' / (tag + '.mp4')
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix('.tmp.mp4')
        subprocess.run([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-i', str(src), '-vf', 'scale=1280:720', '-c:v', 'libx264',
                        '-preset', 'veryfast', '-crf', '25', '-g', '6', '-bf', '0', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                        str(tmp)], check=True)
        tmp.replace(dst)
    return dst


def mark(day, body):
    from storage import file_lock
    OUT.mkdir(parents=True, exist_ok=True)
    with file_lock(str(OUT / ('%s.json' % day)) + '.lock'):
        st = load(day)
        op = body.get('op')
        if op == 'add':
            kind = body['kind']
            if kind not in KINDS:
                raise ValueError('kind')
            m = {'id': '%s_%d' % (body['tag'], round(float(body['film']) * 100)), 'tag': body['tag'], 'film': round(float(body['film']), 2),
                 'kind': 'in' if kind.endswith('in') else 'out', 'staff': kind.startswith('staff')}
            st['marks'] = [x for x in st['marks'] if x['id'] != m['id']] + [m]
            save(day, st, {'op': 'add', **m})
        elif op == 'remove':
            st['marks'] = [x for x in st['marks'] if x['id'] != body['id']]
            save(day, st, {'op': 'remove', 'id': body['id']})
        elif op == 'done':
            if body['tag'] not in st['done']:
                st['done'].append(body['tag'])
            save(day, st, {'op': 'done', 'tag': body['tag']})
        elif op == 'undone':
            st['done'] = [t for t in st['done'] if t != body['tag']]
            save(day, st, {'op': 'undone', 'tag': body['tag']})
        else:
            raise ValueError('op')
    return {'ok': True, 'marks': len(st['marks']), 'done': len(st['done'])}


def truth_marks(day):
    """The marks as door_v2.truth entries (counter's clock)."""
    import door_learn
    return [{'t': m['film'] + door_learn.SHIFT, 'kind': m['kind'], 'staff': m['staff'], 'id': 'mark:' + m['id']} for m in load(day)['marks']]


def register(app, root=None):
    from flask import abort, jsonify, render_template, request, send_file

    def valid(day):
        if not re.fullmatch(r'\d{8}', day):
            abort(400)

    @app.get('/doormark')
    def doormark_page():
        return render_template('doormark.html')

    @app.get('/api/doormark/<day>')
    def doormark_day(day):
        valid(day)
        import day_movie
        start, _ = day_movie.clock(day, str(ROOT))
        return jsonify({'day': day, 'film_start': start, 'stretches': stretches(day), 'state': load(day)})

    @app.get('/api/doormark/video/<tag>')
    def doormark_video(tag):
        if not re.fullmatch(r'door_\d{8}_\d{5}', tag):
            abort(400)
        return send_file(str(video(tag)), mimetype='video/mp4', conditional=True)

    @app.post('/api/doormark/<day>')
    def doormark_mark(day):
        valid(day)
        try:
            return jsonify(mark(day, request.get_json(force=True) or {}))
        except (ValueError, KeyError) as exc:
            return jsonify({'error': str(exc)}), 400
