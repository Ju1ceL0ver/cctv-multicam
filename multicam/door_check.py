"""The owner checks where the door model and his /door answers disagree: /doorcheck (03.10.2026).

His /door answers cover only the live counter's events, so a crossing the counter never wrote is not in them; when
the model finds one there it looks false, and nothing tells which it is. This page shows exactly those moments --
the model says somebody went in or out, nothing answered is near -- as a loop of the day film around the moment,
zoomed on the person, and takes one key.

  build DAY RUN.jsonl.gz [MIN_P]   the learned rule's scored events (door_learn.py -> learn_preds_<day>.json), minus
                                   the ones within TOL of an answered crossing of the same direction -> items
  answers -> data/door_check/<day>.json {item: {answer, at}}, every answer also in <day>.history.jsonl
  keys: 1 customer in, 2 customer out, 3 nobody crossed, 4 staff in, 5 staff out, 6 cannot tell"""
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_check'
ANSWERS = {'1': 'in', '2': 'out', '3': 'none', '4': 'staff_in', '5': 'staff_out', '6': 'unsure'}


def build(day, run_path, min_p=0.15):
    import gzip
    import door_learn as L
    import door_v2 as D
    preds = json.load(open(ROOT / 'data' / 'door_v2' / ('learn_preds_%s.json' % day)))
    ticks, spans = D._read(run_path)
    inside = lambda t: any(a <= t <= b for a, b in spans)
    truth = [t for t in D.truth(day, True) if inside(t['t'])]
    near_answer = lambda p: any(abs(t['t'] - p['t']) <= L.TOL and t['kind'] in (p['kind'], 'any') for t in truth)
    at = {}
    for r in ticks:
        for q in r['p']:
            at.setdefault(q['w'], []).append((r['t'], q['box']))
    done = answers(day)                                       # answered items stay as they were through a rebuild
    items = [it for it in globals()['items'](day) if it['id'] in done]
    kept = [it['film'] for it in items]
    for p in sorted(preds, key=lambda p: p['t']):
        if p['p'] < min_p or near_answer(p) or any(abs(p['t'] - L.SHIFT - f) <= 3.0 for f in kept):
            continue
        moment = p['t'] - L.SHIFT                                 # when the model saw it (the film's clock)
        seq = at.get(p['w'], [])
        box = min(seq, key=lambda x: abs(x[0] - moment))[1] if seq else None
        items.append({'id': '%s_%d' % (day, round(moment * 10)), 'day': day, 'film': round(moment, 2), 'kind': p['kind'],
                      'p': p['p'], 'box': box, 'answered_near': [{'kind': t['kind'], 'dt': round(t['t'] - L.SHIFT - moment, 1)}
                                                                 for t in truth if abs(t['t'] - L.SHIFT - moment) <= 20]})
    items.sort(key=lambda it: it['film'])
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({'day': day, 'min_p': min_p, 'items': items}, open(OUT / ('%s_items.json' % day), 'w'), indent=0)
    return items


def items(day):
    p = OUT / ('%s_items.json' % day)
    return json.load(open(p, encoding='utf-8'))['items'] if p.exists() else []


def answers(day):
    p = OUT / ('%s.json' % day)
    return json.load(open(p, encoding='utf-8')) if p.exists() else {}


def answer(day, item, key):
    from storage import atomic_json, file_lock
    if key is not None and key not in ANSWERS.values():
        raise ValueError('unknown answer')
    if item not in {i['id'] for i in items(day)}:
        raise ValueError('unknown item')
    OUT.mkdir(parents=True, exist_ok=True)
    with file_lock(str(OUT / ('%s.json' % day)) + '.lock'):
        a = answers(day)
        if key is None:
            a.pop(item, None)
        else:
            a[item] = {'answer': key, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        atomic_json(OUT / ('%s.json' % day), a)
        with open(OUT / ('%s.history.jsonl' % day), 'a', encoding='utf-8') as f:
            f.write(json.dumps({'item': item, 'answer': key, 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}, ensure_ascii=False) + '\n')
    return {'ok': True, 'answers': len(a)}


def days():
    return sorted(p.name[:8] for p in OUT.glob('*_items.json')) if OUT.exists() else []


def register(app, root=None):
    from flask import abort, jsonify, render_template, request

    def valid(day):
        if not re.fullmatch(r'\d{8}', day):
            abort(400)

    @app.get('/doorcheck')
    def doorcheck_page():
        return render_template('doorcheck.html')

    @app.get('/api/doorcheck')
    def doorcheck_days():
        return jsonify({'days': [{'day': d, 'items': len(items(d)), 'answered': len(answers(d))} for d in days()]})

    @app.get('/api/doorcheck/<day>')
    def doorcheck_day(day):
        valid(day)
        return jsonify({'day': day, 'items': items(day), 'answers': answers(day)})

    @app.post('/api/doorcheck/<day>')
    def doorcheck_answer(day):
        valid(day)
        body = request.get_json(force=True) or {}
        try:
            return jsonify(answer(day, str(body.get('item', '')), body.get('answer')))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    if sys.argv[1] == 'build':
        its = build(sys.argv[2], sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 0.15)
        print(len(its), 'items')
