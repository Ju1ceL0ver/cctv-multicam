import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))


def test_live_events_page_and_answers(tmp_path, monkeypatch):
    import live_events as LE
    from flask import Flask
    live = tmp_path / 'live'
    (live / 'clips' / '20261008').mkdir(parents=True)
    monkeypatch.setattr(LE, 'LIVE', live)
    monkeypatch.setattr(LE, 'REVIEW', live / 'review')
    evs = [{'kind': 'in', 't': 1000.0, 'clock': '2026-10-08 10:05:00', 'role': 'customer', 'p_staff': 0.1, 'tracks': ['w:1']},
           {'kind': 'out', 't': 1100.0, 'clock': '2026-10-08 10:06:40', 'role': 'staff', 'p_staff': 0.8},
           {'kind': 'in', 't': 5.0, 'clock': '2026-10-07 09:00:00'}]
    (live / 'events_cam1.jsonl').write_text('\n'.join(json.dumps(e) for e in evs) + '\n', encoding='utf-8')
    (live / 'clips' / '20261008' / '100450_event.json').write_text(json.dumps({'t0': 990.0, 't1': 1010.0, 'why': ['event']}))
    (live / 'clips' / '20261008' / '100450_event.mp4').write_bytes(b'x')
    (live / 'clips' / '20261008' / '110000_random.json').write_text(json.dumps({'t0': 3000.0, 't1': 3060.0, 'why': ['random']}))
    (live / 'clips' / '20261008' / '110000_random.mp4').write_bytes(b'x')
    app = Flask(__name__, template_folder=str(Path(LE.__file__).parent / 'templates'))
    LE.register(app)
    c = app.test_client()
    assert c.get('/liveevents').status_code == 200
    j = c.get('/api/live/events/20261008').get_json()
    assert [e['kind'] for e in j['events']] == ['in', 'out']                 # only that day
    assert j['events'][0]['clip'] == '100450_event' and j['events'][0]['offset'] == 10.0
    assert j['events'][1]['clip'] is None
    assert [o['name'] for o in j['other_clips']] == ['110000_random']
    r = c.post('/api/live/events/20261008', json={'key': j['events'][0]['key'], 'verdict': 1}).get_json()
    assert r['summary']['checked'] == 1 and r['summary']['right'] == 1
    r = c.post('/api/live/events/20261008', json={'key': j['events'][1]['key'], 'verdict': 2}).get_json()
    assert r['summary']['checked'] == 2 and r['summary']['right'] == 1
    assert c.post('/api/live/events/20261008', json={'key': 'x', 'verdict': 9}).status_code == 400
    assert c.get('/api/live/clip/20261008/100450_event').status_code == 200
    assert c.get('/api/live/clip/20261008/..%2Fx').status_code in (400, 404)
    assert (live / 'review' / '20261008.history.jsonl').exists()


def test_roles_decided_again_show_the_last_answer(tmp_path, monkeypatch):
    """09.10: door_live writes roles_<day>.jsonl when the grown staff gallery changes an earlier event's role"""
    import live_events as LE
    live = tmp_path / 'live'
    (live / 'records').mkdir(parents=True)
    monkeypatch.setattr(LE, 'LIVE', live)
    evs = [{'kind': 'in', 't': 1000.0, 'clock': '2026-10-08 10:05:00', 'role': 'customer', 'p_staff': 0.1, 'tracks': ['w:1']},
           {'kind': 'out', 't': 1100.0, 'clock': '2026-10-08 10:06:40', 'role': 'customer', 'p_staff': 0.2}]
    (live / 'events_cam1.jsonl').write_text('\n'.join(json.dumps(e) for e in evs) + '\n', encoding='utf-8')
    k = LE.key_of(evs[0])
    (live / 'records' / 'roles_20261008.jsonl').write_text(
        json.dumps({'key': k, 'role': 'staff', 'p_staff': 0.3, 'gallery_sim': 0.71, 'was': 'customer'}) + '\n' +
        json.dumps({'key': k, 'role': 'staff', 'p_staff': 0.35, 'gallery_sim': 0.74, 'was': 'customer', 'final': True}) + '\n',
        encoding='utf-8')
    out = LE.events('20261008')
    assert out[0]['role'] == 'staff' and out[0]['role_was'] == 'customer' and out[0]['p_staff'] == 0.35 and out[0]['role_final']
    assert out[1]['role'] == 'customer' and 'role_was' not in out[1]
    assert LE.key_of(out[0]) == k                                    # the key (the owner's answers) does not move
    assert LE.summary('20261008', out, {})['staff'] == 1
