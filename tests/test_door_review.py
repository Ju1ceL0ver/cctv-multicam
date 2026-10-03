import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


@pytest.fixture
def site(tmp_path, monkeypatch):
    import cv2
    from flask import Flask
    import door_review
    root = tmp_path / 'multicam'
    live = tmp_path / 'retail_analytics' / 'runs' / 'live'
    snaps = live / 'snapshots' / '20260918'
    snaps.mkdir(parents=True)
    root.mkdir()
    monkeypatch.delenv('RA_LIVE_DIR', raising=False)
    rows = []
    for k, (t, ev) in enumerate([('10:00:01', 'entry'), ('10:00:03', 'exit'), ('10:05:00', 'exit'),
                                 ('2026-09-19T10:00:00', 'entry')]):
        full, crop = snaps / ('%d_full.jpg' % k), snaps / ('%d_crop.jpg' % k)
        cv2.imwrite(str(full), np.full((1440, 2560, 3), 90, np.uint8))
        cv2.imwrite(str(crop), np.full((200, 80, 3), 90, np.uint8))
        stamp = t if 'T' in t else '2026-09-18T%s.000+07:00' % t
        rows.append({'event_id': '%032x' % (k + 1), 'time_local': stamp, 'unix_ms': 1789700403000 + k * 1000,
                     'event': ev, 'role': 'customer', 'global_id': k, 'track_id': k,
                     'box_x1': 1000, 'box_y1': 400, 'box_x2': 1200, 'box_y2': 900,
                     'snapshot_full': str(full), 'snapshot_crop': str(crop)})
    rows.append({'event_id': '%032x' % 99, 'time_local': '2026-09-18T11:00:00.000+07:00', 'unix_ms': 0,
                 'event': 'entry', 'role': 'customer', 'global_id': 1, 'track_id': 1,
                 'box_x1': 0, 'box_y1': 0, 'box_x2': 1, 'box_y2': 1,
                 'snapshot_full': str(tmp_path / 'elsewhere.jpg'), 'snapshot_crop': str(tmp_path / 'elsewhere.jpg')})
    (tmp_path / 'elsewhere.jpg').write_bytes(b'x')
    (live / 'entrance_events.jsonl').write_text('\n'.join(json.dumps(r) for r in rows) + '\n{"half', encoding='utf-8')
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(door_review.__file__), 'templates'))
    door_review.register(app, lambda: str(root))
    return app.test_client(), rows, root


def test_day_lists_only_that_day_in_time_order(site):
    client, rows, _ = site
    assert client.get('/api/door').get_json()['days'] == ['20260918', '20260919']
    j = client.get('/api/door/20260918').get_json()
    assert [e['time'] for e in j['events']] == ['10:00:01', '10:00:03', '10:05:00', '11:00:00']
    assert j['film'] is False and j['events'][0]['film'] is None


def test_answers_undo_and_summary(site):
    client, rows, root = site
    ent, ext = rows[0]['event_id'], rows[1]['event_id']
    assert client.post('/api/door/20260918', json={'event': ent, 'kind': 'in'}).get_json()['label']['person'] == ent
    r = client.post('/api/door/20260918', json={'event': ext, 'kind': 'out', 'person': ent})
    assert r.get_json()['label']['person'] == ent
    s = client.get('/api/door/20260918/summary').get_json()
    assert s['visits'] == 1 and s['visits_whole'] == 1 and s['counter_customer_entries'] == 2
    assert client.post('/api/door/20260918', json={'event': ext, 'kind': None}).get_json()['label'] is None
    assert client.get('/api/door/20260918/summary').get_json()['visits_whole'] == 0
    history = (root / 'data/door_review/20260918.history.jsonl').read_text().splitlines()
    assert len(history) == 3


def test_clear_keeps_history(site):
    import door_review
    client, rows, root = site
    client.post('/api/door/20260918', json={'event': rows[0]['event_id'], 'kind': 'in'})
    state = door_review.clear('20260918', str(root))
    assert state['labels'] == {} and state['revision'] == 2
    assert 'cleared' in (root / 'data/door_review/20260918.history.jsonl').read_text().splitlines()[-1]


def _ev(k, t):
    return {'event_id': k, 'unix_ms': int(t * 1000), 'time_local': '2026-09-18T10:%02d:%02d' % divmod(t, 60)}


def test_exit_closes_the_latest_entry_of_that_person():
    """She came in, went out, came back and went out again: two visits, each exit with the
    entry right before it -- not the second exit with the first entry."""
    import door_review
    ev = {k: _ev(k, t) for k, t in (('a', 10), ('b', 100), ('c', 200), ('d', 300))}
    lab = {'a': {'kind': 'in', 'person': 'a'}, 'b': {'kind': 'out', 'person': 'a'},
           'c': {'kind': 'in', 'person': 'a'}, 'd': {'kind': 'out', 'person': 'a'}}
    V = door_review.visits(ev, lab)
    assert [(v['entry'], v['exit']) for v in V] == [('a', 'b'), ('c', 'd')]


def test_later_exit_wins_and_first_entry_stands():
    import door_review
    ev = {k: _ev(k, t) for k, t in (('a', 10), ('a2', 20), ('b', 100), ('b2', 130))}
    lab = {'a': {'kind': 'in', 'person': 'a'}, 'a2': {'kind': 'in', 'person': 'a'},
           'b': {'kind': 'out', 'person': 'a'}, 'b2': {'kind': 'out', 'person': 'a'}}
    V = door_review.visits(ev, lab)
    assert [(v['entry'], v['exit'], v['seconds']) for v in V] == [('a', 'b2', 120.0)]
    # an exit whose entry nobody saw is a visit without a start
    V = door_review.visits(ev, {'b': {'kind': 'out', 'person': 'unseen'}})
    assert [(v['entry'], v['exit']) for v in V] == [(None, 'b')]


def test_person_must_have_come_in_before(site):
    client, rows, _ = site
    ent, ext, late = rows[0]['event_id'], rows[1]['event_id'], rows[2]['event_id']
    client.post('/api/door/20260918', json={'event': ext, 'kind': 'in'})
    # the exit at 10:00:01 cannot belong to somebody who came in at 10:00:03
    assert client.post('/api/door/20260918', json={'event': ent, 'kind': 'out', 'person': ext}).status_code == 400
    # a person is named by the entry that first brought them in, not by an exit
    client.post('/api/door/20260918', json={'event': ent, 'kind': 'none'})
    assert client.post('/api/door/20260918', json={'event': late, 'kind': 'out', 'person': ent}).status_code == 400
    assert client.post('/api/door/20260918', json={'event': late, 'kind': 'out', 'person': ext}).status_code == 200


def test_pictures_only_from_the_snapshot_folder(site):
    client, rows, _ = site
    e = rows[0]['event_id']
    assert client.get('/api/door/20260918/snap/%s/crop' % e).status_code == 200
    z = client.get('/api/door/20260918/snap/%s/zoom' % e)
    assert z.status_code == 200 and z.data[:2] == b'\xff\xd8'
    assert client.get('/api/door/20260918/snap/%032x/crop' % 99).status_code == 404
    assert client.get('/api/door/20260918/snap/%s/other' % e).status_code == 404


def test_page_renders(site):
    client, _, _ = site
    r = client.get('/door')
    assert r.status_code == 200 and 'door.js' in r.get_data(as_text=True)


def test_staff_are_people_with_their_own_visits(site):
    """A member of staff comes in and goes out like anybody, and the role stays with the person."""
    client, rows, root = site
    ent, ext, late = rows[0]['event_id'], rows[1]['event_id'], rows[2]['event_id']
    r = client.post('/api/door/20260918', json={'event': ent, 'kind': 'in', 'role': 'staff'})
    assert r.get_json()['label']['person'] == ent
    client.post('/api/door/20260918', json={'event': ext, 'kind': 'out', 'person': ent})
    s = client.get('/api/door/20260918/summary').get_json()
    assert s['visits'] == 0 and s['staff_visits'] == 1 and s['list'][0]['staff']
    # a staff exit whose entry was not seen still counts as staff
    client.post('/api/door/20260918', json={'event': late, 'kind': 'out', 'person': 'unseen', 'role': 'staff'})
    assert client.get('/api/door/20260918/summary').get_json()['staff_visits'] == 2
    # said the other way round once, every visit of that person follows
    assert client.post('/api/door/20260918/role', json={'person': ent, 'role': 'customer'}).status_code == 200
    s = client.get('/api/door/20260918/summary').get_json()
    assert s['visits'] == 1 and s['staff_visits'] == 1
    assert client.post('/api/door/20260918/role', json={'person': ext, 'role': 'staff'}).status_code == 400
    assert client.post('/api/door/20260918', json={'event': ent, 'kind': 'none', 'role': 'staff'}).status_code == 400
