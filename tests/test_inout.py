import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


def _day(tmp_path):
    import cv2
    base = tmp_path / 'data' / 'seg_datasets'
    (base / 'pseudo' / 'train' / 'images').mkdir(parents=True)
    (base / 'pseudo' / 'drafts').mkdir(parents=True)
    frame = np.full((720, 1280, 3), 90, np.uint8)
    lab = np.zeros((720, 1280), np.uint8)
    lab[300:500, 600:660] = 1          # on the cam1 floor
    lab[100:180, 100:130] = 2          # far up in the gallery
    lab[10:12, 10:12] = 3              # a speck: not a person
    for day in ('20260917', '20260919'):
        for cam in ('cam1', 'cam2'):
            for k in range(3):
                i = '%s_%s_100000_0000_%03d' % (day, cam, k)
                cv2.imwrite(str(base / 'pseudo' / 'train' / 'images' / (i + '.jpg')), frame)
                cv2.imwrite(str(base / 'pseudo' / 'drafts' / (i + '.png')), lab)


def test_build_spreads_over_days_and_keeps_answers(tmp_path):
    import inout
    _day(tmp_path)
    got = inout.build(str(tmp_path), n=8)
    assert len(got) == 8
    assert {v['day'] for v in got.values()} == {'20260917', '20260919'}
    assert {v['cam'] for v in got.values()} == {'cam1', 'cam2'}
    assert all(v['value'] in (1, 2) for v in got.values())          # the speck is never offered
    one = sorted(got)[0]
    inout.answer(str(tmp_path), one, 2)
    again = inout.build(str(tmp_path), n=8, seed=5)                  # a rebuild keeps what was answered
    assert one in again and again[one] == got[one]


def test_crop_has_room_around_the_person():
    import inout
    x1, y1, x2, y2 = inout.crop_box([600, 300, 660, 500])
    assert (x1, x2) == (570, 690) and (y1, y2) == (275, 525)
    assert inout.crop_box([0, 0, 40, 700]) == [0, 0, 60, 720]         # clipped to the frame


def test_pages_and_answers(tmp_path):
    from flask import Flask
    import inout
    _day(tmp_path)
    inout.build(str(tmp_path), n=6)
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(inout.__file__), 'templates'))
    inout.register(app, lambda: str(tmp_path))
    c = app.test_client()
    assert c.get('/inout').status_code == 200
    j = c.get('/api/inout').get_json()
    assert len(j['todo']) == 6 and j['done'] == 0 and j['total'] == 6
    first = j['todo'][0]
    assert set(j['meta'][first]) == {'day', 'cam', 'near'}
    assert c.post('/api/inout', json={'id': first, 'label': 1}).status_code == 200
    for bad in (6, -1, 1.5, True, '1'):
        assert c.post('/api/inout', json={'id': first, 'label': bad}).status_code == 400
    assert c.post('/api/inout', json={'id': 'nope', 'label': 1}).status_code == 404
    j = c.get('/api/inout').get_json()
    assert first not in j['todo'] and j['labels'] == {first: 1} and j['hist']['1'] == 1
    assert c.post('/api/inout', json={'id': first, 'label': None}).status_code == 200
    assert c.get('/api/inout').get_json()['done'] == 0
    hist = (tmp_path / 'data' / 'inout' / 'history.jsonl').read_text().splitlines()
    assert [json.loads(l)['label'] for l in hist] == [1, None]
    for what in ('crop', 'plain', 'frame'):
        r = c.get('/inout-img/%s/%s' % (first, what))
        assert r.status_code == 200 and r.data[:2] == b'\xff\xd8'
    assert c.get('/inout-img/%s/other' % first).status_code == 404
    assert c.get('/inout-img/..%2Fx/crop').status_code == 404
