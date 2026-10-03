import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


def test_scores_drafts_and_finds_new_ones(tmp_path):
    import cv2
    from flask import Flask
    import rate
    base = tmp_path / 'data' / 'seg_datasets'
    (base / 'rf' / 'train' / 'images').mkdir(parents=True)
    (base / 'pseudo' / 'drafts').mkdir(parents=True)
    frame = np.full((72, 128, 3), 90, np.uint8)
    lab = np.zeros((72, 128), np.uint8); lab[10:40, 20:40] = 1; lab[20:60, 70:90] = 2
    for i in ('a_1', 'b_2'):
        cv2.imwrite(str(base / 'rf' / 'train' / 'images' / (i + '.jpg')), frame)
        cv2.imwrite(str(base / 'pseudo' / 'drafts' / (i + '.png')), lab)
    cv2.imwrite(str(base / 'pseudo' / 'drafts' / 'orphan.png'), lab)          # no frame: not offered
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(rate.__file__), 'templates'))
    rate.register(app, lambda: str(tmp_path))
    c = app.test_client()
    j = c.get('/api/rate').get_json()
    assert sorted(j['todo']) == ['a_1', 'b_2'] and j['total'] == 2 and j['rated'] == 0
    first = j['todo'][0]
    assert c.post('/api/rate', json={'id': first, 'score': 4}).status_code == 200
    for bad in (6, -1, 2.5, True, '3'):
        assert c.post('/api/rate', json={'id': first, 'score': bad}).status_code == 400
    assert c.post('/api/rate', json={'id': 'nope', 'score': 1}).status_code == 404
    j = c.get('/api/rate').get_json()
    assert j['todo'] == [x for x in ['a_1', 'b_2'] if x != first] and j['ratings'] == {first: 4} and j['hist'][4] == 1
    assert c.post('/api/rate', json={'id': first, 'score': None}).status_code == 200     # taken back
    assert c.get('/api/rate').get_json()['rated'] == 0
    hist = (tmp_path / 'data' / 'rate' / 'history.jsonl').read_text().splitlines()
    assert [json.loads(l)['score'] for l in hist] == [4, None]
    r = c.get('/rate-img/a_1/drawn')
    assert r.status_code == 200 and r.headers['X-People'] == '2' and r.data[:2] == b'\xff\xd8'
    assert c.get('/rate-img/a_1/raw').status_code == 200
    assert c.get('/rate-img/..%2Fx/raw').status_code == 404
    assert c.get('/rate-img/a_1/other').status_code == 404
    # a new draft appears: found after a rescan
    cv2.imwrite(str(base / 'rf' / 'train' / 'images' / 'c_3.jpg'), frame)
    cv2.imwrite(str(base / 'pseudo' / 'drafts' / 'c_3.png'), lab)
    rate.index(str(tmp_path), force=True)
    assert 'c_3' in c.get('/api/rate').get_json()['todo']
    assert c.get('/rate').status_code == 200
