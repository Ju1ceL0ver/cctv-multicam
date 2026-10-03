import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))
import floorpair  # noqa: E402


def test_fit_recovers_a_homography_and_routes(tmp_path):
    root = str(tmp_path)
    os.makedirs(os.path.join(root, 'data', 'seg_datasets', 'backgrounds'))
    for cam in ('cam1', 'cam2'):
        cv2.imwrite(os.path.join(root, 'data', 'seg_datasets', 'backgrounds', '20260922_%s_100006_0001.jpg' % cam),
                    np.zeros((72, 128, 3), np.uint8))
    H = np.array([[0.9, 0.1, 30], [-0.05, 1.1, 12], [1e-5, 2e-5, 1]])
    a = np.array([[200, 900], [800, 1000], [1500, 1200], [2200, 950], [1200, 700], [600, 1300]], float)
    b = cv2.perspectiveTransform(a.reshape(-1, 1, 2), H).reshape(-1, 2)
    pairs = [{'c1': list(p), 'c2': list(q)} for p, q in zip(a, b)]
    f = floorpair.fit(root, pairs)
    assert f['max'] < 0.5
    assert np.allclose(floorpair.project(root, f['H'], 'cam1', a[:1]), b[:1], atol=0.5)
    from flask import Flask
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'multicam', 'templates'))
    floorpair.register(app, lambda: root)
    c = app.test_client()
    assert c.get('/floorpair').status_code == 200
    assert c.get('/api/floorpair').get_json()['backgrounds'] == [['20260922_cam1_100006_0001.jpg', '20260922_cam2_100006_0001.jpg']]
    r = c.post('/api/floorpair', json={'pairs': pairs, 'bg': ['x', 'y']}).get_json()
    assert r['fit']['max'] < 0.5
    assert json.load(open(os.path.join(root, 'data', 'floor_pairs.json')))['pairs'][0]['c1'] == [200.0, 900.0]


def test_warp_returns_an_image_with_alpha(tmp_path):
    root = str(tmp_path)
    b = os.path.join(root, 'data', 'seg_datasets', 'backgrounds')
    os.makedirs(b)
    cv2.imwrite(os.path.join(b, '20260922_cam1_100006_0001.jpg'), np.full((72, 128, 3), 200, np.uint8))
    from flask import Flask
    app = Flask(__name__)
    floorpair.register(app, lambda: root)
    H = np.eye(3); H[0, 2] = 1280                  # moved half a frame right: the left half has nothing
    r = app.test_client().get('/api/floorpair/warp?bg=20260922_cam1_100006_0001.jpg&h=' + json.dumps(H.tolist()))
    img = cv2.imdecode(np.frombuffer(r.data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert r.status_code == 200 and img.shape == (72, 128, 4)
    assert img[:, :60, 3].max() < 20 and img[:, 70:, 3].min() > 230
