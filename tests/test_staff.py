import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))
from test_inout import _day  # noqa: E402


def _emb(tmp_path):
    """Person 1 of every frame is the same staff member (one look on every day), person 2 a different customer each time."""
    rng = np.random.default_rng(0)
    staff_c, staff_s = rng.normal(size=3840), rng.normal(size=1024)
    out = tmp_path / 'data' / 'teacher_emb'
    out.mkdir(parents=True)
    for f in (tmp_path / 'data' / 'seg_datasets' / 'pseudo' / 'drafts').glob('*.png'):
        cloth = np.stack([staff_c + rng.normal(scale=0.3, size=3840), rng.normal(size=3840)])
        shape = np.stack([staff_s + rng.normal(scale=0.3, size=1024), rng.normal(size=1024)])
        np.savez(out / (f.stem + '.npz'), labels=np.array([1, 2]), boxes=np.array([[600, 300, 660, 500], [100, 100, 130, 180]]),
                 cloth=cloth, shape=shape)


def test_build_queue_learn(tmp_path, monkeypatch):
    import staff
    staff.ONLY_CAM = ""          # the door-camera filter off: the fixture has both cameras
    staff.DEDUP = 1.01            # and no near-duplicate skipping: the fixture people are near-copies
    monkeypatch.setattr(staff, 'PCA_CLOTH', 4)
    monkeypatch.setattr(staff, 'PCA_SHAPE', 4)
    monkeypatch.setattr(staff, 'MIN_EACH', 2)
    _day(tmp_path)
    _emb(tmp_path)
    root = str(tmp_path)
    assert staff.build(root) == 24
    todo, m = staff.queue(root, 24)
    assert m['p'] is None and len(set(todo)) == 24
    assert todo[0].endswith('_p1')                       # cold start: the one who comes back every day goes first
    ids = staff.feats(root)['ids']
    for pid in ids[:4] + ids[-4:]:                       # both days
        staff.answer(root, pid, 2 if pid.endswith('_p1') else 1)
    m = staff.fit(root)
    F = staff.feats(root)
    p = m['p']
    assert all((p[F['row'][i]] >= 0.5) == i.endswith('_p1') for i in F['ids'])
    assert m['cv']['acc'] == 1.0
    todo, _ = staff.queue(root, 30)
    assert len(todo) == 16 and not set(todo) & set(staff.labels(root))


def test_pages(tmp_path, monkeypatch):
    from flask import Flask
    import staff
    staff.ONLY_CAM = ""          # the door-camera filter off: the fixture has both cameras
    staff.DEDUP = 1.01            # and no near-duplicate skipping: the fixture people are near-copies
    monkeypatch.setattr(staff, 'PCA_CLOTH', 4)
    monkeypatch.setattr(staff, 'PCA_SHAPE', 4)
    _day(tmp_path)
    _emb(tmp_path)
    staff.build(str(tmp_path))
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), '..', 'multicam', 'templates'))
    staff.register(app, str(tmp_path))
    c = app.test_client()
    assert c.get('/staff').status_code == 200
    j = c.get('/api/staff').get_json()
    pid = j['todo'][0]
    assert c.post('/api/staff', json={'id': pid, 'label': 2}).status_code == 200
    assert c.post('/api/staff', json={'id': pid, 'label': 7}).status_code == 400
    assert c.get('/staff-img/%s/crop' % pid).status_code == 200
    assert c.get('/api/staff').get_json()['done'] == 1
