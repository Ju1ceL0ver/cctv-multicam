"""/xcam: questions from moments both cameras see people, answers as pairs, the page."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


def _fake(monkeypatch):
    import xcam
    monkeypatch.setattr(xcam, 'candidate_times', lambda day, root: [100.0, 105.0, 200.0, 300.5])
    seen = {'cam1': [('c:1', [400, 300, 600, 900], [[200, 150], [300, 150], [300, 450], [200, 450]]), ('c:2', [1200, 200, 1400, 800], [])],
            'cam2': [('c:7', [100, 100, 300, 700], []), ('c:8', [900, 400, 1100, 1100], [[450, 200], [550, 200], [550, 550]]), ('c:9', [2000, 100, 2200, 600], [])]}
    monkeypatch.setattr(xcam, 'seen_at', lambda day, root, t: seen)
    frame = lambda day, cam, t: np.full((1440, 2560, 3), 120 if cam == 'cam1' else 60, np.uint8)
    return frame


def test_build_and_pairs(tmp_path, monkeypatch):
    import xcam
    frame = _fake(monkeypatch)
    got = xcam.build(str(tmp_path), n=6, days=('20260917',), frame_at=frame, log=lambda *a: None)
    # moments at least GAP apart: 100 and 105 cannot both be taken; two people on camera 1 each
    assert len({v['t'] for v in got.values()}) == 3 and len(got) == 6
    one = sorted(got)[0]
    assert len(got[one]['cam2']) == 3 and (tmp_path / 'data' / 'xcam' / 'img' / got[one]['frames']['cam2']).exists()
    xcam.answer(str(tmp_path), one, 2)
    p = xcam.pairs(str(tmp_path))
    assert [x[4] for x in p] == [False, True, False] and p[1][3] == 'c:8'
    xcam.answer(str(tmp_path), sorted(got)[1], -1)                     # can't tell: no pairs
    assert len(xcam.pairs(str(tmp_path))) == 3
    again = xcam.build(str(tmp_path), n=6, days=('20260917',), seed=3, frame_at=frame, log=lambda *a: None)
    assert one in again                                                 # answered questions survive a rebuild


def test_page_and_answers(tmp_path, monkeypatch):
    from flask import Flask
    import xcam
    frame = _fake(monkeypatch)
    xcam.build(str(tmp_path), n=6, days=('20260917',), frame_at=frame, log=lambda *a: None)
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(xcam.__file__), 'templates'))
    xcam.register(app, lambda: str(tmp_path))
    c = app.test_client()
    assert c.get('/xcam').status_code == 200
    j = c.get('/api/xcam').get_json()
    first = j['todo'][0]
    assert j['total'] == 6 and j['meta'][first]['offered'] == 3
    for what in ('frame1', 'frame2', 'crop1', 'crop2/3'):
        r = c.get('/xcam-img/%s/%s' % (first, what))
        assert r.status_code == 200 and r.data[:2] == b'\xff\xd8'
    assert c.get('/xcam-img/%s/crop2/4' % first).status_code == 404
    assert c.post('/api/xcam', json={'id': first, 'match': 4}).status_code == 400
    assert c.post('/api/xcam', json={'id': first, 'match': 0}).status_code == 200
    assert c.get('/api/xcam').get_json()['none'] == 1
    assert c.post('/api/xcam', json={'id': first, 'match': None}).status_code == 200
    assert c.get('/api/xcam').get_json()['done'] == 0


def test_duplicates_are_one_person():
    import xcam
    a = ('t:1', [800, 400, 1000, 1000], [[400, 200], [500, 200], [500, 500], [400, 500]])
    dup = ('t:2', [805, 405, 1000, 1000], [[402, 202], [500, 202], [500, 500], [402, 500]])
    other = ('t:3', [1600, 400, 1800, 1000], [[800, 200], [900, 200], [900, 500], [800, 500]])
    assert [r[0] for r in xcam.dedup([a, dup, other])] == ['t:1', 't:3']
