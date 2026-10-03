import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


def test_edits_are_stored_as_they_were_painted_and_never_lost(tmp_path):
    import cv2
    from flask import Flask
    import paint
    folder = tmp_path / 'data' / 'paint'
    folder.mkdir(parents=True)
    json.dump({'items': [{'id': '000', 'machine_people': 2}]}, open(folder / 'manifest.json', 'w'))
    init = np.zeros((paint.H, paint.W), np.uint8)
    init[10:50, 10:40] = 1
    cv2.imwrite(str(folder / '000_init.png'), init)
    cv2.imwrite(str(folder / '000.jpg'), np.zeros((paint.H, paint.W, 3), np.uint8))
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(paint.__file__), 'templates'))
    paint.register(app, lambda: str(tmp_path))
    c = app.test_client()
    first = c.get('/paint-img/000/labels')
    assert cv2.imdecode(np.frombuffer(first.data, np.uint8), cv2.IMREAD_UNCHANGED).max() == 1   # the machine's draft
    edit = init.copy()
    edit[60:90, 60:90] = 7                       # somebody the machine missed
    rgba = np.dstack([edit, edit, edit, np.full_like(edit, 255)])            # what the page's canvas sends
    png = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', rgba)[1]).decode()
    r = c.post('/api/paint/000', json={'png': png, 'done': True})
    assert r.status_code == 200 and r.get_json()['people'] == 2
    back = cv2.imdecode(np.frombuffer(c.get('/paint-img/000/labels').data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert (back == edit).all()
    c.post('/api/paint/000', json={'png': png, 'done': True})
    assert len(list((folder / 'history').glob('000_*.png'))) == 1          # the earlier edit is kept
    small = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', edit[:10])[1]).decode()
    assert c.post('/api/paint/000', json={'png': small}).status_code == 400
    assert c.get('/paint-img/../x/frame').status_code == 404
    assert c.get('/paint').status_code == 200


def test_specks_leave_unedited_drafts_only(tmp_path, monkeypatch):
    import cv2
    import paint
    monkeypatch.setattr(paint, 'OUT', tmp_path)
    json.dump({'items': [{'id': '000'}, {'id': '001'}]}, open(tmp_path / 'manifest.json', 'w'))
    lab = np.zeros((paint.H, paint.W), np.uint8)
    lab[100:200, 100:150] = 3          # a person
    lab[400:405, 400:405] = 3          # a speck of the same person
    lab[500:503, 600:603] = 4          # a person that is nothing but a speck
    cv2.imwrite(str(tmp_path / '000_init.png'), lab)
    cv2.imwrite(str(tmp_path / '001_init.png'), lab)
    cv2.imwrite(str(tmp_path / '001_mask.png'), lab)         # the owner has been here: hands off
    assert paint.clean_drafts() == 1
    out = cv2.imread(str(tmp_path / '000_init.png'), cv2.IMREAD_UNCHANGED)
    assert (out == 3).sum() == 100 * 50 and not (out == 4).any()
    assert (cv2.imread(str(tmp_path / '001_init.png'), cv2.IMREAD_UNCHANGED) == lab).all()


def test_batches_are_listed_and_more_starts_once(tmp_path, monkeypatch):
    import cv2
    from flask import Flask
    import bg
    import paint
    folder = tmp_path / 'data' / 'paint'
    folder.mkdir(parents=True)
    items = [{'id': '%03d' % k, 'machine_people': 1, 'day': '2026091%d' % (7 + k % 3), 'batch': 1 if k < 3 else 2}
             for k in range(6)]
    json.dump({'items': items}, open(folder / 'manifest.json', 'w'))
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(paint.__file__), 'templates'))
    paint.register(app, lambda: str(tmp_path))
    c = app.test_client()
    j = c.get('/api/paint').get_json()
    assert [it['batch'] for it in j['items']] == [1, 1, 1, 2, 2, 2] and j['job'] is None
    started = []
    monkeypatch.setattr(bg, 'running', lambda pattern: [])
    monkeypatch.setattr(bg, 'spawn', lambda name, args, **kw: started.append(args))
    assert c.post('/api/paint/more').status_code == 200 and started == [['paint.py', 'more', '100']]
    monkeypatch.setattr(bg, 'running', lambda pattern: ['123|python paint.py more 100'])
    assert c.post('/api/paint/more').status_code == 409          # never two harvests at once


def test_python_for_jobs_is_never_pythonw(tmp_path, monkeypatch):
    import bg
    (tmp_path / 'python.exe').write_text('')
    monkeypatch.setattr(bg.sys, 'executable', str(tmp_path / 'pythonw.exe'))
    assert bg._python() == str(tmp_path / 'python.exe')
