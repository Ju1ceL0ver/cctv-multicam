import base64
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


def test_take_as_is_take_with_edits_and_skip(tmp_path):
    import cv2
    from flask import Flask
    import fix
    folder = tmp_path / 'data' / 'fix'
    folder.mkdir(parents=True)
    H, W = 120, 200
    items = [{'id': 'w_cam1_%05d' % k, 'batch': 1, 'set': 'test', 'tag': 'w', 'cam': 'cam1', 'tick': k, 'w': W, 'h': H,
              'people': 1, 'persons': [5]} for k in range(3)]
    json.dump({'items': items, 'batches': {'1': 'test'}}, open(folder / 'manifest.json', 'w'))
    init = np.zeros((H, W), np.uint8)
    init[10:50, 10:40] = 1
    for it in items:
        cv2.imwrite(str(folder / ('%s_init.png' % it['id'])), init)
        cv2.imwrite(str(folder / ('%s.jpg' % it['id'])), np.zeros((H, W, 3), np.uint8))
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(fix.__file__), 'templates'))
    fix.register(app, lambda: str(tmp_path))
    c = app.test_client()
    a, b, d = (it['id'] for it in items)
    r = c.post('/api/fix/' + a, json={'verdict': 'take', 'png': None}).get_json()
    assert r == dict(r, verdict='take', edited=False, people=1)             # good as the teacher drew it
    edit = init.copy()
    edit[60:90, 60:90] = 7                                                  # somebody the teacher missed
    rgba = np.dstack([edit, edit, edit, np.full_like(edit, 255)])           # what the page's canvas sends
    png = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', rgba)[1]).decode()
    r = c.post('/api/fix/' + b, json={'verdict': None, 'png': png}).get_json()
    assert r['verdict'] is None and r['edited']                             # edits kept, no verdict yet
    r = c.post('/api/fix/' + b, json={'verdict': 'take', 'png': None}).get_json()
    assert r == dict(r, verdict='take', edited=True, people=2)              # the verdict keeps the stored edit
    back = cv2.imdecode(np.frombuffer(c.get('/fix-img/%s/labels' % b).data, np.uint8), cv2.IMREAD_UNCHANGED)
    assert (back == edit).all()
    assert c.post('/api/fix/' + d, json={'verdict': 'skip'}).get_json()['verdict'] == 'skip'
    small = 'data:image/png;base64,' + base64.b64encode(cv2.imencode('.png', edit[:10])[1]).decode()
    assert c.post('/api/fix/' + a, json={'verdict': 'take', 'png': small}).status_code == 400
    assert c.post('/api/fix/' + a, json={'verdict': 'maybe'}).status_code == 400
    assert c.post('/api/fix/nobody', json={'verdict': 'take'}).status_code == 400
    lst = c.get('/api/fix').get_json()
    assert len(lst['items']) == 3 and lst['state'][d]['verdict'] == 'skip' and lst['batches'] == {'1': 'test'}
    assert c.get('/fix').status_code == 200
