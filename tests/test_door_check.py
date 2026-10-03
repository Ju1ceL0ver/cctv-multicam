import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'multicam'
sys.path.insert(0, str(ROOT))


def test_items_answers_and_routes(tmp_path, monkeypatch):
    import door_check as C
    monkeypatch.setattr(C, 'OUT', tmp_path)
    json.dump({'day': '20260919', 'items': [{'id': '20260919_1000', 'day': '20260919', 'film': 100.0, 'kind': 'in', 'p': 0.5,
                                             'box': [0.2, 0.3, 0.1, 0.3], 'answered_near': []}]},
              open(tmp_path / '20260919_items.json', 'w'))
    from flask import Flask
    app = Flask(__name__, template_folder=str(ROOT / 'templates'))
    C.register(app)
    c = app.test_client()
    assert c.get('/doorcheck').status_code == 200
    assert c.get('/api/doorcheck').get_json()['days'][0]['items'] == 1
    assert c.post('/api/doorcheck/20260919', json={'item': '20260919_1000', 'answer': 'none'}).status_code == 200
    assert C.answers('20260919')['20260919_1000']['answer'] == 'none'
    assert c.post('/api/doorcheck/20260919', json={'item': 'nope', 'answer': 'none'}).status_code == 400
    assert c.post('/api/doorcheck/20260919', json={'item': '20260919_1000', 'answer': 'maybe'}).status_code == 400
    assert c.post('/api/doorcheck/20260919', json={'item': '20260919_1000', 'answer': None}).status_code == 200
    assert C.answers('20260919') == {}
    assert c.get('/api/doorcheck/2026').status_code == 400
