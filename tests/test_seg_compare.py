import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'multicam'))


def test_the_comparison_stays_blind_and_counts_the_winner(tmp_path):
    from flask import Flask
    import seg_compare
    folder = tmp_path / 'data' / 'segcompare'
    folder.mkdir(parents=True)
    items = [{'id': '000', 'order': ['sam', 'student', 'teacher']}, {'id': '001', 'order': ['teacher', 'sam', 'student']}]
    (folder / 'manifest.json').write_text(json.dumps({'items': items}))
    (folder / '000_a.jpg').write_bytes(b'\xff\xd8x')
    app = Flask(__name__, template_folder=os.path.join(os.path.dirname(seg_compare.__file__), 'templates'))
    seg_compare.register(app, lambda: str(tmp_path))
    c = app.test_client()
    listed = c.get('/api/segcompare').get_json()
    assert listed['items'] == [{'id': '000'}, {'id': '001'}]          # which picture is which is not sent
    assert c.post('/api/segcompare', json={'id': '000', 'best': 'a'}).status_code == 200
    assert c.post('/api/segcompare', json={'id': '001', 'best': 'b'}).status_code == 200
    assert c.post('/api/segcompare', json={'id': '001', 'best': 'd'}).status_code == 400
    t = seg_compare.tally(str(tmp_path))
    assert t['best'] == {'student': 0, 'teacher': 0, 'sam': 2} and t['answered'] == 2
    # two equally good: both are credited as shared, neither as a sole winner
    assert c.post('/api/segcompare', json={'id': '000', 'best': ['c', 'a']}).status_code == 200
    assert c.post('/api/segcompare', json={'id': '000', 'best': ['a', 'x']}).status_code == 400
    assert c.post('/api/segcompare', json={'id': '000', 'best': []}).status_code == 400
    t = seg_compare.tally(str(tmp_path))
    assert t['best'] == {'student': 0, 'teacher': 0, 'sam': 1}
    assert t['best_shared'] == {'student': 0, 'teacher': 1, 'sam': 1}
    assert c.get('/segcompare-img/000_a.jpg').status_code == 200
    assert c.get('/segcompare-img/../x.jpg').status_code == 404
    assert c.get('/segcompare').status_code == 200
