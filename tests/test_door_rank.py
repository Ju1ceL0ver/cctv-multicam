import hashlib
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multicam'))
from flask import Flask
from storage import atomic_json,read_json
from door_rank_page import register


def test_rank_answers_are_separate_validated_and_images_are_content_addressed(tmp_path):
    folder=tmp_path/'data/door_identity/20261014';image=folder/'views/a/view.jpg'
    image.parent.mkdir(parents=True);image.write_bytes(b'jpeg-placeholder')
    view=dict(path=str(image),sha256=hashlib.sha256(image.read_bytes()).hexdigest(),event='a',clock='2026-10-14 10:00:00',kind='in')
    atomic_json(folder/'rank_queue.json',dict(policy='train_only',pairs=[dict(id='pair1',left=view,right=view)]))
    app=Flask(__name__);register(app,lambda:tmp_path);client=app.test_client()
    payload=client.get('/api/doorrank').get_json()
    assert 'path' not in payload['pairs'][0]['left']
    assert client.post('/api/doorrank',json=dict(id='bad',label='same')).status_code==400
    assert client.post('/api/doorrank',json=dict(id='pair1',label='bogus')).status_code==400
    assert client.post('/api/doorrank',json=dict(id='pair1',label='same')).status_code==200
    assert client.get('/api/doorrank').get_json()['answers']=={'pair1':'same'}
    assert read_json(folder/'rank_answers.json',{})['pair1']['left']==view['sha256']
    assert not (tmp_path/'data/visits/20261014/people_answers.json').exists()
    assert client.get('/api/doorrank/image?id=pair1&side=left').status_code==200
    image.write_bytes(b'changed')
    assert client.get('/api/doorrank/image?id=pair1&side=left').status_code==409
    assert client.get('/api/doorrank?day=../bad').status_code==400
