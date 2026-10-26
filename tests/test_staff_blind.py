"""Held-day answers must remain separate and no model predictions may reach the page."""
import json,sys
from pathlib import Path
import pytest
from flask import Flask
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multicam'))
import staff_blind


def fixture(root):
 f=staff_blind.folder(root);(f/'images').mkdir(parents=True)
 ident='a'*32
 (f/'manifest.json').write_text(json.dumps({'days':list(staff_blind.DAYS),'missing_crops':0,
      'items':{ident:{'day':'20260911','time':'10:00:00','cam':'cam1'}}}))
 (f/'images'/(ident+'.jpg')).write_bytes(b'image')
 train=root/'data'/'staff';train.mkdir();(train/'labels.json').write_text('{"training_answer": {"label": 1}}')
 app=Flask(__name__,template_folder=str(Path(__file__).resolve().parents[1]/'multicam'/'templates'))
 staff_blind.register(app,str(root));return app.test_client(),ident,train/'labels.json'


def test_blind_answer_isolation_and_undo(tmp_path):
 c,i,training=fixture(tmp_path);before=training.read_bytes()
 j=c.get('/api/staff-test').get_json()
 assert j['todo']==[i] and j['done']==0
 assert not {'guess','cv','staff_all','counter_role','selected'}&set(j)
 assert c.post('/api/staff-test',json={'id':i,'label':True}).status_code==400
 assert c.post('/api/staff-test',json={'id':'b'*32,'label':2}).status_code==404
 assert c.post('/api/staff-test',json={'id':i,'label':2}).get_json()['done']==1
 assert c.get('/api/staff-test').get_json()['todo']==[]
 assert training.read_bytes()==before
 assert c.post('/api/staff-test',json={'id':i,'label':None}).get_json()['done']==0
 assert c.get('/api/staff-test').get_json()['todo']==[i]
 assert training.read_bytes()==before
 assert c.get('/staff-test').status_code==200
 assert b'guess' not in c.get('/staff-test').data
 assert c.get('/staff-test-img/'+i).status_code==200
 assert c.get('/staff-test-img/'+'b'*32).status_code==404
