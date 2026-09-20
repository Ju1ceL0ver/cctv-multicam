import json
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multicam'))
from storage import atomic_json
from review_store import read_state,transact,review_groups,Conflict,fingerprint
from clip_registry import clip_key,clip_ready
from export_seg_dataset import select_frames,detection_status
from datetime import datetime

@pytest.fixture
def clip(tmp_path):
 d=tmp_path/'c100000';d.mkdir()
 pieces=[{'piece':0,'cam':'cam1','t0':0.,'t1':3.,'dets':[0,1,2,3]},
         {'piece':1,'cam':'cam1','t0':0.,'t1':3.,'dets':[4,5,6,7]}]
 atomic_json(d/'pieces_yolo26x-seg.json',pieces)
 atomic_json(d/'groups_yolo26x-seg.json',[{'person':0,'pieces':[0,1]}])
 atomic_json(d/'meta_yolo26x-seg.json',{'day':'20260917','start':'2026-09-17T10:00:00','seconds':4})
 rows=np.zeros((8,10));rows[:,0]=[0,1,2,3,0,1,2,3]
 np.savez(d/'dets_yolo26x-seg.npz',cam1=rows,cam2=np.empty((0,10)))
 return d

def test_split_group_is_complete_and_undo_survives_reload(clip):
 a=transact(clip,{'pieces':[0],'label':'P1'},0)
 b=transact(clip,{'pieces':[1],'label':'P2'},a['revision'])
 assert review_groups(clip)[0]['done'] and review_groups(clip)[0]['mixed']
 with pytest.raises(Conflict):transact(clip,{'pieces':[0],'label':'P3'},0)
 restored=transact(clip,{'action':'undo'},b['revision'])
 assert restored['labels']=={'0':'P1'}
 assert not review_groups(clip)[0]['done']
 assert (clip/'review_history/00000000.json').exists()

def test_same_size_correction_changes_dataset_hash(clip):
 a=transact(clip,{'pieces':[0],'label':'P1'})
 b=transact(clip,{'pieces':[0],'label':'?'})
 assert len(a['labels'])==len(b['labels'])
 assert fingerprint(a)!=fingerprint(b)

def test_split_preserves_detection_coverage_and_undo(clip):
 a=transact(clip,{'pieces':[0,1],'label':'P1'})
 b=transact(clip,{'action':'split','piece':0,'at':1.5},a['revision'])
 assert b['pieces'][0]['dets']==[0,1]
 assert b['pieces'][-1]['dets']==[2,3]
 assert '2' not in b['labels'] and '0' not in b['labels']
 assert set(review_groups(clip)[0]['pieces'])=={0,1,2}
 c=transact(clip,{'action':'undo'},b['revision'])
 assert c['pieces']==a['pieces'] and c['labels']==a['labels']

def test_unresolved_person_excludes_whole_frame():
 rows=np.zeros((4,10));rows[:,0]=[0,0,1,1]
 selected,skipped=select_frames(rows,{0:'positive',1:'unresolved',2:'positive',3:'negative'},np.arange(5)*4)
 assert selected=={25:[2]} and skipped==1

def test_unknown_identity_can_have_verified_mask(clip):
 a=transact(clip,{'pieces':[0],'label':'?'})
 assert detection_status(a,'cam1')[0]=='unresolved'
 b=transact(clip,{'action':'quality','pieces':[0],'quality':'valid'})
 assert detection_status(b,'cam1')[0]=='positive'
 assert read_state(clip)['labels']['0']=='?'

def test_day_keys_preserve_legacy_without_collision(clip):
 t=datetime(2026,9,17,10)
 assert clip_key('20260917',t,clip.parent)=='c100000'
 assert clip_key('20260918',t.replace(day=18),clip.parent)=='c20260918_100000'
 assert not clip_ready(clip)

def test_http_transactions_validate_revision(clip,monkeypatch):
 import label_pieces as web
 monkeypatch.setattr(web,'CLIPS',str(clip.parent));web.app.config['TESTING']=True
 c=web.app.test_client();c.set_cookie('labeler_key',web.KEY)
 r=c.get('/api/groups/c100000');assert r.status_code==200
 assert r.json['groups'][0]['detail'][0]['raw_t1']==3
 assert c.post('/api/review/c100000',json={'action':'label','pieces':[0],'label':'NEW','revision':0}).status_code==200
 assert c.post('/api/review/c100000',json={'action':'label','pieces':[1],'label':'NEW','revision':0}).status_code==409
 assert c.get('/people').status_code==200
 assert c.get('/static/review.js').status_code==200
 assert c.get('/api/groups/not-a-clip').status_code==400

def test_false_detection_is_reviewed_and_removed_from_identity_gt(clip):
 transact(clip,{'pieces':[0,1],'label':'P1'})
 s=transact(clip,{'action':'quality','pieces':[0],'quality':'false_positive'})
 gt=json.loads((clip/'gt_identity_yolo26x-seg.json').read_text())
 assert '0' not in gt['cam1'] and '4' in gt['cam1']
 assert review_groups(clip)[0]['done']

def test_human_pair_answer_assigns_only_selected_pieces(clip):
 s=transact(clip,{'action':'relation','a':0,'b':1,'decision':'same'})
 assert s['labels']['0']==s['labels']['1']
 s=transact(clip,{'action':'relation','a':0,'b':1,'decision':'different'})
 assert '1' not in s['labels']
 assert len(s['relations'])==1

def test_day_links_preserve_different_confirmed_people(tmp_path):
 from day_visits import build_day,decide
 root=tmp_path/'project';clips=root/'data/raw_clips';clips.mkdir(parents=True)
 for name,label in [('c_a','P1'),('c_b','P2')]:
  d=clips/name;d.mkdir()
  atomic_json(d/'meta_yolo26x-seg.json',{'day':'20260917','start':'2026-09-17T10:00:00','seconds':10})
  atomic_json(d/'pieces_yolo26x-seg.json',[{'piece':0,'cam':'cam1','t0':0,'t1':2,'dets':[0,1]}])
  atomic_json(d/'gt_manual.json',{'0':label})
  rows=np.zeros((2,10));rows[:,0]=[0,1];rows[:,1:5]=[10,20,50,100]
  np.savez(d/'dets_yolo26x-seg.npz',cam1=rows,cam2=np.empty((0,10)))
 result=build_day('20260917',root)
 assert len(result['visits'])==1 and result['links'][0]['source']=='shared_raw_frames'
 by={n['key']:n for n in result['nodes']};a,b=sorted(by)
 result=decide('20260917',{'a':a,'b':b,'decision':'different','revision':0,
   'evidence_a':by[a]['evidence'],'evidence_b':by[b]['evidence']},root)
 assert len(result['visits'])==2
 result=decide('20260917',{'action':'undo','revision':1},root)
 assert len(result['visits'])==1
 assert not result['can_undo']
 with pytest.raises(ValueError):decide('20260917',{'action':'undo','revision':2},root)


def test_managed_recorder_preserves_unprocessed_files(tmp_path,monkeypatch):
 import importlib.util
 from types import SimpleNamespace
 path=Path(__file__).resolve().parents[1]/'jobs/raw_recorder.py'
 spec=importlib.util.spec_from_file_location('recorder_test',path)
 recorder=importlib.util.module_from_spec(spec);spec.loader.exec_module(recorder)
 monkeypatch.setattr(recorder,'OUT',str(tmp_path));monkeypatch.setattr(recorder,'MANAGED_RETENTION',True)
 video=tmp_path/'old.mp4';video.write_bytes(b'footage')
 monkeypatch.setattr(recorder.shutil,'disk_usage',lambda _:SimpleNamespace(free=149*2**30))
 recorder.enforce_disk_cap()
 assert video.exists() and not recorder.capture_has_space()
 monkeypatch.setattr(recorder.shutil,'disk_usage',lambda _:SimpleNamespace(free=151*2**30))
 assert recorder.capture_has_space()
 monkeypatch.setattr(recorder,'PIDFILE',str(tmp_path/'recorder.pid'))
 first=recorder.acquire_camera_lock()
 assert first is not None and recorder.acquire_camera_lock() is None
 first.close()
 recovered=recorder.acquire_camera_lock()
 assert recovered is not None
 recovered.close()


def test_preview_times_are_raw_not_shifted(clip,monkeypatch):
 import label_pieces as web
 state=read_state(clip);state['pieces'][0]['t0']=4.24;state['pieces'][0]['t1']=7.24
 atomic_json(clip/'pieces_yolo26x-seg.json',state['pieces'])
 monkeypatch.setattr(web,'CLIPS',str(clip.parent));web.app.config['TESTING']=True
 client=web.app.test_client();client.set_cookie('labeler_key',web.KEY)
 piece=client.get('/api/pieces/c100000').json['pieces'][0]
 assert piece['raw_t0']==0 and piece['raw_t1']==3 and piece['t0']==4.24
