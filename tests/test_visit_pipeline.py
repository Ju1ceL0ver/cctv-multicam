import sys
from pathlib import Path
from datetime import datetime
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'multicam'))
import visits_build as vb
import night_visits as nv


def node(key,start,end,point=(0.,0.),embedding=(1.,0.)):
    return dict(key=key,clip=key.split(':')[0],t=np.array([start,end]),xy=np.array([point,point]),
                embedding=np.array(embedding),model='osnet')


def test_seams_reject_ambiguous_and_missing_appearance():
    a=node('a:1',0,600);b=node('b:1',600,1200)
    groups,links=vb.join_windows([a,b]);assert len(groups)==1 and len(links)==1
    c=node('b:2',600,1200)
    groups,links=vb.join_windows([a,b,c]);assert len(groups)==3 and not links
    b['embedding']=np.array([])
    assert len(vb.join_windows([a,b])[0])==2


def test_same_window_never_merged_and_impossible_walk_rejected():
    a=node('a:1',0,10);b=node('a:2',11,20)
    assert not vb.join_windows([a,b])[1]
    b=node('b:1',10,20,(10.,0.))
    assert not vb.join_windows([a,b])[1]


def test_anchor_one_to_one_ambiguity_and_distance():
    door=dict(door_a=[0,0],door_b=[1,0])
    a=node('a:1',10,20)
    event=dict(kind='in',seconds=10,key='e',clock='2026-10-15T00:00:10')
    assert vb.anchor([[a]],[event],'in',door)[0]['key']=='e'
    assert not vb.anchor([[a],[a]],[event],'in',door)
    assert not vb.anchor([[node('b:1',10,20,(5.,5.))]],[event],'in',door)


def test_no_smoothing_through_recording_holes():
    a=node('a:1',0,10);a['t']=np.array([0.,.2,10.,10.2]);a['xy']=np.zeros((4,2))
    t,xy=vb.trajectory([a]);assert not np.any((t>.3)&(t<9.9))


def test_night_waits_for_recording_date(monkeypatch):
    monkeypatch.setattr(nv,'DAY','20261015');monkeypatch.setattr(nv,'SPAN','13:01-17:00')
    assert not nv.night(datetime(2026,10,15,8,22))
    assert nv.night(datetime(2026,10,15,21,5))
    assert nv.night(datetime(2026,10,16,8,22))
    assert not nv.night(datetime(2026,10,16,12))


def test_page_validates_paths_and_persists_separate_review(tmp_path):
    from flask import Flask
    from visit_whole_page import register
    from storage import atomic_json
    app=Flask(__name__);register(app,lambda:tmp_path)
    atomic_json(tmp_path/'data/visits/20261015/index.json',dict(day='20261015',visits=[dict(id='v_a_1')]))
    client=app.test_client()
    assert client.get('/api/wholevisits?day=../').status_code==400
    assert client.get('/api/wholevisits?day=20261015').status_code==200
    assert client.post('/api/wholevisits/review',json=dict(day='20261015',id='v_a_1',purity='clean',evidence=client.get('/api/wholevisits?day=20261015').json['visits'][0]['evidence'])).status_code==200
    assert client.get('/api/wholevisits?day=20261015').json['review']['v_a_1']['purity']=='clean'
    assert client.post('/api/wholevisits/review',json=dict(day='20261015',id='v_a_1',purity='clean',evidence='stale')).status_code==409


def test_customer_recording_grace_roles_restarts_and_idle():
    from customer_recording import state
    staff=[dict(t=10,kind='in',role='staff')]
    assert not state(staff,11)['record']
    events=staff+[dict(t=20,kind='in',role='customer')]
    assert state(events,21)['customers']==1
    assert state(events+events,21)['customers']==1
    assert not state(events,2720)['record']
    events.append(dict(t=30,kind='out',role='customer'))
    assert state(events,89)['record']
    assert not state(events,90)['record']
    assert not state([dict(t=30,kind='out',role='customer')],31)['record']
    assert not state([dict(t=20,kind='in',role=None)],21)['record']


def test_managed_recorder_finalizes_mp4_on_stop(monkeypatch,tmp_path):
    import importlib.util, io
    spec=importlib.util.spec_from_file_location('raw_record',Path(__file__).resolve().parents[1]/'jobs/raw_recorder.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    monkeypatch.setattr(mod,'OUT',str(tmp_path));monkeypatch.setattr(mod,'log',lambda *a:None)
    monkeypatch.setattr(mod,'capture_has_space',lambda:True)
    stopped=[False];monkeypatch.setattr(mod,'stop_requested',lambda:stopped[0])
    monkeypatch.setattr(mod.time,'sleep',lambda t:stopped.__setitem__(0,True))
    class Sink(io.BytesIO):
        def close(self): self.saved=self.getvalue();super().close()
    class Process:
        def __init__(self): self.stdin=Sink();self.stderr=io.BytesIO();self.terminated=False
        def poll(self): return None
        def wait(self,timeout): return 0
        def terminate(self): self.terminated=True
    proc=Process()
    monkeypatch.setattr(mod.subprocess,'Popen',lambda *a,**kw:proc)
    from datetime import timedelta
    mod.record_one_day('ffmpeg','rtsp://test',datetime.now()+timedelta(seconds=60))
    assert proc.stdin.saved==b'q\n'
    assert not proc.terminated


def test_keeper_exempts_only_authorized_night_child(monkeypatch):
    import ast, os, psutil
    path=Path(__file__).resolve().parents[1]/'multicam/keeper.py'
    tree=ast.parse(path.read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='authorized_visit_worker')
    ns={'datetime':datetime,'os':os,'ROOT':'/store'}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),ns)
    marker=[True]
    class Parent:
        def cmdline(self):return ['python','night_visits.py']
        def cwd(self):return '/store'
    class Worker:
        def environ(self):return {'RA_VISITS_NIGHT':'1'} if marker[0] else {}
        def parents(self):return [Parent()]
    monkeypatch.setattr(psutil,'Process',lambda pid:Worker())
    check=ns['authorized_visit_worker']
    assert check(123,datetime(2026,10,18,22))
    assert not check(123,datetime(2026,10,19,9,30))
    assert not check(123,datetime(2026,10,19,13))
    marker[0]=False
    assert not check(123,datetime(2026,10,18,22))


def test_build_keeps_door_evidence_and_does_not_claim_verification(tmp_path):
    from storage import atomic_json
    import json
    base=tmp_path/'data/raw_clips/w1';base.mkdir(parents=True)
    atomic_json(base/'people_yolo26x-seg.json',dict(start='2026-10-15T13:00:00',people=[dict(id=0,t=[10,10.2,19.8,20],xy=[[0,0]]*4,cams=['cam1'],height_m=1.7)]))
    atomic_json(base/'meta_yolo26x-seg.json',dict(seconds=600))
    atomic_json(tmp_path/'data/plan_door.json',dict(door_a=[0,0],door_b=[1,0]))
    live=tmp_path/'data/live';live.mkdir(parents=True)
    events=[dict(engine='doorseg_c+zones',t=1,kind='in',clock='2026-10-15T13:00:10',role='customer'),dict(engine='doorseg_c+zones',t=2,kind='out',clock='2026-10-15T13:00:20',role='customer')]
    (live/'events_cam1.jsonl').write_text('\n'.join(json.dumps(e) for e in events))
    result=vb.build('20261015',root=tmp_path,media=False)
    v=result['visits'][0]
    assert v['door_bounded'] and v['role']=='customer'
    assert v['status']=='candidate' and not v['owner_verified']
    assert len(v['coverage_gaps'])==1
    assert v['zones']==[] and result['quality']['verified_visits']==0


def test_media_retention_preserves_owner_labels_and_unrelated_files(tmp_path):
    from storage import atomic_json
    from visit_media import prune_media
    base=tmp_path/'data/visits/20260901';base.mkdir(parents=True)
    atomic_json(base/'index.json',dict(visits=[dict(id='v_w_1',media=dict(mp4='v_w_1.mp4',gif='v_w_1.gif'))]))
    for name in ['v_w_1.mp4','v_w_1.gif','owner.mp4','people_answers.json','whole_review.json']:
        (base/name).write_text('preserve')
    assert prune_media(tmp_path,datetime(2026,10,18))==2
    assert (base/'owner.mp4').exists() and (base/'people_answers.json').exists()
    assert (base/'whole_review.json').exists() and (base/'index.json').exists()


def test_video_preflight_rejects_unreadable_segments_and_missing_start(monkeypatch):
    import rawsource,cv2
    from datetime import timedelta
    start=datetime(2026,10,15,13)
    records=[('good',start),('good2',start+timedelta(minutes=15))]
    monkeypatch.setattr(rawsource,'segments',lambda cam,day:records)
    class Capture:
        def __init__(self,path): self.path=path
        def get(self,key):return -1 if self.path=='bad' else 22500
        def isOpened(self):return self.path!='bad'
        def release(self):pass
    monkeypatch.setattr(cv2,'VideoCapture',Capture)
    assert nv.coverage('20261015',start+timedelta(minutes=10),600) is None
    records[1]=('bad',start+timedelta(minutes=15))
    assert 'no readable' in nv.coverage('20261015',start+timedelta(minutes=10),600)
    assert 'no readable' in nv.coverage('20261015',start-timedelta(seconds=1),600)


def test_visit_crop_uses_full_camera_coordinates():
    from visit_media import portrait
    raw=np.zeros((1440,2560,3),np.uint8)
    raw[300:700,800:1200]=[0,255,0]
    image=portrait(raw,np.array([900,400,1100,600]))
    assert image.shape==(480,360,3)
    assert image[240,180].tolist()==[0,255,0]
