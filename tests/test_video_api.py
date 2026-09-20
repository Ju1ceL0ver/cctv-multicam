import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from storage import atomic_json
from review_store import read_state, transact, Conflict
from video_annotations import video_transact, frame_data
from video_api import accept_sam_proposals


@pytest.fixture
def scene(tmp_path):
    d=tmp_path/'c_demo';d.mkdir()
    rows=np.array([[0,1,1,8,18,.9,4,18,4,1],[.12,2,1,9,18,.9,5,18,5,1]],dtype=np.float32)
    np.savez(d/'dets_yolo26x-seg.npz',cam1=rows,cam2=np.empty((0,10)))
    poly=np.array([[1,1],[8,1],[8,18],[1,18],[2,1],[9,1],[9,18],[2,18]],float)
    np.savez(d/'polys_yolo26x-seg.npz',cam1_pts=poly,cam1_off=[0,4,8],cam2_pts=np.empty((0,2)),cam2_off=[0])
    atomic_json(d/'meta_yolo26x-seg.json',{'day':'20260917','start':'2026-09-17T10:00:00','seconds':1,'width':20,'height':20})
    atomic_json(d/'review_media/source.json',{'dimensions':{'cam1':[20,20],'cam2':[20,20]}})
    atomic_json(d/'pieces_yolo26x-seg.json',[{'piece':0,'cam':'cam1','dets':[0,1],'t0':0,'t1':.12}])
    atomic_json(d/'gt_manual.json',{'0':'P1'})
    return d


def job(scene, revision=0):
    ident='a'*32;d=scene/'sam_jobs'/ident
    atomic_json(d/'job.json',{'id':ident,'status':'ready','review_revision':revision,'request':{'cam':'cam1','model':'small','frame':0,'end_frame':3},'accepted_frames':[]})
    atomic_json(d/'proposals.json',{'frames':[{'cam':'cam1','frame':f,'visible':True,'box':[2,2,8,18],'polygon':[[2,2],[8,2],[8,18],[2,18]]} for f in (0,3)]})
    return ident


def test_same_sam_job_accepts_sequential_frames_without_rerun(scene):
    jid=job(scene)
    first=accept_sam_proposals(scene,jid,{'revision':0,'frames':[0],'confirm_masks':True,'piece':0,'anchor_id':'det:cam1:0','label':'P1'})
    second=accept_sam_proposals(scene,jid,{'revision':first['revision'],'frames':[3],'confirm_masks':True,'piece':0,'anchor_id':'det:cam1:0','label':'P1'})
    assert second['revision']==2 and second['remaining_frames']==[]
    assert len(read_state(scene)['video']['observations'])==2
    assert frame_data(scene,'cam1',0)['observations'][0]['mask_review_source']=='human_accepted_propagation'


def test_sam_never_overwrites_another_verified_anchor(scene):
    s=video_transact(scene,{'action':'upsert_observation','cam':'cam1','frame':3,'id':'det:cam1:1','box':[2,2,8,18],'polygon':[[2,2],[8,2],[8,18],[2,18]],'quality':'valid'},0)
    jid=job(scene,s['revision'])
    result=accept_sam_proposals(scene,jid,{'revision':s['revision'],'frames':[0,3],'confirm_masks':True,'piece':0,'anchor_id':'det:cam1:0','label':'P1'})
    assert result['accepted_frame_indices']==[0] and result['protected_frames']==[3]
    assert frame_data(scene,'cam1',3)['observations'][0]['source']=='human_anchor'


def test_sam_rejects_external_intervening_edit(scene):
    jid=job(scene)
    transact(scene,{'pieces':[0],'label':'P2'},0)
    with pytest.raises(Conflict):
        accept_sam_proposals(scene,jid,{'revision':1,'frames':[0],'confirm_masks':True,'piece':0,'label':'P1'})
    assert not read_state(scene).get('video')


def test_sam_requires_explicit_review_and_prevents_duplicate_person(scene):
    jid=job(scene)
    with pytest.raises(ValueError,match='Подтвердите'):
        accept_sam_proposals(scene,jid,{'revision':0,'frames':[0],'piece':0})
    with pytest.raises(ValueError,match='дубль'):
        accept_sam_proposals(scene,jid,{'revision':0,'frames':[0],'confirm_masks':True})
    assert read_state(scene)['revision']==0


def test_corrected_training_frame_contains_added_person_only_after_completeness(scene):
    from curated_export import select_corrected_frames
    s=video_transact(scene,{'action':'upsert_observation','cam':'cam1','frame':0,'box':[12,2,18,18],'polygon':[[12,2],[18,2],[18,18],[12,18]],'quality':'valid','label':'P2'},0)
    with np.load(scene/'dets_yolo26x-seg.npz') as z: rows=z['cam1'].copy()
    wanted,_,_=select_corrected_frames(scene,'cam1',rows,s,gap=0)
    assert 0 not in wanted
    s=video_transact(scene,{'action':'review_frame','cam':'cam1','frame':0,'reviewed':True},s['revision'])
    wanted,_,_=select_corrected_frames(scene,'cam1',rows,s,gap=0)
    assert len(wanted[0])==2 and any(o['source']=='human_anchor' for o in wanted[0])


def test_http_requires_revision_and_known_labels_include_interval_ids(scene,monkeypatch):
    import label_pieces as web
    monkeypatch.setattr(web,'CLIPS',str(scene.parent))
    client=web.app.test_client();client.set_cookie('labeler_key',web.KEY)
    route='/api/video/c_demo'
    assert client.post(route+'/edit',json={'action':'label_interval','piece':0,'start_frame':0,'end_frame':3,'label':'NEW'}).status_code==400
    r=client.post(route+'/edit',json={'action':'label_interval','piece':0,'start_frame':0,'end_frame':3,'label':'NEW','revision':0})
    assert r.status_code==200
    assert 'P2' in client.get(route+'/info').json['known_labels']
    assert client.post(route+'/edit',json={'action':'label_interval','piece':0,'start_frame':0,'end_frame':3,'label':'P3','revision':0}).status_code==409
    assert client.get(route+'/frame?cam=cam1&frame=1').json['teacher_sampled'] is False
    assert client.get(route+'/frame?cam=cam3&frame=1').status_code==400


def test_sam_mask_only_correction_preserves_identity(scene):
    jid=job(scene)
    accept_sam_proposals(scene,jid,{'revision':0,'frames':[0],'confirm_masks':True,'piece':0,'anchor_id':'det:cam1:0'})
    assert frame_data(scene,'cam1',0)['observations'][0]['label']=='P1'


def test_reading_more_frames_does_not_invalidate_dimension_metadata(scene):
    import json
    from video_media import remember_dimensions
    image=np.zeros((20,20,3),dtype=np.uint8)
    remember_dimensions(scene,'cam1',image)
    path=scene/'review_media/source.json'
    before=(path.stat().st_mtime_ns,path.read_bytes())
    remember_dimensions(scene,'cam1',image)
    assert (path.stat().st_mtime_ns,path.read_bytes())==before
    remember_dimensions(scene,'cam2',image)
    assert 'cam2' in json.loads(path.read_text())['dimensions']


def test_preview_and_native_crop_preserve_original_and_reject_invalid_roi(tmp_path,monkeypatch):
    import cv2
    import video_media
    from storage import atomic_json
    native=np.zeros((1440,2560,3),dtype=np.uint8);native[300:600,400:700]=[40,100,200]
    image=tmp_path/'review_media/frames/native.jpg';image.parent.mkdir(parents=True)
    cv2.imwrite(str(image),native)
    atomic_json(tmp_path/'review_media/source.json',{'dimensions':{'cam1':[2560,1440]}})
    identity={'cam':'cam1','clip_frame':3,'frame_index':103}
    monkeypatch.setattr(video_media,'frame_image',lambda *args:(image,identity))
    original=image.read_bytes()
    preview,info=video_media.display_image(tmp_path,'cam1',3,preview=True)
    assert cv2.imread(str(preview)).shape[:2]==(540,960)
    crop,info=video_media.display_image(tmp_path,'cam1',3,roi=[400,300,700,600])
    assert cv2.imread(str(crop)).shape[:2]==(300,300) and info==identity
    assert image.read_bytes()==original
    assert video_media.display_image(tmp_path,'cam1',3)[0]==image
    assert preview==video_media.display_image(tmp_path,'cam1',3,preview=True)[0]
    for roi in ([0,0,2600,100],[-1,0,100,100],[0,0,0,0],[0,0,1.5,3],[1,2,3]):
        with pytest.raises(ValueError):video_media.display_image(tmp_path,'cam1',3,roi=roi)


def test_http_gzip_export_download_matches_plain_snapshot(scene,monkeypatch):
    import gzip
    import label_pieces as web
    monkeypatch.setattr(web,'CLIPS',str(scene.parent))
    client=web.app.test_client();client.set_cookie('labeler_key',web.KEY)
    plain=client.get('/api/video/c_demo/export')
    packed=client.get('/api/video/c_demo/export?compress=gzip')
    assert plain.status_code==packed.status_code==200
    assert gzip.decompress(packed.data)==plain.data
    assert packed.mimetype=='application/gzip'
    assert 'segmentation-tracks.ndjson.gz' in packed.headers['Content-Disposition']
