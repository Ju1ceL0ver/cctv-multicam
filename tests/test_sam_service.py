"""CPU-only tests for proposal integrity and scheduling; no fake GPU accuracy."""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import sam_service as service
import sam_worker as worker
from storage import atomic_json, read_json


@pytest.fixture
def clip(tmp_path, monkeypatch):
    folder = tmp_path / 'c20260917_100000'
    atomic_json(folder / 'meta_yolo26x-seg.json', {'day': '20260917', 'start': '2026-09-17T10:00:00', 'seconds': 30})
    atomic_json(folder / 'review_media' / 'source.json', {'dimensions': {'cam1': [64, 48], 'cam2': [80, 60]}})
    monkeypatch.setattr(service, '_launch_worker', lambda *args: None)
    return folder


def request(**changes):
    result = {'cam': 'cam1', 'frame': 97, 'end_frame': 101, 'box': [10, 5, 40, 45], 'review_revision': 3}
    result.update(changes)
    return result


def test_enqueue_is_a_proposal_and_preserves_review_revision(clip):
    human = {'revision': 3, 'labels': {'9': 'P2'}}
    atomic_json(clip / 'review_state.json', human)
    job = service.enqueue(clip, request())
    assert job['status'] == 'queued'
    assert job['source'] == 'sam2.1_small'
    assert job['request']['dimensions'] == [64, 48]
    assert job['review_revision'] == 3 and job['frame_count'] == 5
    assert read_json(clip / 'review_state.json') == human
    assert not (clip / 'sam_jobs' / job['id'] / 'proposals.json').exists()


@pytest.mark.parametrize('change', [
    {'end_frame': 347}, {'end_frame': 96}, {'frame': -1}, {'frame': 1.5},
    {'frame': True}, {'end_frame': 750}, {'cam': 'cam3'}, {'model': 'anything.pt'},
    {'box': [1, 2, 65, 40]}, {'box': [1, 2, float('nan'), 40]},
    {'box': None}, {'review_revision': -1}, {'review_revision': '3'},
    {'box': None, 'points': [[20, 20]], 'point_labels': [0]},
    {'box': None, 'points': [[20, 20]], 'point_labels': [True]},
    {'points': [[64, 1]], 'point_labels': [1]},
])
def test_invalid_requests_never_spawn_or_save(clip, change):
    with pytest.raises(ValueError):
        service.enqueue(clip, request(**change))
    assert not list((clip / 'sam_jobs').glob('*/job.json'))


def test_inclusive_frame_cap_and_explicit_large(clip):
    job = service.enqueue(clip, request(end_frame=346, model='large'))
    assert job['frame_count'] == 250
    assert job['source'] == 'sam2.1_large'
    assert job['request']['model'] == 'large'


def test_points_belong_to_one_object(clip):
    normalized = service.validate_request(clip, request(box=None, points=[[20, 10], [30, 30]], point_labels=[1, 0]))
    prompts = worker.prompt_kwargs(normalized)
    assert np.asarray(prompts['points']).shape == (1, 2, 2)
    assert prompts['labels'] == [[1, 0]]
    assert 'bboxes' not in prompts


def test_native_dimensions_are_not_assumed_to_be_half_size(clip):
    normalized = service.validate_request(clip, request(cam='cam2', box=[1, 2, 79, 59]))
    assert normalized['dimensions'] == [80, 60]
    assert worker.prompt_kwargs(normalized)['bboxes'] == [[1, 2, 79, 59]]


def test_cancel_prevents_late_ready_result(clip):
    job = service.enqueue(clip, request())
    service.cancel_job(clip, job['id'])
    service.update_job(clip, job['id'], status='ready')
    assert service.get_job(clip, job['id'])['status'] == 'cancelled'


def test_gpu_window_respects_utc7_and_exact_boundaries():
    for hour, minute, allowed in [(9, 44, True), (9, 45, False), (20, 59, False), (21, 0, True)]:
        assert service.night_window(datetime(2026, 9, 19, hour, minute)) == allowed
    assert not service.night_window(datetime(2026, 9, 19, 2, 45, tzinfo=timezone.utc))
    assert service.night_window(datetime(2026, 9, 19, 14, 0, tzinfo=timezone.utc))


def test_lossless_rle_keeps_holes_and_separate_components(clip):
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[5:20, 7:30] = 1
    mask[10:15, 12:17] = 0
    mask[35:38, 50:54] = 1
    proposal = worker.mask_proposal(mask, service.validate_request(clip, request()), 99)
    rle = proposal['mask_rle']
    restored = np.repeat(np.arange(len(rle['counts'])) % 2, rle['counts']).reshape(rle['size'], order='F')
    assert np.array_equal(restored, mask)
    assert proposal['box'] == [7, 5, 54, 38]
    assert proposal['frame'] == 99 and proposal['cam'] == 'cam1'
    assert sum(rle['counts']) == 48 * 64


def test_empty_mask_is_absence_not_a_fabricated_box(clip):
    proposal = worker.mask_proposal(np.zeros((48, 64), np.uint8), service.validate_request(clip, request()), 99)
    assert proposal['box'] is None and proposal['visible'] is False
    assert proposal['polygon'] == [] and proposal['mask_rle']['counts'] == [48 * 64]


def test_prepare_video_preserves_anchor_and_every_frame(clip, monkeypatch, tmp_path):
    import rawsource
    base = datetime(2026, 9, 17, 10)
    seeks = []
    class FakeStream:
        cap = None
        def __init__(self, cam, day):
            assert cam == 'cam1' and day == '20260917'
        def seek(self, instant):
            seeks.append(instant)
            self.index = round((instant - base).total_seconds() * 25)
        def read(self):
            index = self.index
            self.index += 1
            return base + timedelta(seconds=index / 25), np.full((48, 64, 3), index, np.uint8)
    monkeypatch.setattr(rawsource, 'Stream', FakeStream)
    normalized = service.validate_request(clip, request())
    output = tmp_path / 'exact.avi'
    assert worker.prepare_video(clip, normalized, output) == 5
    assert seeks == [base + timedelta(seconds=97/25)]
    video = cv2.VideoCapture(str(output))
    decoded = []
    while True:
        ok, frame = video.read()
        if not ok:
            break
        decoded.append(int(frame[0, 0, 0]))
    video.release()
    assert decoded == [97, 98, 99, 100, 101]


def test_prepare_video_rejects_gap_instead_of_silent_reindex(clip, monkeypatch, tmp_path):
    import rawsource
    class GapStream:
        cap = None
        def __init__(self, *args): pass
        def seek(self, instant): self.instant = instant
        def read(self): return self.instant + timedelta(seconds=1), np.zeros((48, 64, 3), np.uint8)
    monkeypatch.setattr(rawsource, 'Stream', GapStream)
    with pytest.raises(RuntimeError, match='Разрыв'):
        worker.prepare_video(clip, service.validate_request(clip, request()), tmp_path / 'gap.avi')


def test_supervisor_cancels_only_its_child(clip, monkeypatch):
    job = service.enqueue(clip, request())
    class Child:
        returncode = None
        terminated = False
        def poll(self): return self.returncode
        def terminate(self): self.terminated = True; self.returncode = 1
        def wait(self, timeout): return self.returncode
        def kill(self): pytest.fail('Child should terminate normally')
    child = Child()
    checks = []
    def check(*args):
        checks.append(True)
        if len(checks) > 1:
            raise worker.Cancelled()
    monkeypatch.setattr(worker.subprocess, 'Popen', lambda *args, **kwargs: child)
    monkeypatch.setattr(worker, '_check_abort', check)
    worker._run_child(clip, job['id'])
    assert child.terminated
    assert service.get_job(clip, job['id'])['status'] == 'cancelled'


def test_ready_results_expose_exact_frames_and_request_revision(clip):
    job = service.enqueue(clip, request())
    frames = [{'cam': 'cam1', 'frame': 97, 'source': 'sam2.1_small'}]
    atomic_json(service.job_dir(clip, job['id']) / 'proposals.json', {'frames': frames})
    service.update_job(clip, job['id'], status='ready')
    loaded = service.get_job(clip, job['id'])
    assert loaded['frames'] == frames and loaded['review_revision'] == 3
    with pytest.raises(ValueError):
        service.get_job(clip, '../../secret')


def test_dead_queue_worker_is_reported_instead_of_waiting_forever(clip):
    job = service.enqueue(clip, request())
    path = service.job_dir(clip, job['id']) / 'job.json'
    job['updated_at'] -= 181
    atomic_json(path, job)
    assert service.get_job(clip, job['id'])['status'] == 'failed'


def test_active_queue_is_bounded_but_cancelled_jobs_free_capacity(clip):
    jobs = [service.enqueue(clip, request()) for _ in range(4)]
    with pytest.raises(ValueError, match='четыре'):
        service.enqueue(clip, request())
    service.cancel_job(clip, jobs[0]['id'])
    assert service.enqueue(clip, request())['status'] == 'queued'


def test_one_sam_job_accepts_successive_frames_at_new_revisions(clip):
    job = service.enqueue(clip, request())
    service.update_job(clip, job['id'], status='ready')
    first = service.record_acceptance(clip, job['id'], 4, [97])
    second = service.record_acceptance(clip, job['id'], 5, [98, 99])
    assert second['accepted_revision'] == 5
    assert second['accepted_frames'] == [97, 98, 99]
    assert second['review_revision'] == 3
    with pytest.raises(ValueError):
        service.record_acceptance(clip, job['id'], 6, [102])
    with pytest.raises(ValueError):
        service.record_acceptance(clip, job['id'], 5, [100])
    assert service.job_path(clip, job['id']) == clip / 'sam_jobs' / job['id']


def _teacher_scene(clip):
    rows = np.zeros((2, 10), np.float32)
    rows[:, 0] = [0, .12]
    rows[:, 1:5] = [10, 5, 40, 45]
    np.savez(clip / 'dets_yolo26x-seg.npz', cam1=rows, cam2=np.empty((0, 10)))
    poly = np.array([[10, 5], [40, 5], [40, 45], [10, 45]] * 2)
    np.savez(clip / 'polys_yolo26x-seg.npz', cam1_pts=poly, cam1_off=[0, 4, 8],
             cam2_pts=np.empty((0, 2)), cam2_off=[0])
    atomic_json(clip / 'pieces_yolo26x-seg.json', [{'piece': 0, 'cam': 'cam1', 'dets': [0, 1], 't0': 0, 't1': .12}])
    atomic_json(clip / 'gt_manual.json', {'0': 'P1'})


def _ready_scene_job(clip, revision):
    job = service.enqueue(clip, request(frame=0, end_frame=3, review_revision=revision))
    proposals = [{'cam': 'cam1', 'frame': f, 'visible': True, 'box': [10, 5, 40, 45],
                  'polygon': [[10, 5], [40, 5], [40, 45], [10, 45]]} for f in (0, 3)]
    atomic_json(service.job_dir(clip, job['id']) / 'proposals.json', {'frames': proposals})
    service.update_job(clip, job['id'], status='ready')
    return job['id']


@pytest.mark.parametrize('decision', ['false_positive', 'invalid', 'mixed'])
def test_sam_batch_preserves_explicit_negative_quality(clip, decision):
    from video_api import accept_sam_proposals
    from video_annotations import frame_data, video_transact
    _teacher_scene(clip)
    state = video_transact(clip, {'action': 'quality_interval', 'piece': 0, 'start_frame': 3,
                                 'end_frame': 3, 'quality': decision}, 0)
    job_id = _ready_scene_job(clip, state['revision'])
    accept_sam_proposals(clip, job_id, {'revision': state['revision'], 'frames': [0, 3],
        'confirm_masks': True, 'piece': 0, 'anchor_id': 'det:cam1:0'})
    assert frame_data(clip, 'cam1', 3)['observations'][0]['quality'] == decision


def test_sam_batch_does_not_resurrect_a_deleted_detection(clip):
    from video_api import accept_sam_proposals
    from video_annotations import frame_data, video_transact
    _teacher_scene(clip)
    state = video_transact(clip, {'action': 'delete_observation', 'id': 'det:cam1:1'}, 0)
    job_id = _ready_scene_job(clip, state['revision'])
    accept_sam_proposals(clip, job_id, {'revision': state['revision'], 'frames': [0, 3],
        'confirm_masks': True, 'piece': 0, 'anchor_id': 'det:cam1:0'})
    assert frame_data(clip, 'cam1', 3)['observations'] == []


def test_sam_mask_batch_preserves_existing_identity_interval(clip):
    from video_api import accept_sam_proposals
    from video_annotations import frame_data, video_transact
    _teacher_scene(clip)
    state = video_transact(clip, {'action': 'label_interval', 'piece': 0, 'start_frame': 3,
                                 'end_frame': 3, 'label': 'P2'}, 0)
    job_id = _ready_scene_job(clip, state['revision'])
    accept_sam_proposals(clip, job_id, {'revision': state['revision'], 'frames': [0, 3],
        'confirm_masks': True, 'piece': 0, 'anchor_id': 'det:cam1:0', 'label': 'P1'})
    assert frame_data(clip, 'cam1', 3)['observations'][0]['label'] == 'P2'


def test_segmentation_export_drops_only_frames_after_a_raw_recording_gap(clip, tmp_path, monkeypatch):
    """A skip in one camera's recording used to abort the whole export, and with it the
    night's training. Now the frames from the gap on are dropped and reported; frames before
    it, and every other clip, still go into the dataset."""
    import export_seg_dataset as exporter
    _teacher_scene(clip)
    old_manifest = exporter.source_manifest
    monkeypatch.setattr(exporter, 'ROOT', tmp_path / 'export_root')
    monkeypatch.setattr(exporter, 'CLIPS', clip.parent)
    monkeypatch.setattr(exporter, 'source_manifest', lambda **kw: old_manifest(clips=clip.parent, **kw))
    class GapStream:
        cap = None
        def __init__(self, *args): self.k = 0
        def seek(self, instant): self.start = instant
        def read(self):
            k = self.k
            self.k += 1
            # Recording jumps by one second after its first frame.
            instant = self.start + timedelta(seconds=k / 25 + (1 if k else 0))
            return instant, np.zeros((48, 64, 3), np.uint8)
    monkeypatch.setattr(exporter, 'Stream', GapStream)
    out, manifest = exporter.export(gap=0, val=(), destination=tmp_path / 'dataset')
    assert manifest['complete'] and manifest['counts']['train']['frames'] == 1
    assert sorted(f.name for f in (out / 'train' / 'images').glob('*.jpg')) == ['c20260917_100000_cam1_000000.jpg']
    detail = next(d for d in manifest['details'] if d['cam'] == 'cam1')
    assert detail['raw_gap'] == {'frame': 1, 'offset_s': 1.0, 'dropped_frames': 1}


def test_segmentation_export_still_stops_when_the_raw_video_is_missing(clip, tmp_path, monkeypatch):
    """Missing footage is not a skip in the recording: it is a fault to notice, not to hide."""
    import export_seg_dataset as exporter
    _teacher_scene(clip)
    old_manifest = exporter.source_manifest
    monkeypatch.setattr(exporter, 'ROOT', tmp_path / 'export_root')
    monkeypatch.setattr(exporter, 'CLIPS', clip.parent)
    monkeypatch.setattr(exporter, 'source_manifest', lambda **kw: old_manifest(clips=clip.parent, **kw))
    class EmptyStream:
        cap = None
        def __init__(self, *args): pass
        def seek(self, instant): pass
        def read(self): return None, None
    monkeypatch.setattr(exporter, 'Stream', EmptyStream)
    with pytest.raises(RuntimeError, match='ended early'):
        exporter.export(gap=0, val=(), destination=tmp_path / 'dataset')


def test_sam_batch_matches_saved_manual_anchor_without_creating_duplicate(clip):
    from video_api import accept_sam_proposals
    from video_annotations import frame_data, video_transact
    _teacher_scene(clip)
    box = [45, 5, 61, 45]
    polygon = [[45, 5], [61, 5], [61, 45], [45, 45]]
    state = video_transact(clip, {'action': 'upsert_observation', 'cam': 'cam1', 'frame': 0,
                                'box': box, 'polygon': polygon, 'label': 'P2'}, 0)
    anchor = state['_last_observation_id']
    job_id = _ready_scene_job(clip, state['revision'])
    atomic_json(service.job_dir(clip, job_id) / 'proposals.json', {'frames': [
        {'cam': 'cam1', 'frame': f, 'visible': True, 'box': box, 'polygon': polygon} for f in (0, 3)]})
    accept_sam_proposals(clip, job_id, {'revision': state['revision'], 'frames': [0, 3],
        'confirm_masks': True, 'anchor_id': anchor, 'label': 'P2'})
    anchor_people = [o for o in frame_data(clip, 'cam1', 0)['observations'] if o['label'] == 'P2']
    assert len(anchor_people) == 1 and anchor_people[0]['id'] == anchor
    assert len(frame_data(clip, 'cam1', 3)['observations']) == 2


def test_new_identity_is_reused_across_sequential_sam_acceptance(clip):
    from video_api import accept_sam_proposals
    from video_annotations import frame_data
    _teacher_scene(clip)
    job_id = _ready_scene_job(clip, 0)
    box = [45, 5, 61, 45]
    polygon = [[45, 5], [61, 5], [61, 45], [45, 45]]
    atomic_json(service.job_dir(clip, job_id) / 'proposals.json', {'frames': [
        {'cam': 'cam1', 'frame': f, 'visible': True, 'box': box, 'polygon': polygon} for f in (0, 3)]})
    first = accept_sam_proposals(clip, job_id, {'revision': 0, 'frames': [0], 'confirm_masks': True, 'label': 'NEW'})
    second = accept_sam_proposals(clip, job_id, {'revision': first['revision'], 'frames': [3], 'confirm_masks': True, 'label': 'NEW'})
    a = next(o for o in frame_data(clip, 'cam1', 0)['observations'] if o['id'].startswith('manual:'))
    b = next(o for o in frame_data(clip, 'cam1', 3)['observations'] if o['id'].startswith('manual:'))
    assert a['label'] == b['label']


def test_resolved_new_label_is_recorded_with_acceptance_and_preserved(clip):
    job = service.enqueue(clip, request())
    service.update_job(clip, job['id'], status='ready')
    first = service.record_acceptance(clip, job['id'], 4, [97], accepted_label='P2')
    second = service.record_acceptance(clip, job['id'], 5, [98])
    assert first['accepted_label'] == second['accepted_label'] == 'P2'
    assert second['accepted_frames'] == [97, 98]
    with pytest.raises(ValueError):
        service.record_acceptance(clip, job['id'], 6, [99], accepted_label='NEW')


def test_sam_extension_does_not_duplicate_a_different_existing_person(clip):
    from video_api import accept_sam_proposals
    from video_annotations import frame_data
    _teacher_scene(clip)
    atomic_json(clip / 'pieces_yolo26x-seg.json', [
        {'piece': 0, 'cam': 'cam1', 'dets': [0], 't0': 0, 't1': 0},
        {'piece': 1, 'cam': 'cam1', 'dets': [1], 't0': .12, 't1': .12}])
    atomic_json(clip / 'gt_manual.json', {'0': 'P1', '1': 'P2'})
    job_id = _ready_scene_job(clip, 0)
    try:
        accept_sam_proposals(clip, job_id, {'revision': 0, 'frames': [0, 3],
            'confirm_masks': True, 'piece': 0, 'anchor_id': 'det:cam1:0', 'label': 'P1'})
    except ValueError:
        pass  # Requiring an explicit choice is safe; silent duplicate creation is not.
    people = frame_data(clip, 'cam1', 3)['observations']
    assert len(people) == 1 and people[0]['label'] == 'P2'


def test_manifest_excludes_overlapping_validation_source_intervals(tmp_path):
    import export_seg_dataset as exporter
    origin = datetime(2026, 9, 17, 10)
    cases = {
        'c_validation': (0, 60),
        'c_same_raw_another_name': (0, 60),
        'c_inside': (20, 10),
        'c_enclosing': (-30, 120),
        'c_cross_start': (-10, 20),
        'c_cross_end': (50, 20),
        'c_touch_before': (-30, 30),
        'c_touch_after': (60, 30),
        'c_same_clock_next_day': (24*3600, 60),
    }
    for name, (offset, duration) in cases.items():
        folder = tmp_path / name
        start = origin + timedelta(seconds=offset)
        atomic_json(folder / 'meta_yolo26x-seg.json', {
            'day': start.strftime('%Y%m%d'), 'start': start.isoformat(), 'seconds': duration})
        _teacher_scene(folder)
    manifest = exporter.source_manifest(clips=tmp_path, val=('c_validation',))
    selected = {s['clip']: s['split'] for s in manifest['sources']}
    assert selected == {
        'c_validation': 'val', 'c_touch_before': 'train',
        'c_touch_after': 'train', 'c_same_clock_next_day': 'train'}
