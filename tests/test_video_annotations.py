import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from storage import atomic_json
from review_store import Conflict, fingerprint, read_state, transact
from video_annotations import effective_decision, frame_data, video_transact


@pytest.fixture
def video_clip(tmp_path):
    d = tmp_path / 'c_video'; d.mkdir()
    atomic_json(d / 'meta_yolo26x-seg.json', {'day': '20260918', 'seconds': 1,
               'start': '2026-09-18T10:00:00', 'width': 100, 'height': 80})
    pieces = [{'piece': p, 'cam': 'cam1', 't0': 0, 't1': .24,
               'dets': list(range(p*3, p*3+3))} for p in range(3)]
    pieces.append({'piece': 3, 'cam': 'cam2', 't0': 0, 't1': .24, 'dets': [0, 1, 2]})
    atomic_json(d / 'pieces_yolo26x-seg.json', pieces)
    atomic_json(d / 'gt_manual.json', {'0': 'P1', '1': 'P2', '2': 'P3', '3': 'P4'})
    rows = np.zeros((9, 10), np.float32)
    rows[:, 0] = [0, .12, .24] * 3
    for p in range(3):
        rows[p*3:p*3+3, 1:5] = [5+p*25, 5, 20+p*25, 40]
    np.savez(d / 'dets_yolo26x-seg.npz', cam1=rows, cam2=rows[:3])
    polygons = []
    for row in rows:
        x1, y1, x2, y2 = row[1:5]
        polygons.extend([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    np.savez(d / 'polys_yolo26x-seg.npz', cam1_pts=polygons, cam1_off=np.arange(10)*4,
             cam2_pts=polygons[:12], cam2_off=np.arange(4)*4)
    np.savez(d / 'grid_yolo26x-seg.npz', t=np.array([0, .12, .24, .36]))
    return d


def add_person(frame=1, **extra):
    return {'action': 'upsert_observation', 'cam': 'cam1', 'frame': frame,
            'box': [75, 5, 95, 45], 'polygon': [[75, 5], [95, 5], [95, 45], [75, 45]],
            'label': 'NEW', **extra}


def test_exact_frame_and_empty_teacher_sample(video_clip):
    exact = frame_data(video_clip, 'cam1', 3)
    assert exact['width'] == 100 and exact['height'] == 80
    assert len(exact['observations']) == 3 and exact['teacher_sampled']
    assert {o['frame'] for o in exact['observations']} == {3}
    gap = frame_data(video_clip, 'cam1', 1)
    assert gap['observations'] == [] and not gap['teacher_sampled']
    empty = frame_data(video_clip, 'cam1', 9)
    assert empty['observations'] == [] and empty['teacher_sampled']
    assert not empty['frame_reviewed']


def test_missed_person_is_independent_anchor_and_undo(video_clip):
    original = fingerprint(read_state(video_clip))
    state = video_transact(video_clip, add_person(), 0)
    person = frame_data(video_clip, 'cam1', 1)['observations'][0]
    assert person['id'].startswith('manual:') and person['piece'] is None
    assert person['label'] == 'P5' and person['source'] == 'human_anchor'
    assert person['quality'] == 'unreviewed'
    assert fingerprint(state) != original
    restored = video_transact(video_clip, {'action': 'undo'}, state['revision'])
    assert not frame_data(video_clip, 'cam1', 1)['observations']
    assert len(restored['pieces']) == 4


def test_mask_correction_preserves_npz_and_invalidates_frame_review(video_clip):
    source = video_clip / 'dets_yolo26x-seg.npz'
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    s = video_transact(video_clip, {'action': 'review_frame', 'cam': 'cam1', 'frame': 3, 'reviewed': True}, 0)
    assert frame_data(video_clip, 'cam1', 3)['frame_reviewed']
    s = video_transact(video_clip, {'action': 'upsert_observation', 'id': 'det:cam1:1',
        'cam': 'cam1', 'frame': 3, 'box': [6, 5, 20, 40],
        'polygon': [[6, 5], [20, 5], [20, 40], [6, 40]], 'quality': 'valid'}, s['revision'])
    corrected = frame_data(video_clip, 'cam1', 3)
    assert not corrected['frame_reviewed']
    assert corrected['observations'][0]['quality'] == 'valid'
    assert corrected['observations'][0]['mask_review_source'] == 'human_anchor'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    video_transact(video_clip, {'action': 'undo'}, s['revision'])
    assert frame_data(video_clip, 'cam1', 3)['frame_reviewed']


def test_completeness_never_approves_mask_or_identity_and_tracks_new_objects(video_clip):
    s = video_transact(video_clip, add_person(label='?'), 0)
    s = video_transact(video_clip, {'action': 'review_frame', 'cam': 'cam1', 'frame': 1, 'reviewed': True}, s['revision'])
    data = frame_data(video_clip, 'cam1', 1)
    assert data['frame_reviewed']
    assert data['observations'][0]['quality'] == 'unreviewed'
    assert data['observations'][0]['label'] == '?'
    s = video_transact(video_clip, add_person(label='P8'), s['revision'])
    assert not frame_data(video_clip, 'cam1', 1)['frame_reviewed']


def test_interval_is_inclusive_and_materializes_exact_detection_ids(video_clip):
    s = video_transact(video_clip, {'action': 'label_interval', 'piece': 0,
        'start_frame': 3, 'end_frame': 6, 'label': 'P9'}, 0)
    assert frame_data(video_clip, 'cam1', 0)['observations'][0]['label'] == 'P1'
    assert frame_data(video_clip, 'cam1', 3)['observations'][0]['label'] == 'P9'
    assert frame_data(video_clip, 'cam1', 6)['observations'][0]['label'] == 'P9'
    assert s['labels']['0'] == 'P1'
    gt = json.loads((video_clip / 'gt_identity_yolo26x-seg.json').read_text())
    assert [gt['cam1'][str(i)] for i in range(3)] == ['P1', 'P9', 'P9']
    s = video_transact(video_clip, {'action': 'quality_interval', 'piece': 0,
        'start_frame': 3, 'end_frame': 3, 'quality': 'false_positive'}, s['revision'])
    gt = json.loads((video_clip / 'gt_identity_yolo26x-seg.json').read_text())
    assert '1' not in gt['cam1'] and gt['cam1']['2'] == 'P9'
    assert effective_decision(s, 0, 3, 'det:cam1:1') == ('P9', 'false_positive')


def test_swap_is_atomic_bounded_and_rejects_cross_camera(video_clip):
    s = video_transact(video_clip, {'action': 'swap_interval', 'piece': 0,
        'other_piece': 1, 'start_frame': 3, 'end_frame': 3}, 0)
    assert [o['label'] for o in frame_data(video_clip, 'cam1', 3)['observations'][:2]] == ['P2', 'P1']
    assert [o['label'] for o in frame_data(video_clip, 'cam1', 0)['observations'][:2]] == ['P1', 'P2']
    assert [o['label'] for o in frame_data(video_clip, 'cam1', 6)['observations'][:2]] == ['P1', 'P2']
    with pytest.raises(ValueError):
        video_transact(video_clip, {'action': 'swap_interval', 'piece': 0,
            'other_piece': 3, 'start_frame': 3, 'end_frame': 3}, s['revision'])
    assert read_state(video_clip)['revision'] == s['revision']
    video_transact(video_clip, {'action': 'undo'}, s['revision'])
    assert [o['label'] for o in frame_data(video_clip, 'cam1', 3)['observations'][:2]] == ['P1', 'P2']


def test_version_collision_and_invalid_batch_leave_no_partial_changes(video_clip):
    s = video_transact(video_clip, add_person(), 0)
    with pytest.raises(Conflict):
        video_transact(video_clip, add_person(frame=2), 0)
    before = (video_clip / 'review_state.json').read_bytes()
    with pytest.raises(ValueError):
        video_transact(video_clip, {'action': 'batch', 'operations': [add_person(frame=2),
            add_person(frame=3, box=[0, 0, float('nan'), 50])]}, s['revision'])
    assert (video_clip / 'review_state.json').read_bytes() == before
    assert not frame_data(video_clip, 'cam1', 2)['observations']


def test_batch_shares_new_identity_and_one_undo(video_clip):
    s = video_transact(video_clip, {'action': 'batch', 'operations': [add_person(frame=1), add_person(frame=2)]}, 0)
    assert frame_data(video_clip, 'cam1', 1)['observations'][0]['label'] == 'P5'
    assert frame_data(video_clip, 'cam1', 2)['observations'][0]['label'] == 'P5'
    assert s['revision'] == 1
    video_transact(video_clip, {'action': 'undo'}, 1)
    assert frame_data(video_clip, 'cam1', 1)['observations'] == []
    assert frame_data(video_clip, 'cam1', 2)['observations'] == []


def test_saved_different_blocks_legacy_merge_label_and_interval(video_clip):
    s = transact(video_clip, {'action': 'relation', 'a': 0, 'b': 1, 'decision': 'different'}, 0)
    for operation in [
        {'action': 'merge', 'from': 'P2', 'to': 'P1'},
        {'action': 'label', 'pieces': [1], 'label': 'P1'},
        {'action': 'label_interval', 'piece': 1, 'start_frame': 3, 'end_frame': 3, 'label': 'P1'},
    ]:
        with pytest.raises(ValueError, match='разные'):
            transact(video_clip, operation, s['revision'])
        assert read_state(video_clip)['revision'] == s['revision']


def test_transitive_contradiction_and_same_relation_interval_are_rejected(video_clip):
    s = transact(video_clip, {'action': 'relation', 'a': 0, 'b': 2, 'decision': 'different'}, 0)
    s = transact(video_clip, {'action': 'relation', 'a': 0, 'b': 1, 'decision': 'same'}, s['revision'])
    with pytest.raises(ValueError):
        transact(video_clip, {'action': 'relation', 'a': 1, 'b': 2, 'decision': 'same'}, s['revision'])
    with pytest.raises(ValueError):
        transact(video_clip, {'action': 'label_interval', 'piece': 0, 'start_frame': 3, 'end_frame': 6, 'label': 'P9'}, s['revision'])
    assert read_state(video_clip)['revision'] == s['revision']


def test_manual_membership_intervals_override_and_explicit_anchor_wins_later(video_clip):
    s = video_transact(video_clip, add_person(piece=0, label='P1'), 0)
    oid = s['_last_observation_id']
    s = video_transact(video_clip, {'action': 'label_interval', 'piece': 0, 'start_frame': 0, 'end_frame': 3, 'label': 'P8'}, s['revision'])
    assert frame_data(video_clip, 'cam1', 1)['observations'][0]['label'] == 'P8'
    s = video_transact(video_clip, add_person(id=oid, label='P9'), s['revision'])
    assert frame_data(video_clip, 'cam1', 1)['observations'][0]['label'] == 'P9'
    assert effective_decision(s, 0, 1, oid)[0] == 'P9'
    s = transact(video_clip, {'action': 'label', 'pieces': [0], 'label': 'P7'}, s['revision'])
    assert frame_data(video_clip, 'cam1', 1)['observations'][0]['label'] == 'P7'


def test_rle_propagation_cannot_be_silently_approved(video_clip):
    op = add_person(mask_rle={'size': [80, 100], 'counts': [10, 5, 7985]},
                    source='sam_propagation', quality='valid')
    with pytest.raises(ValueError, match='явного'):
        video_transact(video_clip, op, 0)
    op['approval'] = 'accepted_propagation'
    op['provenance'] = {'job_id': 'pilot', 'model': 'sam2.1_l', 'anchor_id': 'a'}
    s = video_transact(video_clip, op, 0)
    person = frame_data(video_clip, 'cam1', 1)['observations'][0]
    assert person['mask_review_source'] == 'human_accepted_propagation'
    assert not frame_data(video_clip, 'cam1', 1)['frame_reviewed']
    bad = add_person(frame=2, mask_rle={'size': [80, 100], 'counts': [1, 2]})
    with pytest.raises(ValueError, match='RLE'):
        video_transact(video_clip, bad, s['revision'])


def test_observation_id_cannot_move_and_delete_is_undoable(video_clip):
    with pytest.raises(ValueError, match='перенести'):
        video_transact(video_clip, add_person(id='det:cam1:1'), 0)
    s = video_transact(video_clip, {'action': 'delete_observation', 'id': 'det:cam1:1'}, 0)
    assert len(frame_data(video_clip, 'cam1', 3)['observations']) == 2
    gt = json.loads((video_clip / 'gt_identity_yolo26x-seg.json').read_text())
    assert '1' not in gt['cam1']
    video_transact(video_clip, {'action': 'undo'}, s['revision'])
    assert len(frame_data(video_clip, 'cam1', 3)['observations']) == 3


def test_cached_dimensions_and_input_bounds(video_clip):
    atomic_json(video_clip / 'review_media/source.json', {'dimensions': {'cam1': [120, 90]}})
    assert frame_data(video_clip, 'cam1', 0)['width'] == 120
    for operation in [add_person(frame=25), add_person(box=[-1, 0, 5, 5]),
                      add_person(polygon=[[1, 1], [2, 2], [3, 3]])]:
        with pytest.raises(ValueError):
            video_transact(video_clip, operation, 0)
