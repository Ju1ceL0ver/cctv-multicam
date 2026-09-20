import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
from day_visits import build_day, clip_nodes, decide
from review_store import read_state
from storage import atomic_json
from video_annotations import video_transact


@pytest.fixture
def day_project(tmp_path):
    root = tmp_path / 'project'
    for name, label, offset in [('c_a', 'P1', 0), ('c_b', 'P2', 10)]:
        d = root / 'data/raw_clips' / name; d.mkdir(parents=True)
        atomic_json(d / 'meta_yolo26x-seg.json', {'day': '20260918', 'seconds': 2,
            'start': f'2026-09-18T10:00:{offset:02d}', 'width': 100, 'height': 80})
        atomic_json(d / 'pieces_yolo26x-seg.json', [{'piece': 0, 'cam': 'cam1',
            't0': 0, 't1': 1.2, 'dets': [0, 1, 2]}])
        atomic_json(d / 'groups_yolo26x-seg.json', [{'person': 0, 'pieces': [0]}])
        atomic_json(d / 'gt_manual.json', {'0': label})
        rows = np.zeros((3, 10), np.float32); rows[:, 0] = [0, .6, 1.2]
        rows[:, 1:5] = [5, 5, 20, 40]
        np.savez(d / 'dets_yolo26x-seg.npz', cam1=rows, cam2=np.empty((0, 10)))
        np.savez(d / 'emb_osnet_ain_x1_0_msmt17.npz', cam1=np.array([[1., 0.]]*3), cam2=np.empty((0, 2)))
    return root


def nodes(root):
    return {n['key']: n for n in build_day('20260918', root)['nodes']}


def test_interval_splits_node_membership_without_reusing_entire_piece(day_project):
    d = day_project / 'data/raw_clips/c_a'
    video_transact(d, {'action': 'label_interval', 'piece': 0, 'start_frame': 15,
                      'end_frame': 30, 'label': 'P9'}, 0)
    by = nodes(day_project)
    assert by['c_a:P1']['observations'] == ['det:cam1:0']
    assert by['c_a:P9']['observations'] == ['det:cam1:1', 'det:cam1:2']
    assert by['c_a:P9']['reviewed']
    assert by['c_a:P9']['pieces'] == by['c_a:P1']['pieces'] == [0]
    assert by['c_a:P1']['last'] < by['c_a:P9']['first']


def test_manual_only_identity_has_node_without_fake_embedding(day_project):
    d = day_project / 'data/raw_clips/c_a'
    s = video_transact(d, {'action': 'upsert_observation', 'cam': 'cam1', 'frame': 5,
        'box': [30, 5, 50, 45], 'polygon': [[30, 5], [50, 5], [50, 45], [30, 45]], 'label': 'P8'}, 0)
    by = nodes(day_project)
    assert by['c_a:P8']['pieces'] == [] and by['c_a:P8']['reviewed']
    assert by['c_a:P8']['observations'] == [s['_last_observation_id']]
    assert by['c_a:P8']['representative']['frame'] == 5
    meta = json.loads((d / 'meta_yolo26x-seg.json').read_text())
    raw = {n['key']: n for n in clip_nodes(d, meta)}
    assert raw['c_a:P8']['_feat'] is None


def test_deleted_observation_and_changed_box_affect_visit_evidence(day_project):
    d = day_project / 'data/raw_clips/c_a'
    s = video_transact(d, {'action': 'delete_observation', 'id': 'det:cam1:1'}, 0)
    video_transact(d, {'action': 'upsert_observation', 'id': 'det:cam1:2', 'cam': 'cam1',
        'frame': 30, 'box': [10, 10, 30, 50]}, s['revision'])
    meta = json.loads((d / 'meta_yolo26x-seg.json').read_text())
    node = clip_nodes(d, meta)[0]
    assert node['observations'] == ['det:cam1:0', 'det:cam1:2']
    assert len(node['_obs']) == 2
    assert any(obs[2:] == (10, 10, 30, 50) for obs in node['_obs'])
    assert not any(obs[1] == round(node['first'] * 25)+15 for obs in node['_obs'])


def test_video_change_invalidates_old_human_day_link(day_project):
    day = build_day('20260918', day_project)
    a, b = 'c_a:P1', 'c_b:P2'
    by = {n['key']: n for n in day['nodes']}
    day = decide('20260918', {'a': a, 'b': b, 'decision': 'same', 'revision': 0,
        'evidence_a': by[a]['evidence'], 'evidence_b': by[b]['evidence']}, day_project)
    assert len(day['visits']) == 1
    d = day_project / 'data/raw_clips/c_a'
    video_transact(d, {'action': 'label_interval', 'piece': 0, 'start_frame': 15,
        'end_frame': 30, 'label': 'P9'}, 0)
    day = build_day('20260918', day_project)
    assert day['stale_decisions'] == 1
    assert not any(link['source'] == 'human' for link in day['links'])
    assert len(day['visits']) == 3


def test_explicit_arbitrary_pair_links_manual_person_without_descriptor(day_project):
    d = day_project / 'data/raw_clips/c_a'
    video_transact(d, {'action': 'upsert_observation', 'cam': 'cam1', 'frame': 5,
        'box': [30, 5, 50, 45], 'polygon': [[30, 5], [50, 5], [50, 45], [30, 45]], 'label': 'P8'}, 0)
    before = (d / 'review_state.json').read_bytes()
    day = build_day('20260918', day_project)
    by = {n['key']: n for n in day['nodes']}
    # The UI may choose either order; evidence stays attached to its original node.
    a, b = 'c_b:P2', 'c_a:P8'
    assert not any(b in (r['a'], r['b']) for r in day['proposals'])
    for decision, expected_visits in [('same', 2), ('different', 3), ('unsure', 3)]:
        day = decide('20260918', {'a': a, 'b': b, 'decision': decision,
            'revision': day['revision'], 'evidence_a': by[a]['evidence'],
            'evidence_b': by[b]['evidence']}, day_project)
        assert len(day['visits']) == expected_visits
        saved = json.loads((day_project / 'data/day_review/20260918.json').read_text())['links'][0]
        assert (saved['a'], saved['b']) == (b, a)
        assert saved['evidence_a'] == by[b]['evidence']
        assert saved['decision'] == decision
    # A day link never overwrites mask or identity annotation inside the source clip.
    assert (d / 'review_state.json').read_bytes() == before


def test_manual_pair_uses_current_evidence_and_refuses_stale_answer(day_project):
    d = day_project / 'data/raw_clips/c_a'
    s = video_transact(d, {'action': 'upsert_observation', 'cam': 'cam1', 'frame': 5,
        'box': [30, 5, 50, 45], 'label': 'P8'}, 0)
    by = nodes(day_project)
    video_transact(d, {'action': 'upsert_observation', 'id': s['_last_observation_id'],
        'cam': 'cam1', 'frame': 5, 'box': [31, 5, 51, 45], 'label': 'P8'}, s['revision'])
    with pytest.raises(ValueError, match='Отрезки изменились'):
        decide('20260918', {'a': 'c_a:P8', 'b': 'c_b:P2', 'decision': 'same',
            'revision': 0, 'evidence_a': by['c_a:P8']['evidence'],
            'evidence_b': by['c_b:P2']['evidence']}, day_project)
    assert not (day_project / 'data/day_review/20260918.json').exists()
