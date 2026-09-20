import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import product_export
from product_export import export_clip
from review_store import Conflict, read_state
from storage import atomic_json
from video_annotations import video_transact


@pytest.fixture
def export_source(tmp_path):
    d = tmp_path / 'data' / 'raw_clips' / 'c_export'; d.mkdir(parents=True)
    atomic_json(d / 'meta_yolo26x-seg.json', {'day': '20260919', 'start': '2026-09-19T10:00:00',
               'seconds': 1, 'width': 100, 'height': 80})
    rows = np.zeros((4, 10), np.float32)
    rows[:, 0] = [0, .12, .24, .36]
    rows[:, 1:5] = [10, 10, 40, 60]
    np.savez(d / 'dets_yolo26x-seg.npz', cam1=rows, cam2=rows[:2])
    poly = [[10, 10], [40, 10], [40, 60], [10, 60]]
    np.savez(d / 'polys_yolo26x-seg.npz', cam1_pts=poly*4, cam1_off=[0,4,8,12,16],
             cam2_pts=poly*2, cam2_off=[0,4,8])
    np.savez(d / 'grid_yolo26x-seg.npz', t=np.array([0, .12, .24, .36, .48]))
    atomic_json(d / 'pieces_yolo26x-seg.json', [
        {'piece': 0, 'cam': 'cam1', 'dets': [0,1,2,3], 't0': 0, 't1': .36},
        {'piece': 1, 'cam': 'cam2', 'dets': [0,1], 't0': 0, 't1': .12}])
    atomic_json(d / 'groups_yolo26x-seg.json', [{'person': 7, 'pieces': [0,1]}])
    atomic_json(d / 'gt_manual.json', {'0': 'P1'})
    atomic_json(d / 'sync_estimate.json', {'status': 'accepted', 'cam1_to_cam2_s': .12})
    return d


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def observations(path):
    return [r for r in records(path) if r['type'] == 'observation']


def apply(d, operation):
    return video_transact(d, operation, read_state(d)['revision'])


def test_export_applies_interval_mask_manual_deletion_and_keeps_provenance(export_source):
    d = export_source
    apply(d, {'action': 'label_interval', 'piece': 0, 'start_frame': 3, 'end_frame': 6, 'label': 'P8'})
    polygon = [[12, 10], [40, 10], [40, 60], [12, 60]]
    apply(d, {'action': 'upsert_observation', 'id': 'det:cam1:1', 'cam': 'cam1', 'frame': 3,
              'box': [12,10,40,60], 'polygon': polygon, 'quality': 'valid'})
    apply(d, {'action': 'upsert_observation', 'cam': 'cam2', 'frame': 1,
              'box': [55,10,80,50], 'polygon': [[55,10],[80,10],[80,50],[55,50]],
              'quality': 'valid', 'label': 'P9', 'source': 'sam_propagation',
              'approval': 'accepted_propagation', 'provenance': {'job_id': 'testjob', 'model': 'small'}})
    apply(d, {'action': 'delete_observation', 'cam': 'cam1', 'frame': 9, 'id': 'det:cam1:3'})
    path = export_clip(d)
    out = records(path); obs = observations(path)
    assert out[0]['revision'] == 4 and out[0]['cameras'] == ['cam1', 'cam2']
    assert out[0]['scope'] == 'single_clip_two_cameras'
    assert out[0]['calibration_status']['synchronization'] == 'accepted'
    by_id = {r['observation_id']: r for r in obs}
    assert by_id['det:cam1:0']['label'] == 'P1'
    assert by_id['det:cam1:1']['label'] == by_id['det:cam1:2']['label'] == 'P8'
    corrected = by_id['det:cam1:1']
    assert corrected['polygon'] == polygon and corrected['mask_confirmed']
    assert corrected['mask_review_source'] == 'human_anchor'
    assert corrected['identity_id'] == 'c_export:human:P8'
    assert corrected['timestamp'] == '2026-09-19T10:00:00.120'
    assert corrected['aligned_timestamp_cam2'] == '2026-09-19T10:00:00.240'
    assert 'det:cam1:3' not in by_id
    manual = next(o for o in obs if o['observation_id'].startswith('manual:'))
    assert manual['cam'] == 'cam2' and manual['raw_frame'] == 1
    assert manual['source'] == 'sam_propagation'
    assert manual['provenance']['job_id'] == 'testjob'
    assert manual['mask_review_source'] == 'human_accepted_propagation'
    assert any(r['type'] == 'deletion' and r['observation_id'] == 'det:cam1:3' for r in out)
    assert out[-1]['complete_visits_certified'] is False


def test_identity_unknown_machine_and_full_frame_approval_are_separate(export_source):
    d = export_source
    first = observations(export_clip(d))
    cam2 = next(o for o in first if o['cam'] == 'cam2')
    assert cam2['identity_id'] == 'c_export:auto:G0'
    assert cam2['identity_status'] == 'machine_proposal'
    assert not cam2['identity_confirmed'] and cam2['unreviewed']
    apply(d, {'action': 'label_interval', 'piece': 1, 'start_frame': 0, 'end_frame': 0, 'label': '?'})
    apply(d, {'action': 'review_frame', 'cam': 'cam2', 'frame': 0, 'reviewed': True})
    obs = next(o for o in observations(export_clip(d)) if o['cam'] == 'cam2' and o['raw_frame'] == 0)
    assert obs['identity_id'] is None and obs['identity_status'] == 'UNKNOWN' and obs['ambiguous']
    assert obs['frame_reviewed'] and not obs['mask_confirmed'] and not obs['identity_confirmed']
    apply(d, {'action': 'quality_interval', 'piece': 0, 'start_frame': 3, 'end_frame': 3, 'quality': 'false_positive'})
    excluded = next(o for o in observations(export_clip(d)) if o['observation_id'] == 'det:cam1:1')
    assert excluded['excluded'] and not excluded['usable_identity']


def test_empty_sampled_frames_do_not_invent_people(export_source):
    out = records(export_clip(export_source))
    empty = [r for r in out if r['type'] == 'frame' and r['raw_frame'] == 12]
    assert len(empty) == 2
    assert all(r['teacher_sampled'] and r['observation_count'] == 0 and not r['frame_reviewed'] for r in empty)
    assert not any(r['type'] == 'observation' and r['raw_frame'] == 12 for r in out)
    assert not any(r['type'] == 'frame' and r['raw_frame'] == 1 for r in out)


def test_export_cache_is_immutable_and_invalidates_same_count_answers_and_sources(export_source):
    d = export_source
    first = export_clip(d); original = first.read_bytes()
    assert export_clip(d) == first
    apply(d, {'action': 'label_interval', 'piece': 0, 'start_frame': 0, 'end_frame': 9, 'label': 'P6'})
    second = export_clip(d)
    assert second != first and first.read_bytes() == original
    assert all(o['label'] == 'P6' for o in observations(second) if o['cam'] == 'cam1')
    atomic_json(d / 'groups_yolo26x-seg.json', [{'person': 8, 'pieces': [1]}, {'person': 7, 'pieces': [0]}])
    third = export_clip(d)
    assert third != second and second.exists()
    with pytest.raises(ValueError, match='directory'):
        export_clip(d, d / 'overwrite.ndjson')


def test_human_ids_are_scoped_to_clip_not_global(export_source, tmp_path):
    first = export_clip(export_source)
    second_folder = tmp_path / 'c_other'
    shutil.copytree(export_source, second_folder, ignore=shutil.ignore_patterns('product_exports'))
    second = export_clip(second_folder, tmp_path / 'downloads')
    a = next(o for o in observations(first) if o['label'] == 'P1')
    b = next(o for o in observations(second) if o['label'] == 'P1')
    assert a['identity_id'] != b['identity_id']
    assert b['identity_id'] == 'c_other:human:P1'


def test_concurrent_review_rejects_stale_publish_and_cleans_temporary(export_source, monkeypatch):
    original = product_export.frame_data
    changed = False
    def racing(folder, cam, frame, state=None):
        nonlocal changed
        result = original(folder, cam, frame, state=state)
        if not changed:
            changed = True
            apply(folder, {'action': 'label_interval', 'piece': 0, 'start_frame': 0, 'end_frame': 3, 'label': 'P6'})
        return result
    monkeypatch.setattr(product_export, 'frame_data', racing)
    with pytest.raises(Conflict, match='изменились'):
        export_clip(export_source)
    assert not list((export_source / 'product_exports').glob('*.ndjson'))
    assert not list((export_source / 'product_exports').glob('*.tmp'))


def test_lossless_rle_and_accepted_propagation_remain_distinct_from_human_anchor(export_source):
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[10:50, 55:80] = 1
    values = mask.flatten(order='F')
    counts, previous, length = [], 0, 0
    for value in values:
        if value == previous:
            length += 1
        else:
            counts.append(length); previous = int(value); length = 1
    counts.append(length)
    rle = {'size': [80,100], 'counts': counts}
    apply(export_source, {'action': 'upsert_observation', 'cam': 'cam2', 'frame': 1,
                         'box': [55,10,80,50], 'mask_rle': rle, 'quality': 'valid',
                         'source': 'sam_propagation', 'approval': 'accepted_propagation',
                         'provenance': {'job_id': 'rle-job', 'input_revision': 0}})
    row = next(o for o in observations(export_clip(export_source)) if o['raw_frame'] == 1)
    assert row['mask_rle'] == rle and row['polygon'] == []
    assert row['mask_review_source'] == 'human_accepted_propagation'
    assert not row['identity_confirmed'] and row['mask_confirmed']


def test_compressed_export_roundtrip_reproducible_header_and_cache(export_source):
    import gzip
    source = export_clip(export_source)
    before = source.read_bytes()
    compressed = product_export.compressed_export(source)
    zipped = compressed.read_bytes()
    assert compressed.name == source.name + '.gz'
    assert gzip.decompress(zipped) == before
    assert zipped[4:8] == b'\x00' * 4  # gzip mtime=0
    assert zipped[3] & 8 == 0  # No random temporary filename in the header.
    stamp = compressed.stat().st_mtime_ns
    assert product_export.compressed_export(source) == compressed
    assert compressed.stat().st_mtime_ns == stamp
    assert source.read_bytes() == before


def test_new_immutable_export_gets_new_gzip_without_overwriting_old(export_source):
    import gzip
    first = export_clip(export_source)
    gzip_first = product_export.compressed_export(first)
    old_plain, old_gzip = first.read_bytes(), gzip_first.read_bytes()
    apply(export_source, {'action': 'label_interval', 'piece': 0,
                         'start_frame': 0, 'end_frame': 9, 'label': 'P20'})
    second = export_clip(export_source)
    gzip_second = product_export.compressed_export(second)
    assert second != first and gzip_second != gzip_first
    assert gzip.decompress(gzip_second.read_bytes()) == second.read_bytes()
    assert gzip.decompress(gzip_second.read_bytes()) != old_plain
    assert first.read_bytes() == old_plain and gzip_first.read_bytes() == old_gzip
    assert not list(first.parent.glob('*.tmp'))


def test_compression_failure_never_publishes_partial_archive(export_source, monkeypatch):
    source = export_clip(export_source)
    before = source.read_bytes()
    def failed_copy(incoming, outgoing, length):
        outgoing.write(incoming.read(32))
        raise OSError('simulated interrupted write')
    monkeypatch.setattr(product_export.shutil, 'copyfileobj', failed_copy)
    with pytest.raises(OSError, match='interrupted'):
        product_export.compressed_export(source)
    assert source.read_bytes() == before
    assert not source.with_name(source.name + '.gz').exists()
    assert not list(source.parent.glob('*.tmp'))
