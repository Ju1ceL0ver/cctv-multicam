"""Immutable, revisioned two-camera observations with every human correction applied.

This is a product interchange format, not a claim that all exported observations
are ground truth. Identity, mask approval and full-frame completeness stay separate.
No global person identifier or floor coordinate is inferred by this exporter.
"""
import copy
import gzip
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from review_store import Conflict, fingerprint, read_state
from storage import file_lock, read_json
from video_annotations import CAMS, FPS, TAG, _assets, frame_data

SCHEMA = 'cctv.observations.v1'


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _source_paths(folder):
    data = folder.parent.parent
    return [folder / (prefix + '_' + TAG + suffix) for prefix, suffix in (
        ('meta', '.json'), ('dets', '.npz'), ('polys', '.npz'), ('grid', '.npz'),
        ('groups', '.json'))] + [folder / 'sync_estimate.json',
                              folder / 'review_media/source.json',
                              data / 'cam_sync.json', data / 'calib_final.json',
                              data / 'floor_map_gt.json']


def _signature(folder):
    result = []
    for path in _source_paths(folder):
        item = {'source': str(path.relative_to(folder)) if path.is_relative_to(folder)
                else 'data/' + path.name, 'exists': path.exists()}
        if item['exists']:
            stat = path.stat()
            item.update(size=stat.st_size, mtime_ns=stat.st_mtime_ns)
            # Small metadata can change without a size change; hash it as well.
            if path.suffix == '.json':
                item['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result.append(item)
    return result


def _calibration(folder, meta):
    data = folder.parent.parent
    local = read_json(folder / 'sync_estimate.json', {})
    table = read_json(data / 'cam_sync.json', {})
    if local.get('status') == 'accepted':
        sync, sync_status = local, 'accepted'
    else:
        sync = table.get('clips', {}).get(folder.name) or table.get(meta.get('day')) or {}
        sync_status = 'fallback' if sync else 'unavailable'
    geometry_files = [p.name for p in (data / 'calib_final.json', data / 'floor_map_gt.json') if p.exists()]
    return {'synchronization': sync_status,
            'cam1_to_cam2_s': float(sync.get('cam1_to_cam2_s', 0)),
            'geometry': 'files_present_not_validated_by_export' if geometry_files else 'unavailable',
            'geometry_files': geometry_files, 'floor_positions_exported': False,
            'note': 'No absolute floor accuracy or complete-visit accuracy is certified by this export.'}


def _group_map(folder, state):
    """Machine IDs are scoped to this clip and group definition, never human IDs."""
    groups = read_json(folder / ('groups_' + TAG + '.json'), [])
    by_piece = {}
    for index, group in enumerate(groups):
        # Source ordinal is unambiguous even if source person is null or duplicated.
        group_id = f'{folder.name}:auto:G{index}'
        for piece in group.get('pieces', []):
            by_piece.setdefault(int(piece), set()).add(group_id)
    for piece in state['pieces']:
        pid = int(piece['piece'])
        if pid not in by_piece and piece.get('origin_piece') is not None:
            by_piece[pid] = set(by_piece.get(int(piece['origin_piece']), ()))
    return by_piece


def _observation_record(clip, obs, frame, start, calibration, group_map):
    cam, index = obs['cam'], int(obs['frame'])
    label, quality = obs.get('label'), obs.get('quality', 'unreviewed')
    piece = obs.get('piece')
    local_track = f'{clip}:{cam}:piece:{piece}' if piece is not None else f'{clip}:observation:{obs["id"]}'
    candidates = sorted(group_map.get(piece, ())) if piece is not None else []
    known = isinstance(label, str) and label not in ('', '?', 'UNKNOWN')
    if known:
        identity_id, identity_status = f'{clip}:human:{label}', 'human_assigned'
    elif label in ('?', 'UNKNOWN'):
        identity_id, identity_status = None, 'UNKNOWN'
    elif len(candidates) == 1:
        identity_id, identity_status = candidates[0], 'machine_proposal'
    else:
        identity_id, identity_status = None, 'unreviewed'
    excluded = quality == 'false_positive'
    ambiguous = identity_status in ('UNKNOWN', 'unreviewed') or quality == 'mixed' or len(candidates) > 1
    seconds = index / FPS
    aligned = seconds + (calibration['cam1_to_cam2_s'] if cam == 'cam1' else 0)
    record = {'type': 'observation', 'observation_id': obs['id'], 'cam': cam,
              'raw_frame': index, 'raw_seconds': seconds,
              'timestamp': (start + timedelta(seconds=seconds)).isoformat(timespec='milliseconds'),
              'aligned_seconds_cam2': aligned,
              'aligned_timestamp_cam2': (start + timedelta(seconds=aligned)).isoformat(timespec='milliseconds'),
              'box': obs.get('box'), 'polygon': obs.get('polygon') or [],
              'local_track_id': local_track, 'identity_id': identity_id, 'label': label,
              'identity_status': identity_status, 'identity_confirmed': known,
              'mask_quality': quality, 'mask_confirmed': quality == 'valid',
              'mask_review_source': obs.get('mask_review_source', 'teacher_proposal'),
              'source': obs.get('source', 'teacher'),
              'provenance': copy.deepcopy(obs.get('provenance') or {}),
              'piece': piece, 'frame_reviewed': bool(frame['frame_reviewed']),
              'teacher_sampled': bool(frame['teacher_sampled']),
              'excluded': excluded, 'ambiguous': ambiguous,
              'unreviewed': not known or quality == 'unreviewed',
              'usable_identity': known and not excluded and quality != 'mixed',
              'machine_identity_candidates': candidates}
    if obs.get('mask_rle') is not None:
        record['mask_rle'] = obs['mask_rle']
    if obs.get('approval') is not None:
        record['approval'] = obs['approval']
    if obs.get('detection_index') is not None:
        record['teacher_detection_index'] = obs['detection_index']
    return record


def _frames(assets, state):
    wanted = {cam: set(assets['by_frame'][cam]) for cam in CAMS}
    if assets.get('sampled') is not None:
        for cam in CAMS:
            wanted[cam].update(assets['sampled'])
    video = state.get('video', {})
    for item in video.get('observations', {}).values():
        if item.get('cam') in CAMS:
            wanted[item['cam']].add(int(item['frame']))
    for name in video.get('frame_reviews', {}):
        cam, _, frame = name.partition(':')
        if cam in CAMS:
            wanted[cam].add(int(frame))
    return wanted


def _snapshot_identity(state):
    return {'revision': int(state['revision']), 'review_fingerprint': fingerprint(state)}


def export_clip(folder, destination=None):
    """Return an immutable NDJSON Path; destination is an optional output directory.

    New decisions create a new content-keyed file. A concurrent edit aborts export
    instead of publishing a file presented as the current revision. Existing valid
    snapshots remain available. This does not mutate labels, tracks or source arrays.
    """
    folder = Path(folder).resolve()
    if destination is not None and Path(destination).suffix.lower() in ('.json', '.jsonl', '.ndjson'):
        raise ValueError('destination must be a directory; immutable export filenames contain their version')
    out_dir = Path(destination) if destination is not None else folder / 'product_exports'
    state = copy.deepcopy(read_state(folder))
    signature = _signature(folder)
    identity = _snapshot_identity(state)
    cache_key = _digest({'schema': SCHEMA, 'clip': folder.name, **identity, 'sources': signature})[:24]
    output = out_dir / (folder.name + '-' + cache_key + '.ndjson')
    out_dir.mkdir(parents=True, exist_ok=True)
    with file_lock(out_dir / (cache_key + '.lock'), timeout=60):
        if output.exists():
            with file_lock(folder / '.review.lock'):
                if _snapshot_identity(read_state(folder)) != identity or _signature(folder) != signature:
                    raise Conflict('Разметка или исходные данные изменились; повторите экспорт')
            return output
        meta = read_json(folder / ('meta_' + TAG + '.json'), {})
        if not meta.get('start'):
            raise ValueError('У записи нет исходного времени начала')
        start = datetime.fromisoformat(meta['start'])
        calibration = _calibration(folder, meta)
        group_map = _group_map(folder, state)
        with np.load(folder / ('dets_' + TAG + '.npz')) as source:
            if any(cam not in source.files for cam in CAMS):
                raise ValueError('Для экспорта нужны данные обеих камер')
        assets = _assets(folder)
        wanted = _frames(assets, state)
        temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
        counts = {'frames': 0, 'observations': 0, 'excluded': 0,
                  'human_identity_observations': 0, 'mask_confirmed_observations': 0,
                  'frame_reviewed': 0, 'deleted_observations': 0}
        try:
            with temporary.open('w', encoding='utf-8', newline='\n') as stream:
                def write(item):
                    stream.write(json.dumps(item, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')
                write({'type': 'header', 'schema': SCHEMA, 'clip': folder.name,
                       'scope': 'single_clip_two_cameras', **identity, 'export_key': cache_key,
                       'input_day': meta.get('day'), 'input_start': meta['start'],
                       'input_seconds': meta.get('seconds'), 'fps': FPS, 'cameras': list(CAMS),
                       'timestamps': 'offset_aware' if start.tzinfo is not None else 'source_local_timezone_unspecified',
                       'coordinates': 'native_image_pixels_xy', 'box_format': 'xyxy',
                       'rle_format': 'COCO_uncompressed_column_major',
                       'identity_scope': 'clip_only; human IDs and machine proposals are distinct; no cross-clip identity guarantee',
                       'machine_id_scope': 'source group ordinal within this immutable snapshot; regrouping may change IDs',
                       'unknown_identity': 'identity_status=UNKNOWN and identity_id=null',
                       'calibration_status': calibration, 'sources': signature,
                       'coverage': 'Only sampled, observed, or explicitly reviewed raw frames; no invented detections or occluded trajectories.',
                       'training_warning': 'This export contains unreviewed proposals; inspect identity, mask, completeness and exclusion fields separately.'})
                for cam in CAMS:
                    for index in sorted(wanted[cam]):
                        frame = frame_data(folder, cam, index, state=state)
                        counts['frames'] += 1
                        counts['frame_reviewed'] += int(frame['frame_reviewed'])
                        write({'type': 'frame', 'cam': cam, 'raw_frame': index,
                               'timestamp': (start + timedelta(seconds=index / FPS)).isoformat(timespec='milliseconds'),
                               'width': frame['width'], 'height': frame['height'],
                               'teacher_sampled': frame['teacher_sampled'],
                               'frame_reviewed': frame['frame_reviewed'],
                               'frame_evidence': frame.get('frame_evidence'),
                               'observation_count': len(frame['observations'])})
                        for obs in frame['observations']:
                            record = _observation_record(folder.name, obs, frame, start, calibration, group_map)
                            write(record)
                            counts['observations'] += 1
                            counts['excluded'] += int(record['excluded'])
                            counts['human_identity_observations'] += int(record['identity_confirmed'])
                            counts['mask_confirmed_observations'] += int(record['mask_confirmed'])
                for oid, item in state.get('video', {}).get('observations', {}).items():
                    if item.get('deleted'):
                        write({'type': 'deletion', 'observation_id': oid, 'cam': item['cam'],
                               'raw_frame': item['frame'], 'source': 'human_exclusion', 'excluded': True})
                        counts['deleted_observations'] += 1
                write({'type': 'summary', **counts, 'complete_visits_certified': False})
                stream.flush()
                os.fsync(stream.fileno())
            with file_lock(folder / '.review.lock'):
                if _snapshot_identity(read_state(folder)) != identity or _signature(folder) != signature:
                    raise Conflict('Разметка или исходные данные изменились во время экспорта; повторите экспорт')
                os.replace(temporary, output)
            return output
        finally:
            if temporary.exists():
                temporary.unlink()


def compressed_export(path):
    """Return a cached gzip companion to an immutable NDJSON snapshot.

    export_clip gives every new revision/source snapshot a new filename. The
    gzip cache follows that filename; neither the source nor older companions
    are replaced. Header time/name are omitted for reproducible compressed bytes.
    """
    source = Path(path).resolve()
    if source.suffix.lower() != '.ndjson' or not source.is_file():
        raise ValueError('Для сжатия нужен существующий неизменяемый NDJSON-экспорт')
    output = source.with_name(source.name + '.gz')
    with file_lock(output.with_name(output.name + '.lock'), timeout=60):
        if output.exists():
            return output
        before = source.stat()
        signature = (before.st_size, before.st_mtime_ns, before.st_ino)
        temporary = output.with_name(output.name + '.' + uuid.uuid4().hex + '.tmp')
        try:
            with source.open('rb') as incoming, temporary.open('wb') as outgoing:
                with gzip.GzipFile(filename='', mode='wb', fileobj=outgoing,
                                   compresslevel=6, mtime=0) as compressed:
                    shutil.copyfileobj(incoming, compressed, length=1024 * 1024)
                outgoing.flush()
                os.fsync(outgoing.fileno())
            after = source.stat()
            if (after.st_size, after.st_mtime_ns, after.st_ino) != signature:
                raise Conflict('Исходный экспорт изменился во время сжатия; создайте новый снимок')
            os.replace(temporary, output)
            return output
        finally:
            if temporary.exists():
                temporary.unlink()
