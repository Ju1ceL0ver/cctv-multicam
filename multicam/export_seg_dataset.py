"""Export immutable, fully annotated teacher-mask frames from human-reviewed pieces.

Unknown identity and mask quality are separate. If any teacher detection in a frame
is unresolved, omit the whole frame; never silently teach that person as background.
"""
import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
from rawsource import Stream, FPS
from review_store import read_state, fingerprint
from storage import atomic_json, read_json, file_lock

ROOT = Path(__file__).resolve().parent
CLIPS = ROOT / 'data/raw_clips'
TAG = 'yolo26x-seg'
VERSION = 3


def reviewed(clip):
    state = read_state(Path(CLIPS) / clip)
    return (state['labels'], state['pieces']) if state['labels'] or state['quality'] else None


def source_manifest(clips=CLIPS, gap=1.0, val=('c183400',)):
    sources = []
    validation_windows = []
    for clip in val:
        meta = read_json(Path(clips) / clip / ('meta_' + TAG + '.json'), {})
        if meta.get('start') and meta.get('seconds'):
            start = datetime.fromisoformat(meta['start'])
            validation_windows.append((start, start + timedelta(seconds=float(meta['seconds']))))
    for d in sorted(Path(clips).glob('c*')):
        if not d.is_dir():
            continue
        state = read_state(d)
        if not state['labels'] and not state['quality'] and not state.get('video'):
            continue
        files = [d / (n + '_' + TAG + ext) for n, ext in
                 (('dets', '.npz'), ('polys', '.npz'), ('meta', '.json'))]
        if not all(p.exists() for p in files):
            continue
        split = 'val' if d.name in val else 'train'
        meta = read_json(d / ('meta_' + TAG + '.json'))
        start = datetime.fromisoformat(meta['start'])
        end = start + timedelta(seconds=float(meta['seconds']))
        if split == 'train' and any(start < b and a < end for a, b in validation_windows):
            continue  # Different clip names can still refer to the same raw frames.
        sources.append({'clip': d.name, 'review': fingerprint(state),
                        'split': split,
                        'files': [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]})
    result = {'version': VERSION, 'gap': gap, 'val': list(val), 'sources': sources}
    result['id'] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()[:20]
    return result


def detection_status(state, cam):
    """accepted positive, explicitly rejected false detection, or unresolved."""
    status = {}
    for p in state['pieces']:
        if p['cam'] != cam:
            continue
        key = str(p['piece'])
        quality = state['quality'].get(key, 'unreviewed')
        label = state['labels'].get(key)
        value = ('negative' if quality == 'false_positive' else
                 'positive' if quality == 'valid' or
                 (quality == 'unreviewed' and label and label != '?') else 'unresolved')
        for i in p['dets']:
            if i in status and status[i] != value:
                status[i] = 'unresolved'
            else:
                status[i] = value
    return status


def select_frames(detections, status, offsets, gap=1.0):
    """Retain full frames only if every teacher detection has a usable decision."""
    by = {}
    for i, f in enumerate(np.rint(detections[:, 0] * FPS).astype(int)):
        by.setdefault(int(f), []).append(i)
    selected, skipped, last = {}, 0, -1e9
    for f, ids in sorted(by.items()):
        if any(status.get(i, 'unresolved') == 'unresolved' or
               (status.get(i) == 'positive' and offsets[i+1] - offsets[i] < 3) for i in ids):
            skipped += 1; continue
        if f / FPS - last < gap:
            continue
        selected[f] = [i for i in ids if status.get(i) == 'positive']
        last = f / FPS
    return selected, skipped


def export(gap=1.0, val=('c183400',), destination=None):
    manifest = source_manifest(gap=gap, val=val)
    base = ROOT / 'data/seg_datasets'
    out = Path(destination) if destination else base / manifest['id']
    with file_lock(base / '.export.lock'):
        existing = read_json(out / 'manifest.json')
        if existing and existing.get('id') == manifest['id'] and existing.get('complete'):
            atomic_json(ROOT / 'data/seg_dataset_current.json', {'path': str(out), 'id': manifest['id']})
            print('dataset unchanged:', out, flush=True)
            return out, existing
        if out.exists():
            raise RuntimeError('Destination exists without matching completed manifest: %s' % out)
        stage = out.with_name(out.name + '.building')
        # Only this task's unfinished staging directory is disposable.
        if stage.exists():
            shutil.rmtree(stage)
        for split in ('train', 'val'):
            for sub in ('images', 'labels'):
                (stage / split / sub).mkdir(parents=True, exist_ok=True)
        counts = {s: {'frames': 0, 'instances': 0} for s in ('train', 'val')}
        details = []
        for source in manifest['sources']:
            clip = source['clip']; d = CLIPS / clip
            state = read_state(d)
            if fingerprint(state) != source['review']:
                raise RuntimeError('Review changed during export; retry for a consistent snapshot')
            meta = read_json(d / ('meta_' + TAG + '.json'))
            split = source.get('split', 'val' if clip in val else 'train')
            with np.load(d / ('dets_' + TAG + '.npz')) as dz, np.load(d / ('polys_' + TAG + '.npz')) as pz:
                for cam in ('cam1', 'cam2'):
                    pts, off = pz[cam + '_pts'], pz[cam + '_off']
                    curated = bool(state.get('video'))
                    exclusions = {}
                    if curated:
                        from curated_export import select_corrected_frames
                        wanted, skipped, exclusions = select_corrected_frames(d, cam, dz[cam], state, gap)
                    else:
                        wanted, skipped = select_frames(dz[cam], detection_status(state, cam), off, gap)
                    details.append({'clip': clip, 'cam': cam, 'selected': len(wanted), 'unresolved_frames': skipped, 'corrected': curated, 'exclusions': exclusions})
                    if not wanted:
                        continue
                    stream = Stream(cam, meta['day']); stream.seek(datetime.fromisoformat(meta['start']))
                    raw_gap = None
                    try:
                        for k in range(max(wanted) + 1):
                            stamp, frame = stream.read()
                            if frame is None or stamp is None:
                                raise RuntimeError('Raw video ended early: %s %s frame %d' % (clip, cam, k))
                            expected = datetime.fromisoformat(meta['start']) + timedelta(seconds=k / FPS)
                            if abs((stamp - expected).total_seconds()) > .5 / FPS:
                                # The recording skips here. Nothing at or after this frame is trusted, but
                                # the frames before it are good and the other clips are unaffected: one gap
                                # in one camera must not stop the whole night's training.
                                raw_gap = {'frame': k, 'offset_s': round((stamp - expected).total_seconds(), 3),
                                           'dropped_frames': sum(1 for f in wanted if f >= k)}
                                print('Raw recording gap, later frames dropped: %s %s %s' % (clip, cam, raw_gap), flush=True)
                                break
                            if k not in wanted:
                                continue
                            h, w = frame.shape[:2]
                            rows = []
                            for i in wanted[k]:
                                poly = (i['polygon'] if curated else pts[off[i]:off[i+1]].astype(float)) / [w, h]
                                poly = np.clip(poly, 0, 1)
                                rows.append('0 ' + ' '.join('%.5f %.5f' % (x, y) for x, y in poly))
                            name = '%s_%s_%06d' % (clip, cam, k)
                            target = stage / split / 'images' / (name + '.jpg')
                            if not cv2.imwrite(str(target), cv2.resize(frame, (round(w*.625), round(h*.625))), [cv2.IMWRITE_JPEG_QUALITY, 88]):
                                raise RuntimeError('Could not write image: %s' % target)
                            (stage / split / 'labels' / (name + '.txt')).write_text('\n'.join(rows) + ('\n' if rows else ''))
                            counts[split]['frames'] += 1; counts[split]['instances'] += len(rows)
                    finally:
                        if stream.cap is not None: stream.cap.release()
                    if raw_gap:
                        details[-1]['raw_gap'] = raw_gap
            if fingerprint(read_state(d)) != source['review']:
                raise RuntimeError('Review changed during export; discard staging and retry')
            print(clip, counts, flush=True)
        manifest.update(counts=counts, details=details, complete=True,
                        target_provenance='teacher masks plus explicitly reviewed video corrections; provenance retained in source state; not independent pixel ground truth')
        (stage / 'data.yaml').write_text('path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: person\n' % out.as_posix())
        atomic_json(stage / 'manifest.json', manifest)
        stage.rename(out)
        atomic_json(ROOT / 'data/seg_dataset_current.json', {'path': str(out), 'id': manifest['id']})
        print('dataset:', counts, '->', out, flush=True)
        return out, manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', nargs='?'); p.add_argument('--gap', type=float, default=1)
    p.add_argument('--val', default='c183400'); p.add_argument('--manifest-only', action='store_true')
    a = p.parse_args()
    if a.manifest_only: print(json.dumps(source_manifest(gap=a.gap, val=a.val.split(',')), indent=1))
    else: export(a.gap, a.val.split(','), a.out)
