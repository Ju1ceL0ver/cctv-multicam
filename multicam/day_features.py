"""One descriptor and one floor share per machine track of a day, computed once.

Two numbers make the whole-day labelling fast, and both are too heavy to compute inside a
request:

* **Where the track walked.** The shop floor is a hand-traced polygon per camera
  (`rooms.py`), and it deliberately excludes the mall gallery behind the glass. Measured on
  17.09: of 2874 tracks, 2489 never put a foot on the shop floor at all, and 2305 of those
  are shorter than ten seconds -- the gallery passing by. They are the bulk of the work and
  none of the shop.
* **What the person looked like.** The median OSNet descriptor over the track's detections,
  so that once the owner has named somebody, every other track of that same person can be
  offered in one grid instead of being hunted down one at a time.

Nothing here decides anything: the descriptor only orders the grid the owner confirms.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TAG = 'yolo26x-seg'
EMB = 'emb_osnet_ain_x1_0_msmt17.npz'
SAMPLE = 24                 # detections per track are enough for a stable median


def store_path(day, root=ROOT):
    return Path(root) / 'data/day_people' / ('%s_tracks.npz' % day)


def clips(day, root):
    from storage import read_json
    for folder in sorted((Path(root) / 'data/raw_clips').glob('c*')):
        meta = read_json(folder / ('meta_%s.json' % TAG), {})
        if meta.get('day') == day:
            yield folder, meta


def build(day, root=ROOT, log=print):
    import rooms
    from review_store import read_state
    root = Path(root)
    masks = {cam: rooms.mask(cam) for cam in ('cam1', 'cam2')}
    keys, floors, features, seconds = [], [], [], []
    began = time.time()
    for folder, meta in clips(day, root):
        state = read_state(folder)
        if not state['pieces'] or not (folder / ('dets_%s.npz' % TAG)).exists():
            continue
        with np.load(folder / ('dets_%s.npz' % TAG)) as archive:
            dets = {cam: archive[cam] for cam in ('cam1', 'cam2')}
        embeddings = None
        if (folder / EMB).exists():
            with np.load(folder / EMB) as archive:
                embeddings = {cam: archive[cam] for cam in ('cam1', 'cam2')}
        for piece in state['pieces']:
            cam, index = piece['cam'], piece['dets']
            rows = dets[cam][index] if index else np.zeros((0, 10))
            if not len(rows):
                continue
            feet_x = np.clip(((rows[:, 1] + rows[:, 3]) / 2).astype(int), 0, masks[cam].shape[1] - 1)
            feet_y = np.clip(rows[:, 4].astype(int), 0, masks[cam].shape[0] - 1)
            keys.append('%s:%d' % (folder.name, piece['piece']))
            floors.append(float((masks[cam][feet_y, feet_x] > 0).mean()))
            seconds.append(float(rows[:, 0].max() - rows[:, 0].min()))
            if embeddings is None or not len(embeddings[cam]):
                features.append(np.zeros(512, np.float32))
                continue
            chosen = index[:: max(1, len(index) // SAMPLE)][:SAMPLE]
            middle = np.median(embeddings[cam][chosen].astype(np.float64), axis=0)
            features.append((middle / max(np.linalg.norm(middle), 1e-8)).astype(np.float32))
        log('%s: %d треков' % (folder.name, len(keys)))
    target = store_path(day, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, keys=np.array(keys), floor=np.array(floors, np.float32),
             seconds=np.array(seconds, np.float32), feature=np.array(features, np.float32))
    log(json.dumps({'day': day, 'tracks': len(keys), 'seconds_spent': round(time.time() - began, 1),
                    'never_on_floor': int(sum(1 for f in floors if f == 0)),
                    'file': str(target)}, ensure_ascii=False))
    return target


def load(day, root=ROOT):
    """{key: (floor share, seconds, descriptor)} or None while it has not been built yet."""
    path = store_path(day, root)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as archive:
        keys = [str(k) for k in archive['keys']]
        return {'keys': keys, 'index': {k: i for i, k in enumerate(keys)},
                'floor': archive['floor'], 'seconds': archive['seconds'], 'feature': archive['feature']}


if __name__ == '__main__':
    build(sys.argv[1])
