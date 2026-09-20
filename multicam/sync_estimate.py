"""Time offset between the two cameras' raw recordings, from people alone.

The recorders name segments by when ffmpeg was launched, but each RTSP stream
starts after its own connection delay, so the two cameras' timelines differ by
seconds -- and by a different amount after every recorder restart. For each
candidate shift, count moments where a WALKING person on cam2 has a cam1
detection on the same floor spot (standing people agree at any shift, so they
carry no timing information). The peak is the offset.

usage: sync_estimate.py CLIP [TAG]   -> prints the peak; cam1 time + offset = cam2 time"""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build
from clipdata import load
from storage import atomic_json, read_json
import argparse


def estimate(dets, feats, embs=None, clean=None, span=8.0, step=0.04, radius=0.4, min_speed=0.5, details=False):
    calib = json.load(open('data/calib_final.json'))
    cams = {c: Camera(c, calib) for c in ('cam1', 'cam2')}
    per_cam, _ = build(cams, dets, feats, embs, clean)
    c1 = [(it['t'], it['xy']) for it in per_cam['cam1']]
    moving = []
    for it in per_cam['cam2']:
        t, xy = it['t'], it['xy']
        if len(t) < 10:
            continue
        v = np.gradient(xy, axis=0) / np.maximum(np.gradient(t), 1e-3)[:, None]
        sp = np.linalg.norm(v, axis=1)
        k = sp > min_speed
        moving.append(np.c_[t[k], xy[k]])
    M = np.concatenate(moving) if moving else np.zeros((0, 3))
    T1 = np.concatenate([np.c_[t, xy] for t, xy in c1]) if c1 else np.zeros((0, 3))
    shifts = np.arange(-span, span + 1e-9, step)
    scores = []
    block_scores = []
    edges = np.linspace(float(M[:, 0].min()), float(M[:, 0].max()) + 1e-6, 4) if len(M) else np.arange(4)
    blocks = np.clip(np.searchsorted(edges, M[:, 0], side="right") - 1, 0, 2)
    q = np.round(M[:, 0] / step).astype(int)
    for s in shifts:
        k1 = np.round((T1[:, 0] + s) / step).astype(int)
        by = {}
        for i, kk in enumerate(k1):
            by.setdefault(kk, []).append(i)
        hit = 0
        bh = np.zeros(3, int)
        for j, kk in enumerate(q):
            idx = by.get(kk)
            if idx and np.min(np.linalg.norm(T1[idx, 1:] - M[j, 1:], axis=1)) < radius:
                hit += 1
                bh[blocks[j]] += 1
        scores.append(hit)
        block_scores.append(bh)
    scores = np.array(scores, float)
    b = int(np.argmax(scores))
    others = np.abs(shifts - shifts[b]) > 0.6
    result = (shifts[b], scores[b], float(np.median(scores[others])), len(M), shifts, scores)
    if not details:
        return result
    bs = np.array(block_scores)
    votes = [float(shifts[np.argmax(bs[:, j])]) for j in range(3) if bs[:, j].max() >= 15]
    second = float(scores[others].max()) if others.any() else 0
    consistent = (len(M) >= 50 and scores[b] >= 40 and abs(shifts[b]) < span - .2
                  and scores[b] > 1.25 * second + 5
                  and len(votes) >= 2 and all(abs(v - shifts[b]) <= .4 for v in votes))
    return result, {'consistent': bool(consistent), 'temporal_votes_s': votes,
                    'peak': int(scores[b]), 'second_peak': second, 'moving_points': len(M),
                    'policy': 'conservative temporal agreement v1; not a calibrated probability'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('clip'); parser.add_argument('tag', nargs='?', default='yolo26x-seg')
    # Existing overnight workers may have been started before the CLI flags were
    # added. Their children already inherit RA_WORKER, so they can use measured
    # offsets on the next window without interrupting an active GPU pass.
    automatic = os.environ.get('RA_WORKER') in ('_am', '_pm')
    parser.add_argument('--write', action='store_true', default=automatic)
    parser.add_argument('--accept-consistent', action='store_true', default=automatic)
    parser.add_argument('--true-time', action='store_true',
                        help='take each frame\'s time from the recording instead of counting 25 a second')
    args = parser.parse_args()
    dets, feats, meta, embs = load(args.clip, args.tag, apply_sync=False, true_times=args.true_time)
    if meta.get('true_t'):
        # Both cameras on one true timeline, so a difference between them is the cameras'
        # offset and not the frames each of them happened to drop.
        origin = min(t.min() for t in meta['true_t'].values() if len(t))
        dets = {cam: d.copy() for cam, d in dets.items()}
        for cam, d in dets.items():
            if len(d):
                d[:, 0] = meta['true_t'][cam] - origin
    result, evidence = estimate(dets, feats, embs, meta.get('clean'), details=True)
    s, peak, base, n, shifts, scores = result
    record = {'clip': args.clip, 'day': meta['day'], 'estimated_offset_s': round(float(s), 4),
              'evidence': evidence, 'status': 'candidate', 'method': 'moving floor trajectories',
              'time_source': meta.get('time_source', 'index')}
    if args.accept_consistent and evidence['consistent']:
        record.update(status='accepted', cam1_to_cam2_s=round(float(s), 4))
    if args.write:
        path = os.path.join('data', 'raw_clips', args.clip, 'sync_estimate.json')
        previous = read_json(path, {})
        if previous.get('status') == 'accepted' and record['status'] != 'accepted':
            previous['latest_candidate'] = record
            record = previous
        atomic_json(path, record)
    print(json.dumps(record))
