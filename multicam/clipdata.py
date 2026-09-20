"""Load a detected raw clip and read its frames by index."""
import os, json, numpy as np
from datetime import datetime, timedelta
from rawsource import Stream, FPS
from storage import read_json

ROOT = os.path.dirname(os.path.abspath(__file__))


def load(clip, tag='yolo26x-seg', apply_sync=True):
    """Detections of a clip. With apply_sync, cam1's times are moved onto cam2's
    timeline using data/cam_sync.json (offset per recording session, keyed by day)."""
    d = os.path.join(ROOT, 'data', 'raw_clips', clip)
    z = dict(np.load(os.path.join(d, 'dets_%s.npz' % tag)))
    meta = json.load(open(os.path.join(d, 'meta_%s.json' % tag)))
    dets = {c: z[c].copy() for c in ('cam1', 'cam2')}
    if apply_sync:
        table = read_json(os.path.join(ROOT, 'data', 'cam_sync.json'), {})
        local = read_json(os.path.join(d, 'sync_estimate.json'), {})
        off = (local if local.get('status') == 'accepted' else
               table.get('clips', {}).get(clip) or table.get(meta['day']))
        override = os.environ.get('RA_SYNC_OFFSET')
        if override is not None:
            off = {'cam1_to_cam2_s': float(override)}
        if off is not None:
            seconds = float(off['cam1_to_cam2_s'])
            dets['cam1'][:, 0] = np.round((dets['cam1'][:, 0] + seconds) * FPS) / FPS
            meta['cam1_offset_s'] = seconds
        meta['sync_status'] = 'accepted' if local.get('status') == 'accepted' else 'fallback'
    feats = {c: z[c + '_feat'] for c in ('cam1', 'cam2')}
    # cleanliness: a detection whose box is largely its own silhouette and is not
    # overlapped by another person is the only kind worth trusting for appearance
    cp = os.path.join(d, 'polys_%s.npz' % tag)
    if os.path.exists(cp):
        import numpy as _np
        zp = dict(_np.load(cp))
        meta['clean'] = {}
        for c in ('cam1', 'cam2'):
            pts, off = zp[c + '_pts'], zp[c + '_off']
            b = z[c][:, 1:5]
            area = _np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1.0)
            fill = _np.zeros(len(b), _np.float32)
            for i in range(len(b)):
                a0, a1 = off[i], off[i + 1]
                if a1 - a0 >= 3:
                    q = pts[a0:a1]
                    fill[i] = abs(_np.dot(q[:, 0], _np.roll(q[:, 1], -1)) - _np.dot(q[:, 1], _np.roll(q[:, 0], -1))) / 2 / area[i]
            t = _np.round(z[c][:, 0] * FPS).astype(int)
            ov = _np.zeros(len(b), _np.float32)
            order = _np.argsort(t)
            start = 0
            for k in range(1, len(order) + 1):
                if k == len(order) or t[order[k]] != t[order[start]]:
                    g = order[start:k]
                    if len(g) > 1:
                        bb = b[g]
                        x1 = _np.maximum(bb[:, None, 0], bb[None, :, 0]); y1 = _np.maximum(bb[:, None, 1], bb[None, :, 1])
                        x2 = _np.minimum(bb[:, None, 2], bb[None, :, 2]); y2 = _np.minimum(bb[:, None, 3], bb[None, :, 3])
                        inter = _np.clip(x2 - x1, 0, None) * _np.clip(y2 - y1, 0, None)
                        frac = inter / area[g][:, None]
                        _np.fill_diagonal(frac, 0)
                        ov[g] = frac.max(1)
                    start = k
            meta['clean'][c] = (fill > 0.30) & (ov < 0.20)
    embs = None
    for name in ('osnet_ain_x1_0_msmt17_masked', 'osnet_ain_x1_0_msmt17'):
        ep = os.path.join(d, 'emb_%s.npz' % name)
        if os.path.exists(ep):
            z2 = dict(np.load(ep))
            embs = {c: z2[c] for c in ('cam1', 'cam2')}
            meta['embeddings'] = name
            break
    return dets, feats, meta, embs


def frames_needed(meta, wanted):
    """wanted: {cam: set(frame_index)} -> {cam: {frame_index: frame (half res)}}; one sequential pass."""
    import cv2
    start = datetime.fromisoformat(meta['start'])
    out = {c: {} for c in wanted}
    last = max((max(v) for v in wanted.values() if v), default=-1)
    streams = {c: Stream(c, meta['day']) for c in wanted}
    for s in streams.values():
        s.seek(start)
    for k in range(last + 1):
        for c, s in streams.items():
            t, fr = s.read()
            if fr is not None and k in wanted[c]:
                out[c][k] = cv2.resize(fr, (1280, 720))
    return out
