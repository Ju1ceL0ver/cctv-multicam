"""ReID embeddings for every detection of a raw clip (both cameras), one or more models.

usage: embed_clip.py CLIP model1.pt [model2.pt ...]  -> data/raw_clips/CLIP/emb_<model>.npz (cam1, cam2 aligned with dets)"""
import sys, os, time, json, numpy as np, torch
from datetime import datetime
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from boxmot.reid.core.reid import ReID
from clipdata import load
from rawsource import Stream, FPS

WDIR = os.path.join(ROOT, 'data', 'weights'); os.makedirs(WDIR, exist_ok=True)
LOCAL = {'osnet_x0_25_msmt17.pt': r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models\osnet_x0_25_msmt17.pt'}


def load_polys(clip, tag='yolo26x-seg'):
    p = os.path.join(ROOT, 'data', 'raw_clips', clip, 'polys_%s.npz' % tag)
    if not os.path.exists(p):
        return None
    z = dict(np.load(p))
    return {c: (z[c + '_pts'], z[c + '_off']) for c in ('cam1', 'cam2')}


def masked_crops(frame, boxes, polys, idx, cam, fill=(114, 114, 114)):
    """Crops with everything outside the person's own outline replaced by flat grey:
    when two people stand shoulder to shoulder, the neighbour would otherwise be part
    of the descriptor."""
    import cv2
    out = []
    for n, b in zip(idx, boxes):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 4 or y2 - y1 < 8:
            out.append(np.zeros((16, 8, 3), np.uint8)); continue
        crop = frame[y1:y2, x1:x2].copy()
        if polys is not None:
            pts, off = polys[cam]
            a, b2 = off[n], off[n + 1]
            if b2 > a:
                poly = (pts[a:b2] - [x1, y1]).astype(np.int32)
                m = np.zeros(crop.shape[:2], np.uint8)
                cv2.fillPoly(m, [poly], 255)
                m = cv2.dilate(m, np.ones((5, 5), np.uint8))
                crop = np.where(m[..., None] > 0, crop, np.array(fill, np.uint8))
        out.append(crop)
    return out


def main():
    clip, models = sys.argv[1], sys.argv[2:]
    masked = os.environ.get('RA_MASKED', '1') == '1'
    dets, feats, meta, _emb = load(clip, apply_sync=False)
    polys = load_polys(clip) if masked else None
    print('masked crops:', polys is not None, flush=True)
    nets = {}
    for m in models:
        path = LOCAL.get(m, os.path.join(WDIR, m))
        nets[m] = ReID(path, device='cuda:0', half=True)
        print('loaded', m, flush=True)
    frame_of = {c: np.round(dets[c][:, 0] * FPS).astype(int) for c in dets}
    by_frame = {c: {} for c in dets}
    for c in dets:
        for i, f in enumerate(frame_of[c]):
            by_frame[c].setdefault(int(f), []).append(i)
    out = {m: {c: np.zeros((len(dets[c]), 0), np.float32) for c in dets} for m in models}
    start = datetime.fromisoformat(meta['start'])
    n = int(meta['frames']); t0 = time.time()
    for c in ('cam1', 'cam2'):
        s = Stream(c, meta['day']); s.seek(start)
        store = {m: None for m in models}
        for k in range(n):
            t, fr = s.read()
            if fr is None:
                break
            idx = by_frame[c].get(k)
            if not idx:
                continue
            boxes = dets[c][idx, 1:5].astype(np.float32)
            crops = masked_crops(fr, boxes, polys, idx, c) if polys is not None else None
            for m, net in nets.items():
                with torch.inference_mode():
                    e = np.asarray(net(crops, None) if crops is not None else net(fr, boxes), np.float32)
                if store[m] is None:
                    store[m] = np.zeros((len(dets[c]), e.shape[1]), np.float32)
                store[m][idx] = e
            if k % 2000 == 0:
                print('%s %s frame %d/%d %.1f fps' % (time.strftime('%H:%M:%S'), c, k, n, (k + 1) / (time.time() - t0)), flush=True)
        for m in models:
            out[m][c] = store[m]
        t0 = time.time()
    d = os.path.join(ROOT, 'data', 'raw_clips', clip)
    suffix = '_masked' if polys is not None else ''
    for m in models:
        np.savez(os.path.join(d, 'emb_%s%s.npz' % (os.path.splitext(m)[0], suffix)), **out[m])
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
