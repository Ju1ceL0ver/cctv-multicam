"""One segmentation pass over synced windows of the raw recordings, both cameras.

There is no separate "detection": the big segmentation model yields the box, the
outline and the foot/head points in a single run, and the outline is what every later
step needs -- masked crops for appearance (a neighbour standing shoulder to shoulder
is erased), the foot point on the floor, and the silhouettes small models are later
distilled on.

Per detection: t (s from window start), box, score, foot and head pixel from the mask,
18-d Lab clothing descriptor -- the same layout as the local prototype, so the
association code runs unchanged.

usage: detect_raw.py CLIP_ID DAY HH:MM:SS SECONDS [weights] [imgsz]"""
import os, sys, time, json, numpy as np, cv2
from datetime import datetime
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(__file__))
from rawsource import Stream, FPS

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, 'data', 'raw_clips')


def describe(frame, poly, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if x2 - x1 < 4 or y2 - y1 < 12:
        return np.zeros(18, np.float32)
    crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
    m = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillPoly(m, [(poly - [x1, y1]).astype(np.int32)], 1)
    h = y2 - y1
    out = []
    for a, b in ((0.0, 0.2), (0.2, 0.55), (0.55, 0.95)):
        part = crop[int(a * h):int(b * h)]
        mm = m[int(a * h):int(b * h)].astype(bool)
        px = part[mm] if mm.sum() > 20 else part.reshape(-1, 3)
        out += list(px.mean(0)) + list(px.std(0))
    return np.array(out, np.float32)


MAXPTS = 64
REID = os.environ.get('RA_REID', 'osnet_ain_x1_0_msmt17.pt')   # '' turns the appearance pass off
MOTION = float(os.environ.get('RA_MOTION', '0.0015'))   # share of changed pixels that counts as movement
HOLD = float(os.environ.get('RA_HOLD', '20'))           # gate stays open this long after the last sign of a person
BATCH = int(os.environ.get('RA_BATCH', '3'))            # timesteps per forward pass: the card idles on pairs alone


def gray(fr):
    return cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (320, 180))


def main():
    clip, day, hms, seconds = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    weights = sys.argv[5] if len(sys.argv) > 5 else r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models\yolo26x-seg.pt'
    imgsz = int(sys.argv[6]) if len(sys.argv) > 6 else 1536
    stride = int(sys.argv[7]) if len(sys.argv) > 7 else int(os.environ.get('RA_STRIDE', '1'))
    tag = os.path.splitext(os.path.basename(weights))[0].split('_')[0]
    start = datetime.strptime(day + hms, '%Y%m%d%H:%M:%S')
    odir = os.path.join(OUT, clip); os.makedirs(odir, exist_ok=True)
    log = open(os.path.join(odir, 'detect_%s.log' % tag), 'a')
    model = YOLO(weights, task='segment')
    net = None
    if REID:
        from boxmot.reid.core.reid import ReID
        net = ReID(os.path.join(ROOT, 'data', 'weights', REID), device='cuda:0', half=True)
    embs = {c: [] for c in ('cam1', 'cam2')}
    streams = {c: Stream(c, day) for c in ('cam1', 'cam2')}
    for s in streams.values():
        s.seek(start)
    rows = {c: [] for c in streams}; feats = {c: [] for c in streams}; polys = {c: [] for c in streams}
    n = int(round(seconds * FPS)); t0 = time.time()
    prev = {c: None for c in streams}
    seen = -1e9        # last time anything alive was observed
    kept = []          # frame indices the model actually ran on
    pend = []          # frames waiting to be looked at together
    k = 0

    def flush(batch, seen):
        imgs = [b[1][c] for b in batch for c in ('cam1', 'cam2')]
        res = model.predict(source=imgs, imgsz=imgsz, conf=0.25, classes=[0], half=True,
                            retina_masks=False, verbose=False)
        alive = None
        if net is not None:
            crops = [(i, imgs[i], r.boxes.xyxy.cpu().numpy().astype(np.float32))
                     for i, r in enumerate(res) if r.boxes is not None and len(r.boxes)]
            vecs = {i: np.asarray(net(im, b), np.float32) for i, im, b in crops}
        for j, (kk, fr) in enumerate(batch):
            for ci, c in enumerate(('cam1', 'cam2')):
                r = res[2 * j + ci]
                if r.boxes is None or not len(r.boxes):
                    continue
                alive = kk / FPS
                B = r.boxes.xyxy.cpu().numpy(); S = r.boxes.conf.cpu().numpy()
                if net is not None:
                    embs[c].append(vecs[2 * j + ci])
                mxy = r.masks.xy if r.masks is not None else [None] * len(B)
                for i, (x1, y1, x2, y2) in enumerate(B):
                    p = mxy[i] if i < len(mxy) else None
                    if p is None or len(p) < 4:
                        p = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                    ymax, ymin = p[:, 1].max(), p[:, 1].min()
                    band = 0.05 * (ymax - ymin)
                    foot = p[p[:, 1] >= ymax - band].mean(0); head = p[p[:, 1] <= ymin + band].mean(0)
                    rows[c].append((kk / FPS, x1, y1, x2, y2, S[i], foot[0], ymax, head[0], ymin))
                    feats[c].append(describe(fr[c], p, (x1, y1, x2, y2)))
                    step = max(1, len(p) // MAXPTS)
                    polys[c].append(np.asarray(p[::step], np.float32))
        return alive
    for k in range(n):
        frames = {}
        for c, s in streams.items():
            t, fr = s.read()
            if t is None:
                break
            frames[c] = fr
        if len(frames) < 2:
            break
        if k % stride:
            continue
        moved = False
        for c, fr in frames.items():
            g = gray(fr)
            if prev[c] is not None and float((cv2.absdiff(g, prev[c]) > 24).mean()) > MOTION:
                moved = True
            prev[c] = g
        tk = k / FPS
        if moved:
            seen = tk
        if k % 2500 < stride:
            log.write('%s frame %d/%d  looked at %d  %.1f frames/s\n'
                      % (time.strftime('%H:%M:%S'), k, n, len(kept), (k + 1) / (time.time() - t0))); log.flush()
        if tk - seen > HOLD:            # empty hall: decoding only, the teacher is not spent here
            continue
        kept.append(k)
        pend.append((k, frames))
        if len(pend) < BATCH:
            continue
        seen = flush(pend, seen) or seen
        pend = []
    if pend:
        flush(pend, seen)
    out = {}
    for c in streams:
        out[c] = np.array(rows[c], np.float32).reshape(-1, 10)
        out[c + '_feat'] = np.array(feats[c], np.float32).reshape(-1, 18)
    np.savez(os.path.join(odir, 'dets_%s.npz' % tag), **out)
    if net is not None:
        e = {}
        for c in streams:
            e[c] = np.concatenate(embs[c]) if embs[c] else np.zeros((0, 512), np.float32)
        np.savez(os.path.join(odir, 'emb_%s.npz' % os.path.splitext(REID)[0]), **e)
    pout = {}
    for c in streams:
        off = np.zeros(len(polys[c]) + 1, np.int64)
        for i, pl in enumerate(polys[c]):
            off[i + 1] = off[i] + len(pl)
        pout[c + '_pts'] = np.concatenate(polys[c]) if polys[c] else np.zeros((0, 2), np.float32)
        pout[c + '_off'] = off
    np.savez_compressed(os.path.join(odir, 'polys_%s.npz' % tag), **pout)
    np.savez_compressed(os.path.join(odir, 'grid_%s.npz' % tag), t=np.array(kept, np.float32) / FPS)
    json.dump({'clip': clip, 'day': day, 'start': start.isoformat(), 'seconds': seconds, 'weights': weights,
               'imgsz': imgsz, 'stride': stride, 'frames': k + 1, 'looked_at': len(kept),
               'detections': {c: int(len(out[c])) for c in streams}},
              open(os.path.join(odir, 'meta_%s.json' % tag), 'w'), indent=1)
    log.write('%s DONE %s  looked at %d/%d frames in %.0f s\n'
              % (time.strftime('%H:%M:%S'), {c: len(out[c]) for c in streams}, len(kept), k + 1, time.time() - t0))
    log.close()


if __name__ == '__main__':
    main()
