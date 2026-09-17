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


def main():
    clip, day, hms, seconds = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    weights = sys.argv[5] if len(sys.argv) > 5 else r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models\yolo26x-seg.pt'
    imgsz = int(sys.argv[6]) if len(sys.argv) > 6 else 1536
    tag = os.path.splitext(os.path.basename(weights))[0]
    start = datetime.strptime(day + hms, '%Y%m%d%H:%M:%S')
    odir = os.path.join(OUT, clip); os.makedirs(odir, exist_ok=True)
    log = open(os.path.join(odir, 'detect_%s.log' % tag), 'a')
    model = YOLO(weights, task='segment')
    streams = {c: Stream(c, day) for c in ('cam1', 'cam2')}
    for s in streams.values():
        s.seek(start)
    rows = {c: [] for c in streams}; feats = {c: [] for c in streams}; polys = {c: [] for c in streams}
    n = int(round(seconds * FPS)); t0 = time.time()
    for k in range(n):
        frames = {}
        for c, s in streams.items():
            t, fr = s.read()
            if t is None:
                break
            frames[c] = fr
        if len(frames) < 2:
            break
        res = model.predict(source=[frames['cam1'], frames['cam2']], imgsz=imgsz, conf=0.25, classes=[0], half=True,
                            retina_masks=False, verbose=False)
        for c, r in zip(('cam1', 'cam2'), res):
            if r.boxes is None or not len(r.boxes):
                continue
            B = r.boxes.xyxy.cpu().numpy(); S = r.boxes.conf.cpu().numpy()
            polys = r.masks.xy if r.masks is not None else [None] * len(B)
            for i, (x1, y1, x2, y2) in enumerate(B):
                p = polys[i] if i < len(polys) else None
                if p is None or len(p) < 4:
                    p = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
                ymax, ymin = p[:, 1].max(), p[:, 1].min()
                band = 0.05 * (ymax - ymin)
                foot = p[p[:, 1] >= ymax - band].mean(0); head = p[p[:, 1] <= ymin + band].mean(0)
                rows[c].append((k / FPS, x1, y1, x2, y2, S[i], foot[0], ymax, head[0], ymin))
                feats[c].append(describe(frames[c], p, (x1, y1, x2, y2)))
                step = max(1, len(p) // MAXPTS)
                polys[c].append(np.asarray(p[::step], np.float32))
        if k % 500 == 0:
            log.write('%s frame %d/%d  %.1f fps\n' % (time.strftime('%H:%M:%S'), k, n, (k + 1) / (time.time() - t0))); log.flush()
    out = {}
    for c in streams:
        out[c] = np.array(rows[c], np.float32).reshape(-1, 10)
        out[c + '_feat'] = np.array(feats[c], np.float32).reshape(-1, 18)
    np.savez(os.path.join(odir, 'dets_%s.npz' % tag), **out)
    pout = {}
    for c in streams:
        off = np.zeros(len(polys[c]) + 1, np.int64)
        for i, pl in enumerate(polys[c]):
            off[i + 1] = off[i] + len(pl)
        pout[c + '_pts'] = np.concatenate(polys[c]) if polys[c] else np.zeros((0, 2), np.float32)
        pout[c + '_off'] = off
    np.savez_compressed(os.path.join(odir, 'polys_%s.npz' % tag), **pout)
    json.dump({'clip': clip, 'day': day, 'start': start.isoformat(), 'seconds': seconds, 'weights': weights, 'imgsz': imgsz,
               'frames': k + 1, 'detections': {c: int(len(out[c])) for c in streams}},
              open(os.path.join(odir, 'meta_%s.json' % tag), 'w'), indent=1)
    log.write('%s DONE %s\n' % (time.strftime('%H:%M:%S'), {c: len(out[c]) for c in streams})); log.close()


if __name__ == '__main__':
    main()
