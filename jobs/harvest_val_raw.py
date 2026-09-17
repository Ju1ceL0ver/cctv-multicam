"""Harvest the hand-review validation set from the raw 25 fps recordings.

Held out by construction: 13-14 September exist only as raw recordings, and no
pseudo-label in the training set comes from those days or from that source.
Frames carry teacher suggestions (boxes + polygons) so review is correcting,
not drawing. Sampled evenly across every 15-minute segment of both days, so
opening hours, lunch and evening light are all represented.
"""
import os, sys, json, glob, random, time

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
RAW = os.path.join(ROOT, 'runs', 'raw')
OUT = os.path.join(ROOT, 'data', 'annotate_val')
LOG = os.path.join(HOME, '_val_harvest.log')
PER_SEGMENT = 8          # candidates probed per 15-minute file
KEEP_EMPTY_SHARE = 0.10
TARGET = 520


def log(*a):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('%s %s\n' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)))


def main():
    sys.path.insert(0, ROOT); os.chdir(ROOT)
    import cv2
    from ultralytics import YOLO
    from retail_analytics.config import load_config
    from retail_analytics.devices import yolo_device
    cfg = load_config()
    model = YOLO(os.path.join(ROOT, cfg.detector.weights), task='segment')
    rng = random.Random(7)
    segs = sorted(glob.glob(os.path.join(RAW, '*', '*.mp4')))
    rng.shuffle(segs)            # interleave days/hours so an early stop is still balanced
    saved = empty = 0
    for seg in segs:
        if saved >= TARGET:
            break
        day = os.path.basename(os.path.dirname(seg))
        group = 'raw_%s' % day
        cap = cv2.VideoCapture(seg)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in sorted(rng.sample(range(0, max(1, n - 1)), min(PER_SEGMENT, max(1, n - 1)))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                continue
            if cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var() < 8.0:
                continue
            r = model.predict(source=frame, classes=[0], imgsz=1536, conf=0.10, iou=cfg.detector.iou,
                              max_det=60, retina_masks=True, half=True, device=yolo_device(),
                              verbose=False)[0]
            boxes = []
            if r.boxes is not None and len(r.boxes):
                xyxy = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy()
                xy = r.masks.xy if r.masks is not None else [None] * len(xyxy)
                for i in range(len(xyxy)):
                    p = xy[i] if i < len(xy) else None
                    poly = None
                    if p is not None and len(p) >= 3:
                        step = max(1, len(p) // 120)
                        poly = [[round(float(a), 1), round(float(b), 1)] for a, b in p[::step]]
                    x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                    boxes.append({'x1': round(x1, 1), 'y1': round(y1, 1), 'x2': round(x2, 1),
                                  'y2': round(y2, 1), 'label': 0, 'confidence': round(float(conf[i]), 4),
                                  'source': 'model', 'polygon': poly})
            if not boxes:
                if empty >= KEEP_EMPTY_SHARE * TARGET:
                    continue
                empty += 1
            stem = '%s_%06d' % (os.path.basename(seg)[:-4], idx)
            os.makedirs(os.path.join(OUT, 'images', group), exist_ok=True)
            os.makedirs(os.path.join(OUT, 'suggestions', group), exist_ok=True)
            cv2.imwrite(os.path.join(OUT, 'images', group, stem + '.jpg'), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            json.dump({'width': frame.shape[1], 'height': frame.shape[0], 'boxes': boxes,
                       'origin': 'raw', 'source_file': os.path.basename(seg), 'frame_index': idx},
                      open(os.path.join(OUT, 'suggestions', group, stem + '.json'), 'w'), indent=1)
            saved += 1
        cap.release()
        log('%s -> saved=%d empty=%d' % (os.path.basename(seg), saved, empty))
    log('DONE saved=%d empty=%d' % (saved, empty))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback; log('FATAL\n' + traceback.format_exc())
