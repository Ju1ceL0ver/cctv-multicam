"""Distillation harvest: label footage with the big model at full strength.

This is the teacher pass. It is deliberately slower and more accurate than
anything the live pipeline can afford:

* ``yolo26x-seg`` at ``imgsz=1536`` with ``conf=0.10`` -- the live pipeline runs
  960/0.25 because it has to keep up with the camera; here we have the GPU to
  ourselves overnight, so the teacher runs in a regime the student will never
  see and therefore has something to teach.
* Masks are kept as polygons, not just boxes. The classifier downstream weighs
  the crop by its silhouette, so a student trained detect-only would hand it a
  box full of laminate display racks. Teacher masks are free -- the segment
  model computes them anyway -- so the student can be ``-seg`` at no labelling
  cost.
* Detections are run through BoT-SORT + OSNet ReID and then **gap-filled along
  each track**: if a track is observed at t-1 and t+1 but the detector missed
  it at t, that is a teacher miss, and the box is interpolated back in. This is
  what lets the student beat the teacher rather than merely copy it -- temporal
  consistency is information the single-frame teacher does not have.

Sampling is quota-driven, not "every Nth frame": frames with people away from
the door ROI are always kept (they are the whole point of going full-frame),
door-only frames are subsampled, and a bounded share of genuinely empty frames
is kept as hard negatives -- this scene has a life-size human figure on a wall
poster and rows of vertical laminate panels, and the student needs to be shown
that those are not people.

The run stops hard at DEADLINE (five minutes before the live service's window
opens) so the GPU is free when the shop opens.
"""

import os, sys, json, time, glob, random, traceback
from datetime import datetime, timedelta

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
OUT = os.path.join(ROOT, 'data', 'annotate_smoke' if os.environ.get('RA_HARVEST_SMOKE') == '1'
                   else 'annotate')
LOG = os.path.join(HOME, '_teacher_harvest.log')
STATE = os.path.join(HOME, '_teacher_harvest_state.json')

DEADLINE_HHMM = (9, 55)          # stop, whatever happens, at 09:55 local
SEGMENT_FRAMES = 300             # consecutive frames per segment (tracking needs continuity)
WARMUP_FRAMES = 20               # tracker needs a few frames before its ids mean anything
SAVE_EVERY = 6                   # candidate frame cadence inside a segment
MAX_INTERP_GAP = 12              # never bridge a gap longer than ~0.5 s
TEACHER_IMGSZ = 1536
TEACHER_CONF = 0.10
MAX_DET = 60
MIN_BLUR = 8.0
TARGET_FRAMES = 14000            # stop early if we somehow get this far
EMPTY_FRAME_SHARE = 0.08         # cap hard negatives at 8% of the set

def log(*a):
    line = '%s %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')

def deadline_ts():
    now = datetime.now()
    d = now.replace(hour=DEADLINE_HHMM[0], minute=DEADLINE_HHMM[1], second=0, microsecond=0)
    if d <= now:
        d += timedelta(days=1)
    return d.timestamp()

# A smoke run uses the same code path on the same data, just bounded to a
# couple of minutes -- the point is to prove the imports, the tracker API and
# the polygon extraction before committing the GPU for five hours.
SMOKE = os.environ.get('RA_HARVEST_SMOKE') == '1'
if SMOKE:
    TARGET_FRAMES = 40
    SNAPSHOT_LIMIT = 10
else:
    SNAPSHOT_LIMIT = 0

DEADLINE = time.time() + 180 if SMOKE else deadline_ts()

def past_deadline():
    return time.time() >= DEADLINE


# ----------------------------------------------------------------- geometry

def box_centre(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

def outside_roi(box, roi):
    cx, cy = box_centre(box)
    x1, y1, x2, y2 = roi
    return not (x1 <= cx <= x2 and y1 <= cy <= y2)


# -------------------------------------------------------------- interpolate

def polygon_for(box, ref_box, ref_polygon):
    """Carry a neighbouring frame's silhouette onto an interpolated box.

    Over the <=0.5 s a gap is allowed to span, a person's outline barely
    changes shape -- it mostly translates and scales with the box. Normalising
    the reference polygon by its own box and re-applying it to the interpolated
    one is a far better mask than filling the rectangle, which is what a
    box-only label would force on the classifier later.
    """
    if not ref_polygon:
        return None
    rw = max(1e-6, ref_box[2] - ref_box[0]); rh = max(1e-6, ref_box[3] - ref_box[1])
    w = box[2] - box[0]; h = box[3] - box[1]
    return [[box[0] + (px - ref_box[0]) / rw * w,
             box[1] + (py - ref_box[1]) / rh * h] for px, py in ref_polygon]


def fill_track_gaps(observations, frames_wanted):
    """observations: {track_id: {frame_idx: (box, polygon, score)}}.

    Returns {frame_idx: [(box, polygon, score, track_id, interpolated)]} covering
    only ``frames_wanted``, with missed detections interpolated back in.
    """
    out = {f: [] for f in frames_wanted}
    wanted = set(frames_wanted)
    for tid, obs in observations.items():
        idx = sorted(obs)
        for f in idx:
            if f in wanted:
                b, p, s = obs[f]
                out[f].append((b, p, s, tid, False))
        # bridge gaps between consecutive observations
        for a, b_ in zip(idx, idx[1:]):
            gap = b_ - a
            if gap <= 1 or gap > MAX_INTERP_GAP:
                continue
            box_a, poly_a, score_a = obs[a]
            box_b, poly_b, score_b = obs[b_]
            for f in range(a + 1, b_):
                if f not in wanted:
                    continue
                t = (f - a) / float(gap)
                box = [box_a[i] + (box_b[i] - box_a[i]) * t for i in range(4)]
                ref_box, ref_poly = (box_a, poly_a) if t < 0.5 else (box_b, poly_b)
                poly = polygon_for(box, ref_box, ref_poly)
                score = score_a + (score_b - score_a) * t
                out[f].append((box, poly, score, tid, True))
    return out


# ------------------------------------------------------------------- saving

class Writer:
    def __init__(self):
        self.saved = 0
        self.empty_saved = 0
        self.interpolated = 0
        self.by_group = {}

    def allow_empty(self):
        return self.empty_saved < EMPTY_FRAME_SHARE * max(50, self.saved)

    def save(self, group, stem, frame, records, sharpness, meta):
        import cv2
        img_dir = os.path.join(OUT, 'images', group)
        sug_dir = os.path.join(OUT, 'suggestions', group)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(sug_dir, exist_ok=True)
        h, w = frame.shape[:2]
        cv2.imwrite(os.path.join(img_dir, stem + '.jpg'), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        boxes = []
        for box, poly, score, tid, interp in records:
            boxes.append({
                'x1': round(float(box[0]), 1), 'y1': round(float(box[1]), 1),
                'x2': round(float(box[2]), 1), 'y2': round(float(box[3]), 1),
                'label': 0, 'confidence': round(float(score), 4), 'source': 'model',
                'track_id': int(tid), 'interpolated': bool(interp),
                'polygon': ([[round(float(x), 1), round(float(y), 1)] for x, y in poly]
                            if poly else None),
            })
            if interp:
                self.interpolated += 1
        payload = {'width': w, 'height': h, 'sharpness': round(float(sharpness), 1),
                   'boxes': boxes}
        payload.update(meta)
        with open(os.path.join(sug_dir, stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=1)
        self.saved += 1
        if not records:
            self.empty_saved += 1
        self.by_group[group] = self.by_group.get(group, 0) + 1


# -------------------------------------------------------------------- main

def main():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import cv2, numpy as np, torch
    from ultralytics import YOLO
    from retail_analytics.config import load_config
    from retail_analytics.geometry import build_geometry
    from retail_analytics.models.tracker import build_tracker, parse_tracks
    from retail_analytics.devices import yolo_device

    cfg = load_config()
    log('deadline', datetime.fromtimestamp(DEADLINE).isoformat(timespec='seconds'),
        '(%.1f h from now)' % ((DEADLINE - time.time()) / 3600.0))
    log('output ->', OUT)

    weights = os.path.join(ROOT, cfg.detector.weights)
    model = YOLO(weights, task='segment')
    log('teacher', os.path.basename(weights), 'imgsz', TEACHER_IMGSZ, 'conf', TEACHER_CONF)

    def predict(frame):
        r = model.predict(source=frame, classes=[0], imgsz=TEACHER_IMGSZ,
                          conf=TEACHER_CONF, iou=cfg.detector.iou, max_det=MAX_DET,
                          retina_masks=True, half=True, device=yolo_device(),
                          verbose=False)[0]
        if r.boxes is None or not len(r.boxes):
            return [], [], []
        boxes = r.boxes.xyxy.detach().cpu().numpy().astype(float)
        scores = r.boxes.conf.detach().cpu().numpy().astype(float)
        polys = [None] * len(boxes)
        if r.masks is not None and r.masks.xy is not None:
            xy = r.masks.xy
            for i in range(min(len(boxes), len(xy))):
                p = xy[i]
                if p is not None and len(p) >= 3:
                    step = max(1, len(p) // 120)     # 120 points is plenty for a person
                    polys[i] = [(float(a), float(b)) for a, b in p[::step]]
        return boxes, scores, polys

    writer = Writer()

    # ---- pass 1: the event snapshots. Native 2560x1440, every one has a person
    # in it, and they cost four minutes. Single frames, so no tracking here.
    snap_root = os.path.join(ROOT, 'runs', 'live', 'snapshots')
    snaps = []
    for day in sorted(os.listdir(snap_root)):
        d = os.path.join(snap_root, day)
        if not os.path.isdir(d) or not day.isdigit():
            continue
        for p in sorted(glob.glob(os.path.join(d, '*_full.jpg'))):
            snaps.append((day, p))
    if SNAPSHOT_LIMIT:
        snaps = snaps[:SNAPSHOT_LIMIT]
    log('pass 1: %d full-resolution event snapshots' % len(snaps))
    t0 = time.time()
    for i, (day, path) in enumerate(snaps):
        if past_deadline():
            log('deadline hit during pass 1'); break
        frame = cv2.imread(path)
        if frame is None:
            continue
        boxes, scores, polys = predict(frame)
        records = [(boxes[k], polys[k], scores[k], -1, False) for k in range(len(boxes))]
        sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        writer.save('day_%s' % day, 'snap_%s' % os.path.basename(path)[:-9],
                    frame, records, sharp,
                    {'origin': 'snapshot', 'source_file': os.path.basename(path)})
        if (i + 1) % 200 == 0:
            log('  pass1 %d/%d saved=%d (%.1f fps)' %
                (i + 1, len(snaps), writer.saved, (i + 1) / (time.time() - t0)))
    log('pass 1 done: saved=%d in %.1f min' % (writer.saved, (time.time() - t0) / 60))

    # ---- pass 2: session recordings, in segments, with tracking + gap filling
    sess_root = os.path.join(ROOT, 'runs', 'live', 'sessions')
    videos = []
    for day in sorted(os.listdir(sess_root)):
        d = os.path.join(sess_root, day)
        if not os.path.isdir(d) or not day.isdigit():
            continue
        for p in sorted(glob.glob(os.path.join(d, '*.mp4'))):
            videos.append((day, p))
    log('pass 2: %d recordings over %d days' %
        (len(videos), len({d for d, _ in videos})))

    tracker = build_tracker(cfg.tracker, os.path.join(ROOT, cfg.tracker.reid_weights), fps=25)

    # Round-robin over days so an unusually busy day cannot dominate the set,
    # and a random segment offset each visit so we do not keep sampling 10:00.
    by_day = {}
    for day, p in videos:
        by_day.setdefault(day, []).append(p)
    days = sorted(by_day)
    cursor = {day: 0 for day in days}
    rng = random.Random(42)
    segment_no = 0
    roi_cache = {}

    while not past_deadline() and writer.saved < TARGET_FRAMES:
        progressed = False
        for day in days:
            if past_deadline() or writer.saved >= TARGET_FRAMES:
                break
            vids = by_day[day]
            if not vids:
                continue
            path = vids[cursor[day] % len(vids)]
            cursor[day] += 1
            try:
                cap = cv2.VideoCapture(path)
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                if total <= SEGMENT_FRAMES or w == 0:
                    cap.release(); continue
                start = rng.randrange(0, max(1, total - SEGMENT_FRAMES))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)

                if (w, h) not in roi_cache:
                    roi_cache[(w, h)] = build_geometry(cfg.door, w, h).roi
                roi = roi_cache[(w, h)]

                observations = {}
                frames = {}
                sharpness = {}
                for k in range(SEGMENT_FRAMES):
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        break
                    boxes, scores, polys = predict(frame)
                    det = np.column_stack([
                        np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                        np.asarray(scores, dtype=np.float32).reshape(-1, 1),
                        np.zeros((len(boxes), 1), dtype=np.float32)]) \
                        if len(boxes) else np.empty((0, 6), dtype=np.float32)
                    raw = tracker.update(det, frame)
                    tracks = parse_tracks(raw, len(boxes))
                    if k < WARMUP_FRAMES:
                        continue
                    for t in tracks:
                        i = t.detection_index
                        observations.setdefault(t.track_id, {})[k] = (
                            list(boxes[i]), polys[i], float(scores[i]))
                    if (k - WARMUP_FRAMES) % SAVE_EVERY == 0:
                        s = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                          cv2.CV_64F).var()
                        if s >= MIN_BLUR:
                            frames[k] = frame.copy()
                            sharpness[k] = s
                cap.release()

                filled = fill_track_gaps(observations, sorted(frames))
                group = 'day_%s' % day
                kept = 0
                for k in sorted(frames):
                    records = filled.get(k, [])
                    if records:
                        away = any(outside_roi(r[0], roi) for r in records)
                        # Frames with people away from the door are the reason
                        # we are going full-frame at all -- never subsample them.
                        if not away and (k // SAVE_EVERY) % 2 != 0:
                            continue
                    else:
                        if not writer.allow_empty():
                            continue
                    stem = '%s_%07d' % (os.path.basename(path)[:-4], k + start)
                    writer.save(group, stem, frames[k], records, sharpness[k],
                                {'origin': 'recording',
                                 'source_file': os.path.basename(path),
                                 'frame_index': k + start})
                    kept += 1
                segment_no += 1
                progressed = True
                if segment_no % 10 == 0:
                    left = (DEADLINE - time.time()) / 3600.0
                    log('  seg %d | saved=%d (empty=%d, interp boxes=%d) | %.2f h left'
                        % (segment_no, writer.saved, writer.empty_saved,
                           writer.interpolated, left))
                    json.dump({'saved': writer.saved, 'segments': segment_no,
                               'empty': writer.empty_saved,
                               'interpolated': writer.interpolated,
                               'by_group': writer.by_group,
                               'updated': time.strftime('%H:%M:%S')},
                              open(STATE, 'w'), indent=1)
            except Exception as e:
                log('  segment failed on %s: %s %s' % (os.path.basename(path),
                                                       type(e).__name__, str(e)[:160]))
        if not progressed:
            break

    log('DONE saved=%d frames, %d empty, %d interpolated boxes, %d segments'
        % (writer.saved, writer.empty_saved, writer.interpolated, segment_no))
    log('per group:', json.dumps(writer.by_group, indent=1))
    json.dump({'saved': writer.saved, 'segments': segment_no,
               'empty': writer.empty_saved, 'interpolated': writer.interpolated,
               'by_group': writer.by_group, 'finished': time.strftime('%H:%M:%S')},
              open(STATE, 'w'), indent=1)


def start_watchdog():
    """Belt and braces: the main loop checks the clock between segments, but a
    single hung predict() would sail past it while still holding VRAM. The live
    service starts five minutes after DEADLINE and needs the GPU, so a daemon
    thread kills the process outright if the loop has not already stopped."""
    import threading

    def watch():
        while True:
            if past_deadline():
                log('WATCHDOG: deadline reached, killing process to free the GPU')
                os._exit(0)
            time.sleep(10)

    threading.Thread(target=watch, daemon=True).start()


if __name__ == '__main__':
    start_watchdog()
    try:
        main()
    except Exception:
        log('FATAL\n' + traceback.format_exc())
    finally:
        log('process exiting, GPU released')
        os._exit(0)
