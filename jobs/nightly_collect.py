"""Nightly collector: turn one trading day of raw footage from both cameras
into training material, inside the GPU's free window (21:10 -> 09:45).

A day is ~2 million frames across two cameras. Teacher + tracker run at ~7 fps,
so "process everything" is ~80 GPU-hours per night -- it has to be selective,
and the selection has to be cheap:

1. **Activity scan.** Decode keyframes only (``-skip_frame nokey``, ~one per
   GOP) and run the already-trained nano student on them. Minutes, not hours,
   and it yields a per-second map of when and how many people were in view.
2. **Teacher frames.** A stratified sample from the active spans of both
   cameras goes through the big model at full strength (boxes + polygons) --
   pseudo-labels for the next student, including the cam2 viewpoint it has
   never seen.
3. **Whole-visit windows, both cameras in lockstep.** The sprint goal is one
   trajectory per customer from entry to exit across both cameras, so the unit
   of work is a visit, not an activity span: windows are anchored on the live
   service's entry/exit events, and each window is tracked continuously on
   cam1 and cam2 together -- no tracker reset at 15-minute file boundaries,
   frames of the two cameras matched by server time. Every track's box is
   written on every frame, 25 times a second (the raw material for floor trajectories and
   cross-camera association), alongside ReID crops with absolute timestamps.

   Windows are NOT built from occupancy: the live occupancy counter drifts
   (it reported 5 customers inside while tracking one id), and visit sessions
   only close on their 60-minute cap. Activity is no better an anchor -- both
   cameras see shoppers walking past in the mall gallery through the glass,
   so "someone in view" is true almost all day.

Everything is written as it is produced and checkpointed, so the watchdog can
stop it at any point without losing what was already done.
"""
import os, re, sys, json, glob, time, random, subprocess, threading, traceback
from datetime import datetime, timedelta

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
SMOKE = os.environ.get('RA_NIGHT_SMOKE') == '1'
DAY = os.environ.get('RA_NIGHT_DAY') or datetime.now().strftime('%Y%m%d')
CAMERAS = os.environ.get('RA_NIGHT_CAMERAS', 'cam1,cam2').split(',')
BASE = os.path.join(ROOT, 'data', 'nightly_smoke' if SMOKE else 'nightly')
LOG = os.path.join(HOME, '_nightly_collect.log')

STUDENT = os.path.join(ROOT, 'models', 'student_n_v1.pt')
SEGMENT_SECONDS = 900
FPS = 25.0
SCAN_IMGSZ, SCAN_CONF = 960, 0.35
ACTIVE_PAD_S, MERGE_GAP_S, MIN_SPAN_S = 4.0, 20.0, 8.0
TEACHER_FRAMES_PER_CAM = 4 if SMOKE else 400
TEACHER_EMPTY_SHARE = 0.08
TRK_IMGSZ, TRK_CONF = 1280, 0.30
VISIT_BEFORE_S = 20.0          # window opens this long before an entry event
VISIT_SPAN_S = 15 * 60.0       # ...and covers this long after it (and before an exit)
WINDOW_MAX_S = 60 * 60.0       # longer merged windows are cut into hour blocks
TRAJ_EVERY = 1                 # every frame: the model runs at 25 fps, keep every point it produces
STAGES = os.environ.get('RA_NIGHT_STAGES', 'scan,teacher,visits').split(',')
TRACK_CROP_BUDGET = 140
CROP_TARGET_H = 320
CROP_PAD = 0.08
BREAK_MAX_GAP_S, BREAK_MAX_DIST = 10.0, 260.0
DEADLINE_HHMM = (9, 45)


def deadline():
    if SMOKE:
        return time.time() + 480
    now = datetime.now()
    d = now.replace(hour=DEADLINE_HHMM[0], minute=DEADLINE_HHMM[1], second=0, microsecond=0)
    if d <= now:
        d += timedelta(days=1)
    return d.timestamp()


DEADLINE = deadline()


def past():
    return time.time() >= DEADLINE


def log(*a):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(str(x) for x in a)))


# ------------------------------------------------------------------ inputs

def segments(cam):
    """[(path, absolute start datetime)] for one camera-day.

    Files are named <ffmpeg start HHMMSS>_<index>.mp4 and each index is one
    SEGMENT_SECONDS slice, so a segment's wall-clock start is the prefix plus
    index * 900 s. cam1's first two days predate per-camera folders and live
    directly under runs/raw/<day>."""
    dirs = [os.path.join(ROOT, 'runs', 'raw', cam, DAY)]
    if cam == 'cam1':
        dirs.append(os.path.join(ROOT, 'runs', 'raw', DAY))
    out = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, '*.mp4'))):
            m = re.match(r'(\d{6})_(\d{4})\.mp4$', os.path.basename(p))
            if not m:
                continue
            start = datetime.strptime(DAY + m.group(1), '%Y%m%d%H%M%S') \
                + timedelta(seconds=int(m.group(2)) * SEGMENT_SECONDS)
            out.append((p, start))
    return out


# ------------------------------------------------------------ 1: activity

def scan_segment(ff, student, path, limit=0):
    """People count per keyframe of one segment: [[pts_s, count, [[cx, cy], ...]], ...].

    Frames are streamed from ffmpeg and inferred in batches, never held in
    memory as a whole segment (~450 decoded 720p keyframes would be ~1.2 GB).
    Presentation times come from ffmpeg's showinfo filter on stderr, which is
    only complete once the process exits, so results are keyed by frame order
    first and mapped onto real timestamps at the end."""
    import numpy as np
    from retail_analytics.devices import yolo_device
    w, h = 1280, 720
    size = w * h * 3
    info = os.path.join(HOME, '_kf_showinfo_%d.txt' % os.getpid())
    rows = []
    with open(info, 'w') as errf:
        p = subprocess.Popen([ff, '-hide_banner', '-loglevel', 'info', '-skip_frame', 'nokey', '-i', path,
                              '-fps_mode', 'passthrough', '-vf', 'scale=%d:%d,showinfo' % (w, h),
                              '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                             stdout=subprocess.PIPE, stderr=errf)
        batch = []

        def flush():
            if not batch:
                return
            res = student.predict(source=list(batch), imgsz=SCAN_IMGSZ, conf=SCAN_CONF, classes=[0],
                                  half=True, device=yolo_device(), verbose=False)
            for r in res:
                b = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
                rows.append([len(b), [[round((v[0] + v[2]) / 2), round((v[1] + v[3]) / 2)] for v in b]])
            batch.clear()

        n = 0
        while True:
            buf = p.stdout.read(size)
            if len(buf) < size or (limit and n >= limit):
                break
            batch.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
            n += 1
            if len(batch) == 16:
                flush()
        flush()
        p.stdout.close()
        try:
            p.wait(timeout=30)
        except subprocess.TimeoutExpired:
            p.kill()
    pts = [float(x) for x in re.findall(r'pts_time:([0-9.]+)', open(info, errors='replace').read())]
    os.remove(info)
    return [[round(pts[i], 2) if i < len(pts) else None, c, cs] for i, (c, cs) in enumerate(rows)]


def scan_activity(ff, student, cam):
    out_path = os.path.join(BASE, DAY, 'activity_%s.json' % cam)
    if os.path.exists(out_path):
        return json.load(open(out_path))
    segs = segments(cam)[:2] if SMOKE else segments(cam)
    result = {'camera': cam, 'day': DAY, 'segments': []}
    t0 = time.time()
    for path, start in segs:
        if past():
            break
        samples = [x for x in scan_segment(ff, student, path, limit=40 if SMOKE else 0) if x[0] is not None]
        result['segments'].append({'file': path, 'start': start.isoformat(), 'samples': samples})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(result, open(out_path, 'w'))
    kf = sum(len(s['samples']) for s in result['segments'])
    active = sum(1 for s in result['segments'] for x in s['samples'] if x[1])
    log('scan %s: %d segments, %d keyframes, %d with people, %.1f min'
        % (cam, len(result['segments']), kf, active, (time.time() - t0) / 60))
    return result


def spans(activity):
    """Active spans per segment, scored by how much they can teach.

    People-seconds count, co-present people count triple (they are the only
    source of guaranteed negatives), and movement counts: a salesperson parked
    at the desk for an hour is one identity in one pose."""
    out = []
    for seg in activity['segments']:
        s = seg['samples']
        hits = [x for x in s if x[1] > 0]
        if not hits:
            continue
        cur = None
        for x in hits:
            a, b = max(0.0, x[0] - ACTIVE_PAD_S), x[0] + ACTIVE_PAD_S
            if cur and a - cur[1] <= MERGE_GAP_S:
                cur[1] = b
            else:
                if cur:
                    out.append(cur)
                cur = [a, b, seg]
        out.append(cur)
    scored = []
    for a, b, seg in out:
        if b - a < MIN_SPAN_S:
            continue
        inside = [x for x in seg['samples'] if a <= x[0] <= b]
        people = sum(x[1] for x in inside)
        multi = sum(1 for x in inside if x[1] >= 2)
        motion = 0.0
        for p, q in zip(inside, inside[1:]):
            if p[2] and q[2]:
                cp = [sum(v[0] for v in p[2]) / len(p[2]), sum(v[1] for v in p[2]) / len(p[2])]
                cq = [sum(v[0] for v in q[2]) / len(q[2]), sum(v[1] for v in q[2]) / len(q[2])]
                motion += ((cp[0] - cq[0]) ** 2 + (cp[1] - cq[1]) ** 2) ** 0.5
        score = people + 3 * multi + motion / 50.0
        scored.append({'file': seg['file'], 'seg_start': seg['start'], 'a': round(a, 2), 'b': round(b, 2),
                       'score': round(score, 1), 'people': people, 'multi': multi})
    scored.sort(key=lambda s: -s['score'])
    return scored


# ------------------------------------------------------------ 2: teacher

def teacher_frames(teacher, cam, activity, span_list):
    import cv2
    from retail_analytics.devices import yolo_device
    group = '%s_%s' % (cam, DAY)
    img_dir = os.path.join(BASE, 'annotate', 'images', group)
    sug_dir = os.path.join(BASE, 'annotate', 'suggestions', group)
    if os.path.isdir(sug_dir) and len(os.listdir(sug_dir)) >= TEACHER_FRAMES_PER_CAM:
        return
    os.makedirs(img_dir, exist_ok=True); os.makedirs(sug_dir, exist_ok=True)
    rng = random.Random(hash(group) & 0xffff)
    active = [(seg['file'], seg['start'], x[0]) for seg in activity['segments'] for x in seg['samples'] if x[1] > 0]
    empty = [(seg['file'], seg['start'], x[0]) for seg in activity['segments'] for x in seg['samples'] if x[1] == 0]
    n_empty = int(TEACHER_FRAMES_PER_CAM * TEACHER_EMPTY_SHARE)
    # Spread over the whole day, not bunched in the busiest hour.
    active.sort(key=lambda v: (v[1], v[2]))
    step = max(1, len(active) // max(1, TEACHER_FRAMES_PER_CAM - n_empty))
    picks = active[::step][:TEACHER_FRAMES_PER_CAM - n_empty] + rng.sample(empty, min(n_empty, len(empty)))
    saved = 0
    for path, seg_start, t in picks:
        if past():
            break
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read(); cap.release()
        if not ok:
            continue
        r = teacher.predict(source=frame, classes=[0], imgsz=1536, conf=0.10, iou=0.7, max_det=60,
                            retina_masks=True, half=True, device=yolo_device(), verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes):
            xyxy = r.boxes.xyxy.cpu().numpy(); conf = r.boxes.conf.cpu().numpy()
            xy = r.masks.xy if r.masks is not None else [None] * len(xyxy)
            for i in range(len(xyxy)):
                p = xy[i] if i < len(xy) else None
                poly = None
                if p is not None and len(p) >= 3:
                    st = max(1, len(p) // 120)
                    poly = [[round(float(u), 1), round(float(v), 1)] for u, v in p[::st]]
                x1, y1, x2, y2 = (float(v) for v in xyxy[i])
                boxes.append({'x1': round(x1, 1), 'y1': round(y1, 1), 'x2': round(x2, 1), 'y2': round(y2, 1),
                              'label': 0, 'confidence': round(float(conf[i]), 4), 'source': 'model',
                              'polygon': poly})
        stem = '%s_%07d' % (os.path.basename(path)[:-4], int(t * 1000))   # unique: no overwrites
        cv2.imwrite(os.path.join(img_dir, stem + '.jpg'), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        abs_t = datetime.fromisoformat(seg_start) + timedelta(seconds=t)
        json.dump({'width': frame.shape[1], 'height': frame.shape[0], 'boxes': boxes, 'camera': cam,
                   'origin': 'raw', 'source_file': os.path.basename(path), 'time': abs_t.isoformat()},
                  open(os.path.join(sug_dir, stem + '.json'), 'w'), indent=1)
        saved += 1
    log('teacher %s: %d frames' % (cam, saved))


# ------------------------------------------------------------ 3: visits

def visit_windows():
    """[(start, end, n_events)] from the live service's entry/exit events for DAY.

    An entry opens [t - 20 s, t + 15 min]; an exit covers [t - 15 min, t + 20 s],
    so a visit whose entry the counter missed is still inside a window. Overlaps
    merge -- in busy hours that yields one long continuous block, which is exactly
    right for following people -- and blocks over an hour are cut into hours."""
    path = os.path.join(ROOT, 'runs', 'live', 'entrance_events.jsonl')
    iv = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            if DAY[:4] + '-' + DAY[4:6] + '-' + DAY[6:] not in line:
                continue
            try:
                e = json.loads(line)
                t = datetime.fromisoformat(e['time_local']).replace(tzinfo=None)
            except Exception:
                continue
            if t.strftime('%Y%m%d') != DAY:
                continue
            if e.get('event') == 'exit':
                iv.append([t - timedelta(seconds=VISIT_SPAN_S), t + timedelta(seconds=VISIT_BEFORE_S), 1])
            else:
                iv.append([t - timedelta(seconds=VISIT_BEFORE_S), t + timedelta(seconds=VISIT_SPAN_S), 1])
    iv.sort()
    merged = []
    for a, b, n in iv:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
            merged[-1][2] += n
        else:
            merged.append([a, b, n])
    out = []
    for a, b, n in merged:
        while (b - a).total_seconds() > WINDOW_MAX_S:
            cut = a + timedelta(seconds=WINDOW_MAX_S)
            out.append((a, cut, n))
            a = cut
        out.append((a, b, n))
    return out


class CamStream:
    """Frames of one camera-day addressed by server time, across segment files.

    Segment N of a recording starts exactly at prefix + N * 900 s (the camera's
    GOP is 2 s, so ffmpeg's keyframe-aligned cuts land on whole seconds -- 22500
    frames per file, verified), so a frame's time is its segment start plus
    frame index / 25. The cameras' own on-screen clocks are NOT used: cam1's was
    measured 62 s fast while both streams were in step by server time."""

    def __init__(self, cam):
        import cv2
        self.cv2 = cv2
        self.cam = cam
        self.segs = segments(cam)
        self.cap = None
        self.i = -1

    def open_at(self, t):
        for i, (path, st) in enumerate(self.segs):
            if st <= t < st + timedelta(seconds=SEGMENT_SECONDS):
                self._open(i, int((t - st).total_seconds() * FPS))
                return True
        return False

    def _open(self, i, frame_no):
        if self.cap is not None:
            self.cap.release()
        self.i = i
        self.cap = self.cv2.VideoCapture(self.segs[i][0])
        if frame_no:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, frame_no)
        self.k = int(self.cap.get(self.cv2.CAP_PROP_POS_FRAMES))

    def read(self):
        """(time, frame) or (None, None) at the end of the recorded day."""
        while True:
            ok, frame = self.cap.read()
            if ok:
                t = self.segs[self.i][1] + timedelta(seconds=self.k / FPS)
                self.k += 1
                return t, frame
            if self.i + 1 >= len(self.segs):
                return None, None
            self._open(self.i + 1, 0)

    def close(self):
        if self.cap is not None:
            self.cap.release()


class CamTracker:
    """One camera's tracker, trajectory writer, crop saver and pair bookkeeping
    for the duration of one visit window."""

    def __init__(self, cfg, cam, wdir, w, h):
        from retail_analytics.models.tracker import build_tracker
        self.cfg, self.cam = cfg, cam
        self.dir = os.path.join(wdir, cam)
        os.makedirs(self.dir, exist_ok=True)
        self.tracker = build_tracker(cfg.tracker, os.path.join(ROOT, cfg.tracker.reid_weights), fps=FPS)
        self.traj = open(os.path.join(self.dir, 'tracks.csv'), 'w')
        self.traj.write('time,track_id,x1,y1,x2,y2,score\n')
        self.w, self.h = w, h
        s_ = h / 1440.0
        self.min_h, self.min_w = cfg.classifier.min_box_height * s_, cfg.classifier.min_box_width * s_
        self.ref_scale = 1920.0 / max(1, w)
        self.meta = {'camera': cam, 'day': DAY, 'fps': FPS, 'width': w, 'height': h,
                     'detector': os.path.basename(STUDENT), 'tracks': {}, 'breaks': [], 'completed': False}
        self.seen, self.kept, self.last_seen, self.coexist = {}, {}, {}, set()
        self.fi = 0

    def step(self, t, frame, result):
        import cv2, numpy as np
        from retail_analytics.models.tracker import parse_tracks
        r = result
        if r.boxes is None or not len(r.boxes):
            boxes = np.empty((0, 4), np.float32); masks = None
            det = np.empty((0, 6), np.float32)
        else:
            boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
            scores = r.boxes.conf.cpu().numpy().astype(np.float32)
            masks = r.masks.data.cpu().numpy() if r.masks is not None else None
            det = np.column_stack([boxes, scores.reshape(-1, 1), np.zeros((len(boxes), 1), np.float32)])
        tracks = parse_tracks(self.tracker.update(det, frame), len(boxes))
        ts = t.isoformat(timespec='milliseconds')
        alive = sorted({tr.track_id for tr in tracks})
        for i in range(len(alive)):
            for j in range(i + 1, len(alive)):
                self.coexist.add((alive[i], alive[j]))
        for tr in tracks:
            tid = tr.track_id
            x1, y1, x2, y2 = tr.box
            if self.fi % TRAJ_EVERY == 0:
                self.traj.write('%s,%d,%d,%d,%d,%d,%.3f\n' % (ts, tid, x1, y1, x2, y2, tr.score))
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if tid not in self.seen:
                for other, (f0, ox, oy) in list(self.last_seen.items()):
                    gap = self.fi - f0
                    if other != tid and 5 < gap <= BREAK_MAX_GAP_S * FPS:
                        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 * self.ref_scale
                        if dist <= BREAK_MAX_DIST:
                            self.meta['breaks'].append({'lost_track': other, 'new_track': tid, 'gap_frames': gap,
                                                        'gap_seconds': round(gap / FPS, 2),
                                                        'distance_ref_px': round(dist, 1), 'time': ts})
            self.seen[tid] = self.seen.get(tid, 0) + 1
            self.last_seen[tid] = (self.fi, cx, cy)
            self._maybe_crop(tid, tr, frame, masks, ts)
        self.fi += 1

    def _maybe_crop(self, tid, tr, frame, masks, ts):
        import cv2, numpy as np
        x1, y1, x2, y2 = tr.box
        bw, bh = x2 - x1, y2 - y1
        n = self.seen[tid]
        cadence = 4 if n < 120 else (16 if n < 400 else 60)
        if bh < self.min_h or bw < self.min_w or self.kept.get(tid, 0) >= TRACK_CROP_BUDGET or n % cadence:
            return
        px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
        c1, r1 = max(0, int(x1) - px), max(0, int(y1) - py)
        c2, r2 = min(self.w, int(x2) + px), min(self.h, int(y2) + py)
        crop = frame[r1:r2, c1:c2]
        if crop.size == 0:
            return
        sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if sharp < self.cfg.classifier.min_sharpness:
            return
        mcrop = None
        if masks is not None and tr.detection_index < len(masks):
            m = masks[tr.detection_index]
            if m.shape[:2] != (self.h, self.w):
                m = cv2.resize(m, (self.w, self.h), interpolation=cv2.INTER_NEAREST)
            mcrop = (m[r1:r2, c1:c2] > 0.5).astype(np.uint8) * 255
            if (mcrop > 0).mean() < self.cfg.classifier.min_mask_fill:
                return
        sc = CROP_TARGET_H / max(1, crop.shape[0])
        if sc < 1.0:
            size = (max(1, int(crop.shape[1] * sc)), CROP_TARGET_H)
            crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
            if mcrop is not None:
                mcrop = cv2.resize(mcrop, size, interpolation=cv2.INTER_NEAREST)
        tdir = os.path.join(self.dir, 't%06d' % tid)
        os.makedirs(tdir, exist_ok=True)
        cv2.imwrite(os.path.join(tdir, '%07d.jpg' % self.fi), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if mcrop is not None:
            cv2.imwrite(os.path.join(tdir, '%07d.png' % self.fi), mcrop)
        self.kept[tid] = self.kept.get(tid, 0) + 1
        self.meta['tracks'].setdefault(str(tid), []).append({
            'frame': self.fi, 'time': ts, 'box': [int(v) for v in tr.box],
            'sharpness': round(float(sharp), 1), 'has_mask': mcrop is not None})

    def checkpoint(self, done):
        self.traj.flush()
        self.meta['coexist'] = sorted(self.coexist)
        self.meta['frames_processed'] = self.fi
        self.meta['tracks_seen'] = len(self.seen)
        self.meta['completed'] = done
        json.dump(self.meta, open(os.path.join(self.dir, 'meta.json'), 'w'))

    def close(self, done):
        import shutil
        for tid, n in self.kept.items():
            if n < 2:
                shutil.rmtree(os.path.join(self.dir, 't%06d' % tid), ignore_errors=True)
                self.meta['tracks'].pop(str(tid), None)
        self.checkpoint(done)
        self.traj.close()


def process_window(model, cfg, start, end, n_events):
    """Track one visit window on every camera that recorded it, frame-locked by server time."""
    import shutil
    from retail_analytics.devices import yolo_device
    wid = start.strftime('%H%M%S')
    wdir = os.path.join(BASE, 'visits', DAY, wid)
    wj = os.path.join(wdir, 'window.json')
    if os.path.exists(wj) and json.load(open(wj)).get('completed'):
        return None
    shutil.rmtree(wdir, ignore_errors=True)
    os.makedirs(wdir, exist_ok=True)

    streams, trackers = [], []
    for cam in CAMERAS:
        st = CamStream(cam)
        if st.segs and st.open_at(start):
            streams.append(st)
    if not streams:
        return None
    info = {'day': DAY, 'start': start.isoformat(), 'end': end.isoformat(), 'events': n_events,
            'cameras': [s_.cam for s_ in streams], 'completed': False}
    json.dump(info, open(wj, 'w'), indent=1)

    frames_done = 0
    finished = False
    try:
        while not past():
            batch = []
            for st in streams:
                t, fr = st.read()
                if t is None:
                    break
                batch.append((st, t, fr))
            if len(batch) < len(streams):
                finished = True
                break
            # Keep the cameras in step: a camera that dropped frames on the wire
            # falls behind in server time, so let the lagging one catch up.
            tmax = max(t for _, t, _ in batch)
            for k, (st, t, fr) in enumerate(batch):
                while (tmax - t).total_seconds() > 1.5 / FPS:
                    t2, fr2 = st.read()
                    if t2 is None:
                        break
                    t, fr = t2, fr2
                batch[k] = (st, t, fr)
            if min(t for _, t, _ in batch) >= end:
                finished = True
                break
            if not trackers:
                trackers = [CamTracker(cfg, st.cam, wdir, fr.shape[1], fr.shape[0]) for st, _, fr in batch]
            results = model.predict(source=[fr for _, _, fr in batch], classes=[0], imgsz=TRK_IMGSZ,
                                    conf=TRK_CONF, iou=0.7, max_det=60, retina_masks=True, half=True,
                                    device=yolo_device(), verbose=False)
            for trk, (st, t, fr), r in zip(trackers, batch, results):
                trk.step(t, fr, r)
            frames_done += 1
            if frames_done % 2500 == 0:
                for trk in trackers:
                    trk.checkpoint(False)
                log('   window %s at %s, %d synced frames' % (wid, batch[0][1].strftime('%H:%M:%S'), frames_done))
    finally:
        done = finished and not past()
        for trk in trackers:
            trk.close(done)
        for st in streams:
            st.close()
        info['completed'] = done
        info['synced_frames'] = frames_done
        json.dump(info, open(wj, 'w'), indent=1)
    return {'window': wid, 'minutes': round(frames_done / FPS / 60, 1),
            'tracks': {trk.cam: len(trk.seen) for trk in trackers},
            'crops': {trk.cam: sum(len(v) for v in trk.meta['tracks'].values()) for trk in trackers},
            'breaks': {trk.cam: len(trk.meta['breaks']) for trk in trackers}}


# -------------------------------------------------------------------- main

def main():
    sys.path.insert(0, ROOT); os.chdir(ROOT)
    import shutil
    import imageio_ffmpeg
    from ultralytics import YOLO
    from retail_analytics.config import load_config
    cfg = load_config()
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if not os.path.exists(STUDENT):
        shutil.copy2(os.path.join(ROOT, 'runs', 'student_seg', 'v1_n', 'weights', 'best.pt'), STUDENT)
    log('=== night for %s, cameras %s, deadline %s%s' % (DAY, CAMERAS,
        datetime.fromtimestamp(DEADLINE).isoformat(timespec='minutes'), ' [SMOKE]' if SMOKE else ''))

    student = YOLO(STUDENT, task='segment')
    os.makedirs(os.path.join(BASE, DAY), exist_ok=True)
    activity = {}
    if 'scan' in STAGES or 'teacher' in STAGES:
        for cam in CAMERAS:
            if not segments(cam):
                log('no raw footage for %s on %s' % (cam, DAY))
                continue
            activity[cam] = scan_activity(ff, student, cam)
    if 'teacher' in STAGES and activity:
        teacher = YOLO(os.path.join(ROOT, cfg.detector.weights), task='segment')
        for cam in activity:
            if past():
                break
            teacher_frames(teacher, cam, activity[cam], spans(activity[cam]))
        del teacher
        import torch
        torch.cuda.empty_cache()

    if 'visits' in STAGES:
        windows = visit_windows()
        # Densest windows first: if the night runs out, the material left
        # unprocessed is the quiet stretches, not the rush hour.
        windows.sort(key=lambda w: -w[2] / max(60.0, (w[1] - w[0]).total_seconds()))
        forced = os.environ.get('RA_NIGHT_WINDOW')          # "HH:MM:SS-HH:MM:SS", for checks
        if forced:
            a, b = (datetime.strptime(DAY + x, '%Y%m%d%H:%M:%S') for x in forced.split('-'))
            windows = [(a, b, 0)]
        log('visit windows: %d, %.1f h, %d events' % (len(windows),
            sum((b - a).total_seconds() for a, b, _ in windows) / 3600, sum(n for _, _, n in windows)))
        json.dump([[a.isoformat(), b.isoformat(), n] for a, b, n in windows],
                  open(os.path.join(BASE, DAY, 'visit_windows.json'), 'w'), indent=1)
        for a, b, n in windows:
            if past():
                break
            res = process_window(student, cfg, a, b, n)
            if res:
                log('window %s-%s (%d events): %s' % (a.strftime('%H:%M'), b.strftime('%H:%M'), n, json.dumps(res)))
    log('=== night done')


if __name__ == '__main__':
    def watch():
        while True:
            if past():
                log('WATCHDOG: deadline, freeing the GPU')
                os._exit(0)
            time.sleep(10)
    threading.Thread(target=watch, daemon=True).start()
    try:
        main()
    except Exception:
        log('FATAL\n' + traceback.format_exc())
    finally:
        os._exit(0)
