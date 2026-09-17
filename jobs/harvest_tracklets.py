"""Collect dense tracklets: the raw material for replacing OSNet.

What a ReID dataset needs is not boxes but *identity over time*, and this pass
harvests it from footage the shop has already produced, with no hand
labelling at all:

* **Positives are free.** Two crops from the same tracklet are the same person
  by construction.
* **Negatives are free and airtight.** Two tracks alive in the SAME FRAME are
  different people -- there is no way around it. The co-existence matrix
  recorded here is that set of guaranteed negatives.
* **The validation set writes its own shortlist.** Every time a track dies and
  a new one appears nearby moments later, that pair is a candidate identity
  break -- exactly the failure we want to measure. Those pairs are logged so a
  human can later answer "same person?" with one keystroke, which is the
  metric we do not currently have for ReID at all.

Deliberately NOT stored: embeddings. They belong to whichever ReID model
produced them, and the entire point of this exercise is to change that model.
Crops and silhouettes are model-independent and can be re-embedded by anything
later, so those are what goes on disk.

The tracker is built with the footage's REAL frame rate (~12.3 fps), not the
25 its container claims -- otherwise track_buffer is silently doubled and the
motion model integrates against a timestep twice too small.
"""

import os, sys, json, time, glob, traceback
from datetime import datetime, timedelta

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
OUT = os.path.join(ROOT, 'data', 'tracklets')
LOG = os.path.join(HOME, '_tracklets.log')
STATE = os.path.join(HOME, '_tracklets_state.json')

DEADLINE_HHMM = (9, 50)
REAL_FPS = 12.3               # measured: 44163 frames over 3600 wall seconds
DET_IMGSZ = 1280
DET_CONF = 0.25
MAX_DET = 60
CROP_TARGET_H = 320           # keep headroom above the 256 ReID nets want
CROP_PAD = 0.08
DEAD_AFTER = 5                # frames without an update before a track is "gone"
BREAK_MAX_GAP_S = 10.0
BREAK_MAX_DIST = 260.0        # reference px, same scale as stitch_max_distance_px
TRACK_CROP_BUDGET = 140       # per track, spread across its whole life


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


# RA_TRACKLETS_SOURCE=raw reads the verbatim 25 fps / 2560x1440 recordings in
# runs/raw instead of the pipeline's own 12.3 fps session files. Same code, so
# break rates from the two passes are directly comparable -- that comparison is
# how we learn how many identity breaks were caused by frame rate alone.
SOURCE = os.environ.get('RA_TRACKLETS_SOURCE', 'sessions')
if SOURCE == 'raw':
    REAL_FPS = 25.0
    OUT = os.path.join(ROOT, 'data', 'tracklets_raw')
SMOKE = os.environ.get('RA_TRACKLETS_SMOKE') == '1'
if SMOKE:
    OUT = os.path.join(ROOT, 'data', 'tracklets_smoke')
    LOG = os.path.join(HOME, '_tracklets.log')
DEADLINE = time.time() + 210 if SMOKE else deadline_ts()


def past_deadline():
    return time.time() >= DEADLINE


def crop_cadence(seen):
    """How often to keep a crop, as a function of how long the track has lived.

    Dense at the start (a person's first seconds are when the tracker must
    recognise them later), thinning out afterwards -- ReID wants varied poses
    across the whole visit far more than it wants 300 near-identical frames."""
    if seen < 60:
        return 2
    if seen < 200:
        return 8
    return 30


def main():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import cv2, numpy as np
    from ultralytics import YOLO
    from retail_analytics.config import load_config
    from retail_analytics.geometry import build_geometry
    from retail_analytics.models.tracker import build_tracker, parse_tracks
    from retail_analytics.devices import yolo_device

    cfg = load_config()
    log('deadline', datetime.fromtimestamp(DEADLINE).isoformat(timespec='seconds'))
    log('output ->', OUT)
    os.makedirs(OUT, exist_ok=True)

    model = YOLO(os.path.join(ROOT, cfg.detector.weights), task='segment')
    log('source=%s detector imgsz=%d conf=%.2f tracker fps=%.1f'
        % (SOURCE, DET_IMGSZ, DET_CONF, REAL_FPS))

    # Quality gates scale with frame height: the thresholds in configs/default.yaml
    # were chosen for 2560x1440, and these recordings are 1280x720.
    def gates(h):
        s = h / 1440.0
        return (cfg.classifier.min_box_height * s, cfg.classifier.min_box_width * s,
                cfg.classifier.min_sharpness, cfg.classifier.min_mask_fill)

    videos = []
    if SOURCE == 'raw':
        # 15-minute segments. Start at ~12:00 (segment 8) on the busier day and
        # walk forward: the opening hour is mostly an empty shop.
        for day in ('20260913', '20260914'):
            segs = sorted(glob.glob(os.path.join(ROOT, 'runs', 'raw', day, '*.mp4')))
            for p in segs[8:] + segs[:8]:
                videos.append((day, p, os.path.getsize(p)))
    else:
        sess = os.path.join(ROOT, 'runs', 'live', 'sessions')
        for day in sorted(os.listdir(sess)):
            d = os.path.join(sess, day)
            if os.path.isdir(d) and day.isdigit():
                for p in sorted(glob.glob(os.path.join(d, '*.mp4'))):
                    videos.append((day, p, os.path.getsize(p)))
        # Longest files first: continuous tracks are the whole point, and a 1.3 GB
        # hour of footage yields far better tracklets than twenty 40 MB fragments.
        videos.sort(key=lambda v: -v[2])
    if SMOKE:
        videos = videos[:1]
    log('%d recordings available; processing longest first' % len(videos))

    totals = {'videos': 0, 'tracks': 0, 'crops': 0, 'pairs': 0, 'breaks': 0}

    for day, path, _ in videos:
        if past_deadline():
            break
        stem = os.path.basename(path)[:-4]
        vdir = os.path.join(OUT, day, stem)
        done_marker = os.path.join(vdir, 'meta.json')
        if os.path.exists(done_marker):
            try:
                if json.load(open(done_marker, encoding='utf-8')).get('completed'):
                    continue
            except Exception:
                pass
        if os.path.isdir(vdir):
            import shutil as sh
            sh.rmtree(vdir, ignore_errors=True)     # partial run: redo it cleanly
        os.makedirs(vdir, exist_ok=True)
        log('--- %s/%s' % (day, stem))

        tracker = build_tracker(cfg.tracker,
                                os.path.join(ROOT, cfg.tracker.reid_weights),
                                fps=REAL_FPS)
        cap = cv2.VideoCapture(path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not w:
            cap.release(); continue
        min_h, min_w, min_sharp, min_fill = gates(h)
        ref_scale = 1920.0 / max(1, w)      # break distances are in reference px

        meta = {'video': os.path.basename(path), 'day': day, 'width': w, 'height': h,
                'real_fps': REAL_FPS, 'tracks': {}, 'coexist': [], 'breaks': []}
        seen_counts, kept_counts = {}, {}
        last_seen = {}           # track_id -> (frame, cx, cy)
        coexist = set()
        frame_idx = 0

        while not past_deadline():
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            res = model.predict(source=frame, classes=[0], imgsz=DET_IMGSZ,
                                conf=DET_CONF, iou=cfg.detector.iou, max_det=MAX_DET,
                                retina_masks=True, half=True, device=yolo_device(),
                                verbose=False)[0]
            if res.boxes is None or not len(res.boxes):
                det = np.empty((0, 6), dtype=np.float32)
                boxes = np.empty((0, 4)); scores = np.empty((0,)); masks = None
            else:
                boxes = res.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                scores = res.boxes.conf.detach().cpu().numpy().astype(np.float32)
                masks = res.masks.data.detach().cpu().numpy() if res.masks is not None else None
                det = np.column_stack([boxes, scores.reshape(-1, 1),
                                       np.zeros((len(boxes), 1), np.float32)])
            tracks = parse_tracks(tracker.update(det, frame), len(boxes))

            alive = sorted({t.track_id for t in tracks})
            for i in range(len(alive)):
                for j in range(i + 1, len(alive)):
                    coexist.add((alive[i], alive[j]))

            for t in tracks:
                tid = t.track_id
                x1, y1, x2, y2 = t.box
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                # identity-break candidate: a brand new id appearing where one
                # just died is exactly the event we want to measure later
                if tid not in seen_counts:
                    for other, (f0, ox, oy) in list(last_seen.items()):
                        gap = frame_idx - f0
                        if other == tid or gap <= DEAD_AFTER:
                            continue
                        if gap / REAL_FPS > BREAK_MAX_GAP_S:
                            continue
                        dist = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 * ref_scale
                        if dist <= BREAK_MAX_DIST:
                            meta['breaks'].append({
                                'lost_track': other, 'new_track': tid,
                                'gap_frames': gap, 'gap_seconds': round(gap / REAL_FPS, 2),
                                'distance_ref_px': round(dist, 1), 'frame': frame_idx})
                seen_counts[tid] = seen_counts.get(tid, 0) + 1
                last_seen[tid] = (frame_idx, cx, cy)

                bw, bh = x2 - x1, y2 - y1
                if bh < min_h or bw < min_w:
                    continue
                if kept_counts.get(tid, 0) >= TRACK_CROP_BUDGET:
                    continue
                if seen_counts[tid] % crop_cadence(seen_counts[tid]) != 0:
                    continue

                px, py = int(bw * CROP_PAD), int(bh * CROP_PAD)
                cx1, cy1 = max(0, int(x1) - px), max(0, int(y1) - py)
                cx2, cy2 = min(w, int(x2) + px), min(h, int(y2) + py)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue
                sharp = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY),
                                      cv2.CV_64F).var()
                if sharp < min_sharp:
                    continue

                mask_crop = None
                if masks is not None and t.detection_index < len(masks):
                    m = masks[t.detection_index]
                    if m.shape[:2] != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_crop = (m[cy1:cy2, cx1:cx2] > 0.5).astype(np.uint8) * 255
                    fill = float((mask_crop > 0).mean())
                    if fill < min_fill:
                        continue

                scale = CROP_TARGET_H / max(1, crop.shape[0])
                if scale < 1.0:
                    nw, nh = max(1, int(crop.shape[1] * scale)), CROP_TARGET_H
                    crop = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
                    if mask_crop is not None:
                        mask_crop = cv2.resize(mask_crop, (nw, nh),
                                               interpolation=cv2.INTER_NEAREST)

                tdir = os.path.join(vdir, 't%06d' % tid)
                os.makedirs(tdir, exist_ok=True)
                cv2.imwrite(os.path.join(tdir, '%07d.jpg' % frame_idx), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                if mask_crop is not None:
                    cv2.imwrite(os.path.join(tdir, '%07d.png' % frame_idx), mask_crop)
                kept_counts[tid] = kept_counts.get(tid, 0) + 1
                meta['tracks'].setdefault(str(tid), []).append({
                    'frame': frame_idx, 'box': [int(v) for v in t.box],
                    'score': round(float(t.score), 3),
                    'sharpness': round(float(sharp), 1),
                    'has_mask': mask_crop is not None})
            frame_idx += 1
            if frame_idx % 2000 == 0:
                log('   %s frame %d | tracks %d | crops %d | %.2f h left'
                    % (stem, frame_idx, len(seen_counts), sum(kept_counts.values()),
                       (DEADLINE - time.time()) / 3600.0))
                meta['coexist'] = sorted(coexist)
                meta['frames_processed'] = frame_idx
                meta['completed'] = False
                with open(os.path.join(vdir, 'meta.json'), 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=1)
        cap.release()

        meta['coexist'] = sorted(coexist)
        meta['frames_processed'] = frame_idx
        meta['completed'] = not past_deadline()
        with open(os.path.join(vdir, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=1)
        # A track with a single crop teaches nothing: no positive pair exists.
        for tid, n in list(kept_counts.items()):
            if n < 2:
                import shutil as sh
                sh.rmtree(os.path.join(vdir, 't%06d' % tid), ignore_errors=True)
        totals['videos'] += 1
        totals['tracks'] += len([k for k, v in kept_counts.items() if v >= 2])
        totals['crops'] += sum(v for v in kept_counts.values() if v >= 2)
        totals['pairs'] += len(coexist)
        totals['breaks'] += len(meta['breaks'])
        log('   done %s: %d frames, %d usable tracks, %d crops, %d guaranteed-negative '
            'pairs, %d break candidates'
            % (stem, frame_idx, totals['tracks'], totals['crops'],
               len(coexist), len(meta['breaks'])))
        with open(STATE, 'w') as f:
            json.dump(dict(totals, updated=time.strftime('%H:%M:%S')), f, indent=1)

    log('DONE %s' % json.dumps(totals))
    with open(STATE, 'w') as f:
        json.dump(dict(totals, finished=time.strftime('%H:%M:%S')), f, indent=1)


if __name__ == '__main__':
    import threading

    def watch():
        while True:
            if past_deadline():
                log('WATCHDOG: deadline reached, freeing the GPU')
                os._exit(0)
            time.sleep(10)

    threading.Thread(target=watch, daemon=True).start()
    try:
        main()
    except Exception:
        log('FATAL\n' + traceback.format_exc())
    finally:
        log('exiting, GPU released')
        os._exit(0)
