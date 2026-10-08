"""SAM 3.1 over a whole stretch of a day, both cameras on the same shop clock, as the teacher of segmentation
and tracking. Per camera:

1. Ticks every 0.08 s (12.5 fps) on the day film's clock (camera 2 shifted by the day's offset); each tick is
   the recorded raw frame the film shows then (the light copy's frame k is the raw file's frame k).
2. Every tick's frame goes to video.mp4 at 2176x1224 (half of 2560 on each side; H.264 CRF 18, a key frame
   every 12 frames so that training reads any stretch quickly) -- the images the labels belong to.
3. The small student (yolo26n-seg) counts people on every tick; a tick is live when someone was seen within
   HOLD seconds on either side. Empty hall is skipped: SAM costs the same on an empty frame.
4. SAM 3.1 (text "person") over the live stretches, sessions of at most SESSION ticks overlapping by OVERLAP
   (card memory grows ~17 MB a frame). The detector's backbone runs one frame at a time
   (batched_grounding_batch_size=1: at the default 16 the card overflowed to 20 GB and it crawled).
   The frame SAM gets is 1008x1008 -- what it squeezes any frame to itself; the masks are stretched back.
5. chunks.npz: every mask as (session start, tick, SAM's number in that session, prob, box) + its bits in the
   box, at 2176x1224; sam31_reid.py links the sessions and joins pieces of one person.

usage (venv_sam3): sam31_segment.py DAY FILM_START [SECONDS] [DEADLINE HH:MM]
  -> data/sam31_seg/DAY_START/camN/{video.mp4, ticks.json, chunks.npz, info.json}"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CKPT = ROOT / 'data' / 'weights' / 'sam3' / 'sam3.1_multiplex.pt'
STUDENT = ROOT / 'runs' / 'student_seg_all_n' / 'weights' / 'best.pt'
TICK = 0.08
W, H = 2176, 1224
SAM_IN = 1008
HOLD = 2.0
SESSION, OVERLAP = int(os.environ.get('RA_S31_SESSION', '240')), 8   # 240 at night; less by day, next to the live counter
PERSON_CONF = 0.3


def tick_frames(day, cam, t0, n):
    """[(segment name, raw frame index, stamp)] for ticks t0 + i * TICK, None where nothing was recorded."""
    import day_movie
    import day_player
    start, _ = day_movie.clock(day, str(ROOT))
    shift = day_movie.offset_of(day, str(ROOT)) if cam == 'cam2' else 0.0
    segs = [s for s in day_player.segments(day, str(ROOT))[cam] if s.ready and not s.broken]
    out = []
    for i in range(n):
        when = start + t0 + i * TICK + shift
        seg = None
        for s in segs:
            if s.start + s.times()[0] <= when + 1e-6:
                seg = s
        if seg is None:
            out.append(None); continue
        times = seg.times()
        k = int(np.searchsorted(times, when - seg.start + 1e-6, 'right')) - 1
        if k < 0 or when - seg.start > times[-1] + 0.5:
            out.append(None); continue
        out.append((seg.name, k, round(seg.start + float(times[k]), 3)))
    return out


def read_ticks(day, cam, ticks):
    """Frames (BGR, full size) of the ticks in order; the raw files read forward only."""
    import cv2
    from rawsource import segments
    paths = {os.path.basename(p): p for p, _ in segments(cam, day)}
    cap, cur, pos, last = None, None, -1, None
    for tk in ticks:
        if tk is None:
            yield None; continue
        name, k, _ = tk
        if name != cur:
            if cap is not None:
                cap.release()
            cap, cur, pos, last = cv2.VideoCapture(paths[name]), name, -1, None    # from the start: seeking lands wrong
        while pos < k:
            if not cap.grab():
                break
            pos += 1
            if pos == k:
                ok, last = cap.retrieve()
                if not ok:
                    last = None
        yield last
    if cap is not None:
        cap.release()


def encoder(path):
    import day_proxy
    return subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                             '-s', '%dx%d' % (W, H), '-r', str(1 / TICK), '-i', '-', '-c:v', 'libx264', '-preset', 'fast',
                             '-crf', '18', '-g', '12', '-bf', '0', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)


def prepare(day, cam, t0, n, out):
    """Video, SAM's input frames and the student's people count of every tick."""
    import cv2
    from ultralytics import YOLO
    student = YOLO(str(STUDENT))
    ticks = tick_frames(day, cam, t0, n)
    sam_in = out / 'sam_in'
    shutil.rmtree(sam_in, ignore_errors=True); sam_in.mkdir(parents=True)
    enc = encoder(out / 'video.mp4')
    black = np.zeros((H, W, 3), np.uint8)
    people, batch, idx = [0] * n, [], []

    def flush():
        if batch:
            for i, r in zip(idx, student.predict(batch, imgsz=1088, conf=PERSON_CONF, verbose=False, half=True)):
                people[i] = 0 if r.boxes is None else int(len(r.boxes))
            batch.clear(); idx.clear()
    for i, f in enumerate(read_ticks(day, cam, ticks)):
        g = black if f is None else cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        enc.stdin.write(g.tobytes())
        cv2.imwrite(str(sam_in / ('%05d.jpg' % i)), cv2.resize(g, (SAM_IN, SAM_IN), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
        if f is not None:
            batch.append(cv2.resize(g, (1088, 612), interpolation=cv2.INTER_AREA)); idx.append(i)
            if len(batch) == 16:
                flush()
    flush()
    enc.stdin.close(); enc.wait()
    hold = int(round(HOLD / TICK))
    seen = np.array(people) > 0
    live = np.zeros(n, bool)
    for i in np.nonzero(seen)[0]:
        live[max(0, i - hold):i + hold + 1] = True
    json.dump({'ticks': ticks, 'people': people, 'live': live.astype(int).tolist()}, open(out / 'ticks.json', 'w'))
    return live


def sessions(live):
    """[(start, stop, shared)] over the live stretches; shared = ticks shared with the session before."""
    out, i, n = [], 0, len(live)
    while i < n:
        if not live[i]:
            i += 1; continue
        j = i
        while j < n and live[j]:
            j += 1
        s, shared = i, 0
        while s < j:
            e = min(j, s + SESSION)
            out.append((s, e, shared))
            if e >= j:
                break
            s, shared = e - OVERLAP, OVERLAP
        i = j
    return out


def refill(out, s, e):
    """SAM's input frames of ticks s..e-1 again, from the camera's video.mp4 (its frame k is tick k); read
    forward from the start -- seeking by frame number lands wrong."""
    import cv2
    (out / 'sam_in').mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(out / 'video.mp4'))
    k = 0
    while k < e and cap.grab():
        if k >= s:
            f = cap.retrieve()[1]
            cv2.imwrite(str(out / 'sam_in' / ('%05d.jpg' % k)), cv2.resize(f, (SAM_IN, SAM_IN), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
        k += 1
    cap.release()


def build():
    import inspect
    import torch
    from sam3.model_builder import build_sam3_predictor
    import sam3.model.decoder as sam3_decoder        # it asks for Flash Attention only; PyTorch on Windows has none
    from torch.nn.attention import sdpa_kernel, SDPBackend
    sam3_decoder.sdpa_kernel = lambda *a, **k: sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    pred = build_sam3_predictor(checkpoint_path=str(CKPT), version='sam3.1', use_fa3=False, max_num_objects=16,
                                compile=os.environ.get('RA_S31_COMPILE') == '1')   # 08.10: torch.compile, "~2x" per Meta
    orig = pred.model.init_state
    known = set(inspect.signature(orig).parameters)
    pred.model.init_state = lambda *a, **kw: orig(*a, **{k: v for k, v in kw.items() if k in known})
    for m in pred.model.modules():
        if hasattr(m, 'batched_grounding_batch_size'):
            m.batched_grounding_batch_size = 1
    return pred


def label(pred, out, sess, deadline, prev=None, out_size=None, compress=True, state=None):
    """state (08.10, the live door): a dict kept by the caller across sessions -- rows/buf/offs live in memory instead of
    being read back from chunks.npz every session; out_size: the masks' frame (W, H) -- 1280 x 720 live; compress=False
    writes chunks.npz uncompressed. The masks of a frame are resized and packed in worker threads while SAM goes on."""
    import cv2
    import torch
    import sam31_video as SV
    from concurrent.futures import ThreadPoolExecutor
    OW, OH = out_size or (W, H)
    if state is not None and 'rows' in state:
        rows, buf, offs, done = state['rows'], state['buf'], state['offs'], state['done']
    else:
        rows, buf, offs, done = [], [], [0], []
    if prev is not None and not rows:            # an unfinished camera: keep what was done, go on after it
        rows = [tuple(r) for r in prev['rows']]
        o = prev['offs']
        buf = [prev['buf'][o[k]:o[k + 1]] for k in range(len(o) - 1)]
        offs = list(o)
        done = [list(x) for x in prev['done']]

    def post(kl, items):
        res = []
        for i, p, m in items:
            m = cv2.resize(m.astype(np.uint8), (OW, OH), interpolation=cv2.INTER_NEAREST).astype(bool)
            pk = SV.pack(m)
            if pk is not None:
                res.append((kl, i, p, pk))
        return res
    pool = ThreadPoolExecutor(max_workers=4)
    finished = {(a, b) for a, b, _ in done}
    for s, e, shared in sess:
        if (s, e) in finished:
            continue
        if deadline and time.strftime('%H:%M') >= deadline:
            break
        sub = out / ('sess_%05d' % s)
        shutil.rmtree(sub, ignore_errors=True); sub.mkdir()
        for k in range(s, e):
            shutil.copy(out / 'sam_in' / ('%05d.jpg' % k), sub / ('%05d.jpg' % (k - s)))
        local = {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sid = pred.handle_request(dict(type='start_session', resource_path=str(sub), offload_video_to_cpu=True))['session_id']
            first = pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
            local[0] = pool.submit(post, 0, SV.masks_of(first.get('outputs', {}) or {}, (SAM_IN, SAM_IN)))
            try:
                for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
                    kl = int(resp.get('frame_index', len(local)))
                    local[kl] = pool.submit(post, kl, SV.masks_of(resp.get('outputs', {}) or {}, (SAM_IN, SAM_IN)))
            except RuntimeError as exc:                 # 07.10: a session with nobody found -- SAM raises instead of
                if 'No points are provided' not in str(exc):    # returning nothing (the small detector, 19.09)
                    raise
                print('session', s, e, 'nobody found', flush=True)
            pred.handle_request(dict(type='close_session', session_id=sid))
        torch.cuda.empty_cache()
        shutil.rmtree(sub, ignore_errors=True)
        for kl in sorted(local):
            for kl_, i, p, ((x1, y1, x2, y2), bits) in local[kl].result():
                rows.append((s, s + kl_, i, p, x1, y1, x2, y2))
                buf.append(bits); offs.append(offs[-1] + len(bits))
        done.append([s, e, shared])
        print('session', s, e, time.strftime('%H:%M:%S'), flush=True)
    pool.shutdown(wait=True)
    save = np.savez_compressed if compress else np.savez
    tmp = out / 'chunks.tmp.npz'
    save(tmp, rows=np.array(rows, dtype=np.float64).reshape(-1, 8),
         buf=np.concatenate(buf) if buf else np.zeros(0, np.uint8), offs=np.array(offs, dtype=np.int64))
    for i in range(100):                 # 08.10: the live door reads chunks.npz in another thread; Windows refuses to
        try:                             # replace a file while it is open -- wait for the reader
            os.replace(tmp, out / 'chunks.npz')
            break
        except PermissionError:
            if i == 99:
                raise
            time.sleep(0.05)
    if state is not None:
        state.update(rows=rows, buf=buf, offs=offs, done=done)
    return done


def main():
    import torch
    day, t0 = sys.argv[1], float(sys.argv[2])
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 900.0
    deadline = sys.argv[4] if len(sys.argv) > 4 else None
    n = int(round(seconds / TICK))
    base = ROOT / 'data' / 'sam31_seg' / ('%s_%05d' % (day, int(t0)))
    pred = None
    for cam in ('cam1', 'cam2'):
        out = base / cam
        prev = None
        if (out / 'info.json').exists():
            old = json.load(open(out / 'info.json'))
            if not old.get('partial'):
                continue
            z = np.load(out / 'chunks.npz')                 # go on with an unfinished camera
            prev = {'rows': z['rows'], 'buf': z['buf'], 'offs': z['offs'], 'done': old['sessions']}
            live = np.array(json.load(open(out / 'ticks.json'))['live'], bool)
            tA = tB = time.time()
            prep_s = old.get('prepare_s')
        else:
            out.mkdir(parents=True, exist_ok=True)
            tA = time.time()
            live = prepare(day, cam, t0, n, out)
            tB = time.time()
            prep_s = round(tB - tA)
        # sessions as planned before: resuming must not re-cut them (SESSION may differ by day)
        if prev:                                    # the rest, starting where the last done session ended
            st = int(prev['done'][-1][1]) - OVERLAP
            lv = live.copy(); lv[:st] = False
            rest = sessions(lv)
            if rest and rest[0][0] == st:
                rest[0] = (st, rest[0][1], OVERLAP)
            sess = [tuple(x) for x in prev['done']] + rest
            shutil.rmtree(out / 'sam_in', ignore_errors=True)
            refill(out, st, n)                       # the input frames of everything left, once
        else:
            sess = sessions(live)
        if pred is None:
            pred = build()
        torch.cuda.reset_peak_memory_stats()
        done = label(pred, out, sess, deadline, prev)
        tC = time.time()
        shutil.rmtree(out / 'sam_in', ignore_errors=True)
        info = {'day': day, 'cam': cam, 'film_start': t0, 'seconds': seconds, 'ticks': n, 'tick': TICK, 'size': [W, H],
                'sam_input': SAM_IN, 'live_ticks': int(live.sum()), 'sessions': done, 'planned_sessions': len(sess),
                'partial': len(done) < len(sess), 'overlap': OVERLAP, 'prepare_s': prep_s, 'sam_s': round(tC - tB) + (old.get('sam_s', 0) if prev else 0),
                's_per_live_tick': round((tC - tB) / max(1, sum(e - s for s, e, _ in done)), 3),
                'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        json.dump(info, open(out / 'info.json', 'w'), indent=1)
        print(info, flush=True)
        if info['partial']:
            break


if __name__ == '__main__':
    main()
