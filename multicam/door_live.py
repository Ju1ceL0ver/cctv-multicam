"""The live door on camera 1 (06.10.2026): the small SAM 3.1 the way the teacher labelled the door stretches, on the
camera's stream, with a delay of about half a minute (the owner allowed 20-30 s).

camera (RTSP, decoded on the CPU) -> a tick every 0.08 s (12.5 a second, as the teacher's labels): the newest frame
-> frames/ (2176 x 1224, for the ReID crops) and sam_in/ (1008 x 1008, what SAM sees) of the current window ->
every 232 new ticks a SAM session of 240 (8 overlapping, as in sam31_segment) on the small SAM (student encoder,
short heads) -> the window's seams linked and its pieces joined into people by ReID (sam31_reid.py, 0.35) -> door_v2's
run format -> the door rule (door_rule.py) -> every crossing older than LAG seconds and not told yet goes to
data/live/events_cam1.jsonl. A window holds WINDOW_SESSIONS sessions (~9 min), then a new one starts; finished
windows are deleted after KEEP_WINDOWS (the frames are big).

Logs: data/live/live_cam1.log (each session: its seconds per tick, how far behind the camera, people, events) and
data/live/status_cam1.json.

usage (venv_sam3, the card): door_live.py STUDENT_CKPT HEADS_CKPT RULE_NAME [--source VIDEO_FILE] [--minutes M]"""
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIVE = Path(os.environ.get('RA_LIVE_OUT') or ROOT / 'data' / 'live')   # 09.10: tests on records write elsewhere
TICK = 0.08
SESSION, OVERLAP = 240, 8
WINDOW_SESSIONS = int(os.environ.get('RA_LIVE_WSESS', 30))   # 09.10: smaller in tests, to see a window change
LAG = 20.0                       # a crossing is told once the window has this many seconds after it
KEEP_WINDOWS = 3
LEAD = 10.0                      # 08.10: seconds of frames a carried window has before what it may tell
STALE_S = 2.0                    # 09.10: the stream's newest frame older than this -- no ticks until it comes back
DEGRADE_SESSIONS = 1.5           # 08.10: behind by this many sessions -> every second frame until it catches up
W, H, SAM_IN = 2176, 1224, 1008
_SF = Path(__file__).resolve().parent / 'data' / 'door_v2' / 'side_final.json'
SMALL = (json.load(open(_SF)).get('small_masks', False) if _SF.exists() else False) or os.environ.get('RA_LIVE_SMALL') == '1'
if SMALL:                                        # 08.10: masks at 1280 x 720 (all the door needs), kept in memory per window
    W, H = 1280, 720
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS', 'rtsp_transport;tcp')



def _safe_print(line):
    """Windows consoles of background jobs cannot print every character (a '>=' sign killed two runs on 06.10)."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
        print(line.encode(enc, 'replace').decode(enc, 'replace'), flush=True)

def log(text):
    line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), text)
    with open(LIVE / 'live_cam1.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    _safe_print(line)


def camera_url():
    ra = ROOT.parent / 'retail_analytics'
    sys.path.insert(0, str(ra))
    from retail_analytics.config import load_config, load_dotenv
    try:
        load_dotenv()
    except Exception:
        pass
    return load_config().camera.rtsp_url()


class Grabber(threading.Thread):
    """Reads the stream as fast as it comes; at every tick hands over the newest frame (or a file, for a test, at its
    own pace)."""

    def __init__(self, source, out, ring=None):
        super().__init__(daemon=True)
        self.source, self.out, self.ring = source, out, ring
        self.latest, self.lock, self.stop = None, threading.Lock(), False

    def run(self):
        import cv2
        file_mode = not str(self.source).startswith('rtsp')
        pace = os.environ.get('RA_LIVE_PACE') == '1'   # 09.10: a record at the camera's own pace (tests of the lag logic)
        self.w0 = self.tf0 = None
        while not self.stop:
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                log('camera: cannot open, retrying in 5 s')
                time.sleep(5)
                continue
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            k = 0
            while not self.stop:
                ok, f = cap.read()
                if not ok:
                    break
                if file_mode:                        # a recorded file: every second frame of 25 is a tick
                    t0 = os.environ.get('RA_SOURCE_T0')           # 07.10: a test on a record -- the record's own clock
                    tf = float(t0) + k / fps if t0 else time.time()
                    if self.ring is not None:
                        self.ring.offer(tf, f)
                    if k % max(1, int(round(fps * TICK))) == 0:
                        if pace:
                            if self.w0 is None:
                                self.w0, self.tf0 = time.time(), tf
                            time.sleep(max(0.0, self.w0 + (tf - self.tf0) - time.time()))
                        self.out.put((tf, f))
                        while self.out.qsize() > 600 and not pace:
                            time.sleep(0.05)
                    k += 1
                else:
                    now = time.time()
                    with self.lock:
                        self.latest = (now, f)
                    if self.ring is not None:
                        self.ring.offer(now, f)
            cap.release()
            if file_mode:
                self.out.put(None)
                return
            log('camera: stream ended, reopening')

    def ticker(self):
        """Live: one frame per tick from the newest one. 09.10: while the stream is silent (the newest frame older than
        STALE_S) nothing is handed over -- not the same frozen frame again and again for SAM to work on."""
        t_next = time.time()
        stale = False
        while not self.stop:
            t_next += TICK
            time.sleep(max(0.0, t_next - time.time()))
            if time.time() - t_next > 5 * TICK:          # fell behind (the machine was busy): catch up, no burst
                t_next = time.time()
            with self.lock:
                f = self.latest
            now_stale = f is None or time.time() - f[0] > STALE_S
            if now_stale != stale:
                stale = now_stale
                log('camera: %s' % ('no new frames, ticks paused' if stale else 'frames again'))
            if not stale:
                self.out.put((t_next, f[1]))


class Window:
    def __init__(self, t0):
        self.id = time.strftime('w%Y%m%d_%H%M%S', time.localtime(t0))
        self.dir = LIVE / 'cam1' / self.id / 'cam1'
        if self.dir.parent.exists():                    # 08.10: a window of the same name left over (a re-run of a record):
            shutil.rmtree(self.dir.parent, ignore_errors=True)   # its chunks.npz would be appended to
        (self.dir / 'frames').mkdir(parents=True, exist_ok=True)
        (self.dir / 'sam_in').mkdir(exist_ok=True)
        self.t0, self.n, self.sessions, self.told = t0, 0, [], []
        self.times = []
        self.seen = set()
        self.person_of = {}
        self._fr = {}

    def frame(self, k):
        """The tick's frame as RGB (1280 x 720 since 08.10: clips and role views need no more), a few cached."""
        import cv2
        if k not in self._fr:
            if len(self._fr) > 64:
                self._fr.pop(next(iter(self._fr)))
            f = cv2.imread(str(self.dir / 'frames' / ('%05d.jpg' % k)))
            self._fr[k] = None if f is None else f[:, :, ::-1].copy()
        return self._fr[k]

    def add(self, t, frame):
        import cv2
        size = (1280, 720) if PIECES else (W, H)      # ReID (RA_DOOR_PIECES=0) crops its views from full-size frames
        cv2.imwrite(str(self.dir / 'frames' / ('%05d.jpg' % self.n)), cv2.resize(frame, size, interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(str(self.dir / 'sam_in' / ('%05d.jpg' % self.n)), cv2.resize(frame, (SAM_IN, SAM_IN), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        self.times.append(t)
        self.n += 1

    def carry(self, old, c0, move_from=None):
        """08.10: a new window starts with the old one's last frames from tick c0 (copied), so the crossings the old
        window could not tell yet (within LAG of its end) are seen again here, with a lead for SAM to settle; and the
        frames that came after the old window's last session are not lost."""
        for j in range(c0, old.n):
            for sub in ('frames', 'sam_in'):
                a, b = old.dir / sub / ('%05d.jpg' % j), self.dir / sub / ('%05d.jpg' % self.n)
                if move_from is not None and j >= move_from:   # 09.10: the queue after the old window's sessions
                    os.replace(a, b)                           # (nothing of the old window reads it any more)
                else:
                    shutil.copyfile(a, b)
            self.times.append(old.times[j])
            self.n += 1

    def next_session(self):
        s = 0 if not self.sessions else self.sessions[-1][1] - OVERLAP
        if self.n >= s + SESSION:
            return (s, s + SESSION, 0 if not self.sessions else OVERLAP)
        return None


def segment(pred, win, sess):
    """One more session of the window with sam31_segment.label (it keeps the window's earlier sessions)."""
    import sam31_segment as SG
    if SMALL:                                    # 08.10: the window's masks stay in memory; written uncompressed
        if not hasattr(win, 'chunk_state'):
            win.chunk_state = {}
        done = SG.label(pred, win.dir, win.sessions + [sess], None, out_size=(W, H), compress=False, state=win.chunk_state)
        win.sessions = [tuple(x) for x in done]
        json.dump({'day': time.strftime('%Y%m%d', time.localtime(win.t0)), 'cam': 'cam1', 'size': [W, H], 'tick': TICK,
                   'ticks': win.n, 'sessions': [list(x) for x in win.sessions], 'live': True},
                  open(win.dir / 'info.json', 'w'))
        return
    prev = None
    if (win.dir / 'chunks.npz').exists():
        z = np.load(win.dir / 'chunks.npz')
        prev = {'rows': z['rows'].tolist(), 'buf': z['buf'], 'offs': z['offs'].tolist(), 'done': [list(x) for x in win.sessions]}
    done = SG.label(pred, win.dir, win.sessions + [sess], None, prev)
    win.sessions = [tuple(x) for x in done]
    json.dump({'day': time.strftime('%Y%m%d', time.localtime(win.t0)), 'cam': 'cam1', 'size': [W, H], 'tick': TICK,
               'ticks': win.n, 'sessions': [list(x) for x in win.sessions], 'live': True},
              open(win.dir / 'info.json', 'w'))


PIECES = os.environ.get('RA_DOOR_PIECES', '1') == '1'   # 08.10: SAM's own tracks (+ stitching), no ReID subprocess


def people(win, combo=None, bank=None):
    """SAM's tracks of the window (or ReID people, RA_DOOR_PIECES=0) -> door_v2 ticks (t = clock seconds of the tick).
    With combo (door_combo.Live), every person row not seen yet also goes through the side decision (the owner's
    model and/or the shop line); with bank (door_role.Bank), its best views are kept for the role (08.10)."""
    import door_sam as DS
    import sam31_reid as R
    t_people = time.time()
    M = R.Masks(win.dir / 'chunks.npz')
    owned, _ = R.link_seams(M, {int(s): int(sh) for s, e, sh in win.sessions})
    if PIECES:
        person = {int(p): 1000 + int(p) for p in owned}
    else:
        r = subprocess.run([RF, 'sam31_reid.py', '%s/cam1' % win.id, '0.35'], cwd=str(ROOT), capture_output=True, text=True,
                           env=dict(os.environ, RA_S31_ROOT=str(LIVE / 'cam1'), RA_S31_VIDEO='0'))
        if r.returncode or not (win.dir / 'report.json').exists():
            log('ReID failed: %s' % (r.stderr or r.stdout)[-300:])
            return []
        rep = json.load(open(win.dir / 'report.json'))
        person = {int(p): v for p, v in rep['person_of_piece'].items()}
    static = DS.static_people(M, owned, person, win.n, max_px=8.0 * W / 2176)
    t_link = time.time() - t_people
    prof = dict(foot=0.0, frame=0.0, mask=0.0, side=0.0, bank=0.0, n=0)
    if not hasattr(win, 'foot'):
        win.foot = {}                                   # 08.10: (k, row) -> foot, once (it was redone for the whole window)
    boxes_at = {}
    for p, rs in owned.items():
        for r_ in rs:
            boxes_at.setdefault(int(M.rows[r_, 1]), []).append((r_, M.rows[r_, 4:8].astype(float)))
    rows = {}
    order = sorted(((int(M.rows[r_, 1]), int(p), r_) for p, rs in owned.items() for r_ in rs
                    if (person.get(int(p)) or 0) not in static), key=lambda z: z[0])   # 08.10: tick by tick, one frame read each
    for _k, p, r_ in order:
        w = int(person.get(int(p)) or 0)
        if True:
            k = int(M.rows[r_, 1])
            x1, y1, x2, y2 = M.rows[r_, 4:8].astype(int)
            if (k, r_) not in win.foot:
                _t = time.time()
                win.foot[(k, r_)] = DS.foot_of(M.crop(r_))
                prof['foot'] += time.time() - _t
            f = win.foot[(k, r_)]
            t = round(win.times[k], 2) if k < len(win.times) else round(win.t0 + k * TICK, 2)
            if combo is not None:
                win.person_of[(win.id, k, r_)] = '%s:%d' % (win.id, w)
            if combo is not None and (k, r_) not in win.seen:
                win.seen.add((k, r_))
                _t = time.time()
                fr = win.frame(k)
                prof['frame'] += time.time() - _t
                if fr is not None:
                    import cv2
                    _t = time.time()
                    c_ = M.crop(r_)
                    # 08.10: the side is judged at 1280 x 720 (the model works there anyway; the line is drawn there):
                    # full-size masks of every person on every tick took ~40 ms each
                    win.small = (k, fr)
                    sx, sy = 1280.0 / W, 720.0 / H
                    X1, Y1 = int(x1 * sx), int(y1 * sy)
                    cs = c_.astype(np.uint8) if W == 1280 else \
                        cv2.resize(c_.astype(np.uint8), (max(1, int(round(c_.shape[1] * sx))), max(1, int(round(c_.shape[0] * sy)))),
                                   interpolation=cv2.INTER_NEAREST)
                    cs = cs[:720 - Y1, :1280 - X1]

                    def full_mask(cs=cs, X1=X1, Y1=Y1):         # only for a role view (rare); the side works on the crop
                        ms = np.zeros((720, 1280), np.uint8)
                        ms[Y1:Y1 + cs.shape[0], X1:X1 + cs.shape[1]] = cs
                        return ms
                    fx, fy = (x1 + f[0], y1 + f[1]) if f else ((x1 + x2) / 2, y2)
                    prof['mask'] += time.time() - _t
                    _t = time.time()
                    combo.add((win.id, k, r_), t, win.small[1], (cs, X1, Y1), [fx / W, fy / H])
                    prof['side'] += time.time() - _t
                    prof['n'] += 1
                    _t = time.time()
                    if bank is not None:
                        import door_role
                        bx = M.rows[r_, 4:8].astype(float)
                        iso = all(door_role.box_iou(bx, ob) <= door_role.ISOLATED_IOU for orr, ob in boxes_at.get(k, []) if orr != r_)
                        cut = bx[0] <= 2 or bx[1] <= 2 or bx[2] >= W - 3 or bx[3] >= H - 3
                        sc = door_role.score(int(c_.sum() * (2176.0 / W) ** 2), iso, cut)   # areas in the 2176 frame
                        if bank.wants('%s:%d' % (win.id, w), sc):
                            import staff_masked as SM
                            bs = [bx[0] * sx, bx[1] * sy, bx[2] * sx, bx[3] * sy]   # the view from the 1280 x 720 frame
                            bank.offer('%s:%d' % (win.id, w), sc, lambda: SM.crop_masked(fr[:, :, ::-1], full_mask() > 0, bs), bs)
                    prof['bank'] += time.time() - _t
            rows.setdefault(t, []).append({'w': w, 's': round(float(M.rows[r_, 3]), 3),
                                           'box': [round((x1 + x2) / 2 / W, 4), round((y1 + y2) / 2 / (H * 1248 / 1224), 4),
                                                   round((x2 - x1) / W, 4), round((y2 - y1) / (H * 1248 / 1224), 4)],
                                           'foot': [x1 + f[0], y1 + f[1]] if f else None, 'new': False, 'piece': int(p)})
    win.timing = dict({'link': round(t_link, 1), 'rows': round(time.time() - t_people - t_link, 1)},
                      **{k_: round(v_, 1) for k_, v_ in prof.items()})
    return [{'s': 0, 't': t, 'p': rows[t]} for t in sorted(rows)]


def snapshot(win, ticks, e):
    """09.10: the picture of a crossing for the CRM card (as the old counter sent): the window's frame closest to the
    event (1280 x 720, the person boxed) -> data/live/snapshots/<day>/..._full.jpg, the person -> ..._crop.jpg.
    Returns (full, crop) paths, or ('', '') when the person is not on any tick near it."""
    import bisect
    import cv2
    want = {str(x) for x in e.get('tracks') or []} | {str(e.get('bw') or e.get('w'))}
    best = None
    for r in ticks:
        if abs(r['t'] - e['t']) > 3.0:
            continue
        for q in r['p']:
            if '%s:%d' % (win.id, q['w']) in want and (best is None or abs(r['t'] - e['t']) < abs(best[0] - e['t'])):
                best = (r['t'], q['box'])
    if best is None:
        return '', ''
    k = min(max(0, bisect.bisect_left(win.times, best[0] - 0.005)), len(win.times) - 1)
    fr = win.frame(k)
    if fr is None:
        return '', ''
    fr = np.ascontiguousarray(fr[:, :, ::-1])
    sy = 720 * 1248 / 1224                               # people() boxes: x by W, y by H * 1248 / 1224
    cx, cy, bw, bh = best[1][0] * 1280, best[1][1] * sy, best[1][2] * 1280, best[1][3] * sy
    x1, y1, x2, y2 = int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)
    px, py = int(0.15 * bw) + 4, int(0.08 * bh) + 4
    crop = fr[max(0, y1 - py):min(720, y2 + py), max(0, x1 - px):min(1280, x2 + px)].copy()
    cv2.rectangle(fr, (x1, y1), (x2, y2), (0, 255, 0) if e['kind'] == 'in' else (0, 128, 255), 2)
    day = time.strftime('%Y%m%d', time.localtime(e['t']))
    d = LIVE / 'snapshots' / day
    d.mkdir(parents=True, exist_ok=True)
    stem = '%s_%s_%s' % (time.strftime('%H%M%S', time.localtime(e['t'])), e['kind'], str(e.get('w')).replace(':', '_'))
    full, cut = d / (stem + '_full.jpg'), d / (stem + '_crop.jpg')
    cv2.imwrite(str(full), fr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if crop.size:
        cv2.imwrite(str(cut), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(full), str(cut) if crop.size else ''


def save_obs(win, combo):
    """data/live/records/<day>.jsonl: one line per person per tick, once -- the track, where on the frame (window,
    tick, row of chunks.npz), the model's p_inside, the mask's depth past the shop line, its top/height/bottom x, the
    vote and the feet. With these the door can be re-decided by any setting without the GPU."""
    import door_line
    done = getattr(win, 'saved', set())
    rec = LIVE / 'records'
    rec.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, (t, d, top, hh, bx, p, foot) in combo.raw.items():
        if key in done or key[0] != win.id:
            continue
        done.add(key)
        sd = combo.side or {}
        lines.append(json.dumps({'t': round(t, 2), 'win': key[0], 'k': int(key[1]), 'row': int(key[2]),
                                 'track': win.person_of.get(key), 'p': None if p is None else round(float(p), 4),
                                 'd': None if d is None else round(d, 1), 'top': None if top is None else round(top, 1),
                                 'h': None if hh is None else round(hh, 1), 'bx': None if bx is None else round(bx, 1),
                                 'vote': combo.obs.get(key, (None, None))[1], 'foot': foot, 'cfg': sd.get('at')}))
    win.saved = done
    if lines:
        with open(rec / (time.strftime('%Y%m%d', time.localtime(win.t0)) + '.jsonl'), 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')


def prewarm(pred, sf):
    """09.10: the compiled SAM recompiles for every new shape it meets (more people in the frame, longer memory), and a
    240-tick session then costs 0.45-0.85 s per tick instead of 0.22. Warmed on two full sessions of a busy recorded
    door stretch, it keeps 0.22-0.24 s. side_final's "prewarm": the stretch (data/sam31_door/<tag>), "prewarm_sessions"."""
    side = json.load(open(sf)) if sf.exists() else {}
    tag = os.environ.get('RA_LIVE_PREWARM') or side.get('prewarm')
    if not tag or os.environ.get('RA_S31_COMPILE') != '1':
        return
    import sam31_segment as SG
    import sam_speed as SP
    t0 = time.time()
    dst = ROOT / 'data' / 'live' / 'prewarm'
    try:
        n = int(side.get('prewarm_sessions', 2)) * SESSION
        ks = SP.frames(tag, n, int(round(TICK / 0.08)), dst)
        sess = [(s, e, sh) for s, e, sh in __import__('door_micro').sessions_of(len(ks), SESSION, OVERLAP)]
        SG.label(pred, dst, sess, None, out_size=(W, H), compress=False, state={})
        log('prewarm: %d ticks of %s in %.0f s' % (len(ks), tag, time.time() - t0))
    except Exception as exc:                       # never keeps the door from starting
        log('prewarm failed: %s' % str(exc)[:200])
    finally:
        shutil.rmtree(dst, ignore_errors=True)
        import torch
        torch.cuda.empty_cache()


def main(student, heads, rule_name, source=None, minutes=None, stride=None):
    global TICK
    sf = ROOT / 'data' / 'door_v2' / 'side_final.json'
    if stride is None:                             # 08.10: the stride the side setting was chosen for
        stride = json.load(open(sf)).get('stride', 3) if sf.exists() else 3
    TICK = 0.08 * int(stride)                      # the rule was fitted on runs with every stride-th tick (door_micro --stride)
    global SESSION
    # 08.10: ~58 s of video per session; 09.10: side_final's "session" -- at stride 3 sessions of 240 ticks cost 0.45-0.85 s
    # per tick on the card, of 120 ticks 0.23-0.25 s (SAM's per-frame work grows with the session's length)
    side_sess = json.load(open(sf)).get('session') if sf.exists() else None
    SESSION = int(os.environ.get('RA_LIVE_SESSION') or side_sess or max(60, 240 * 3 // int(stride)))
    import door_micro as DM
    import door_rule as DR
    import door_v2 as D
    import torch
    LIVE.mkdir(parents=True, exist_ok=True)
    rule = DR.load(rule_name)
    combo = None
    if os.environ.get('RA_DOOR_COMBO', '1') == '1' and (ROOT / 'data' / 'door_v2' / 'combo_final.json').exists():
        import door_combo
        combo = door_combo.Live()          # the rule + the owner's model + movement + door zone, counted on agreement
    roles = bank = None
    if combo is not None and os.environ.get('RA_DOOR_ROLE', '1') == '1':
        import door_role
        bank = door_role.Bank(LIVE / 'role_views' / time.strftime('%Y%m%d_%H%M%S'))
        try:
            w0 = door_role.Worker()
            log('role: staff model loaded (threshold %.2f, %s)' % (w0.threshold, os.environ.get('RA_ROLE_DEVICE', 'cuda')))
        except Exception as exc:                       # 09.10: the role thread starts it again later
            w0 = None
            log('role: the model did not start (%s), will try again' % str(exc)[:200])
        roles = door_role.Roles(bank, w0, threshold=getattr(w0, 'threshold', None) or 0.45)
    role_q = queue.Queue()

    def role_loop():
        """09.10: the role of each told event (door_role.Roles over the person's accumulated views) and the owner's
        /liveevents answers, in their own thread: the role model runs on the processor (the card is SAM's alone), ~0.7 s
        a view, and must never hold the count. Results -> data/live/records/roles_<day>.jsonl (/liveevents shows the
        last line of each event)."""
        import live_events as LE

        def write(lines):
            (LIVE / 'records').mkdir(parents=True, exist_ok=True)
            with open(LIVE / 'records' / ('roles_%s.jsonl' % time.strftime('%Y%m%d')), 'a', encoding='utf-8') as fr:
                for x in lines:
                    fr.write(json.dumps(x) + '\n')
        last_try = [0.0]
        waiting = []                                   # events told while the role model was down

        def worker_up():
            """09.10: the role model's process alive -- started again (at most once a minute) if it died or never came up"""
            w = roles.worker
            if w is not None and w.p.poll() is None:
                return True
            if time.time() - last_try[0] < 60:
                return False
            last_try[0] = time.time()
            try:
                if w is not None:
                    w.p.kill()
            except Exception:
                pass
            try:
                roles.worker = door_role.Worker()
                log('role: the model (re)started')
                return True
            except Exception as exc:
                roles.worker = None
                log('role: the model did not start: %s' % str(exc)[:200])
                return False
        while True:
            try:
                item = role_q.get(timeout=30)
            except queue.Empty:
                item = ('retry', None)
            if item is None:
                return
            what, arg = item
            try:
                if what in ('event', 'retry'):
                    if what == 'event':
                        waiting.append(arg)
                    if not waiting or not worker_up():
                        continue
                    while waiting:
                        e = waiting[0]
                        out = roles.role(list(e['tracks']))
                        if out.get('error'):               # the model failed on it: keep the event, restart the model
                            last_try[0] = 0.0
                            if roles.worker is not None:
                                try:
                                    roles.worker.p.kill()
                                except Exception:
                                    pass
                            break
                        waiting.pop(0)
                        e.update(out)
                        write([dict(out, key=LE.key_of(e), t=e['t'], kind=e['kind'], was=None, at=time.strftime('%H:%M:%S'))])
                    continue
                if not worker_up():
                    continue
                took = roles.feedback(told_all, LE.load(time.strftime('%Y%m%d')), LE.key_of)
                if took:
                    log('role: %d answers of the owner taken into the staff gallery' % took)
                # 09.10: the day's earlier events again once the owner's answers changed the gallery. Not on the
                # model's own seeds: on the /staff answers a whole-day gallery of the model's sure views did not
                # beat the model (86.9 % = 86.9 %; during the day 87.7 %), the owner's views did (96.6 % vs 90.7 %)
                mode = os.environ.get('RA_ROLE_RESCORE', 'owner')
                ch = roles.rescore(told_all) if (mode == 'always' or (mode == 'owner' and took)) else []
                if ch:
                    write([dict(out, key=LE.key_of(e_), t=e_['t'], kind=e_['kind'], was=was, final=bool(arg),
                                at=time.strftime('%H:%M:%S')) for e_, was, out in ch])
                    log('role: %d earlier events decided again: %s' % (len(ch), ', '.join(
                        '%s %s %s->%s' % (e_['kind'], e_['clock'][11:19], was, out['role']) for e_, was, out in ch[:8])))
            except Exception as exc:
                log('role failed: %s' % str(exc)[:200])
    role_thread = None
    if roles is not None:
        role_thread = threading.Thread(target=role_loop, daemon=True)
        role_thread.start()
    clipper = ring = None
    if os.environ.get('RA_DOOR_CLIPS', '1') == '1':      # 08.10: short clips for the night teacher and the owner
        import door_clips
        ring = door_clips.Ring()                       # the clips at 12.5 fps from the last 4 minutes of the camera
        ring.start()
        clipper = door_clips.Clipper(ring=ring)
    pred = DM.build(student, heads if heads not in ('-', 'none') else None)
    prewarm(pred, sf)
    q = queue.Queue()
    src = source or camera_url()
    g = Grabber(src, q, ring)
    g.start()
    if str(src).startswith('rtsp'):
        threading.Thread(target=g.ticker, daemon=True).start()
    log('start: student %s, heads %s, rule %s (threshold %.2f), combo %s, source %s' % (student, heads, rule_name, rule['thr'],
                                                                              'on' if combo else 'off',
                                                                              'camera 1' if str(src).startswith('rtsp') else src))
    win_old = []
    t_end = time.time() + 60 * float(minutes) if minutes else None
    told_all = []
    events_path = LIVE / 'events_cam1.jsonl'
    live_mode = str(src).startswith('rtsp') or os.environ.get('RA_LIVE_PACE') == '1'
    from concurrent.futures import ThreadPoolExecutor
    post_pool = ThreadPoolExecutor(max_workers=1)     # 08.10: people, events, roles, clips while SAM does the next session
    pending = []

    class Writer(threading.Thread):
        """08.10: the camera's frames go to the window's folder as they come (not while SAM waits); behind by more than
        DEGRADE_SESSIONS sessions, every second frame is skipped (stride x2) until it catches up."""

        def __init__(self):
            super().__init__(daemon=True)
            self.win, self.lock, self.done, self.skipping, self.k = None, threading.Lock(), False, False, 0
            self.busy_until = 0.0

        def idle(self):
            return self.busy_until < time.time()

        def backlog(self):
            w = self.win
            if w is None:
                return 0
            end = w.sessions[-1][1] if w.sessions else 0
            return w.n - end

        def run(self):
            while True:
                item = q.get()
                if item is None:
                    self.done = True
                    return
                t, f = item
                if live_mode and os.environ.get('RA_DEGRADE', '0') == '1':   # 09.10 owner: never skip frames -- off by default
                    bl = self.backlog()            # with hysteresis: skip from 1.5 sessions behind until under half of one
                    behind = bl > DEGRADE_SESSIONS * SESSION or (self.skipping and bl > 0.5 * SESSION)
                    if behind != self.skipping:
                        self.skipping = behind
                        log('behind by %d ticks: %s' % (self.backlog(), 'skipping every second frame' if behind else 'every frame again'))
                    self.k += 1
                    if self.skipping and self.k % 2:
                        continue
                if not live_mode:                       # a record: no further ahead of SAM than two sessions
                    while self.backlog() > 2 * SESSION:
                        time.sleep(0.05)
                with self.lock:
                    self.busy_until = time.time() + 1.0
                    if self.win is None:
                        self.win = Window(t)
                    self.win.add(t, f)

    writer = Writer()
    writer.start()

    def post_session(win, sess, t_sam, final):
        try:
            t0 = time.time()
            ticks = people(win, combo, bank)
            t_reid = time.time() - t0
            t_ev0 = time.time()
            new = []
            if ticks:
                last = win.times[sess[1] - 1]          # 08.10: LAG seconds of frames after a crossing (was: of people)
                tell_from = getattr(win, 'tell_from', None)
                alone = combo is not None and (os.environ.get('RA_COMBO_ALONE') == '1' or combo.cfg['lo'] > 1 or combo.side is not None)
                if not alone:
                    D.add_io(ticks)
                evs = DR.apply(ticks, rule) if combo is None else combo.events(None if alone else DR.apply(ticks, dict(rule, thr=combo.cfg['lo'])), win.person_of)
                for e in evs:
                    if e['t'] > last - LAG and not final:
                        continue
                    if tell_from is not None and e['t'] <= tell_from:   # told by the window before this one
                        continue
                    if any(x['kind'] == e['kind'] and abs(x['t'] - e['t']) <= 2.0 for x in told_all):
                        continue
                    e = {k_: v_ for k_, v_ in e.items() if k_ not in ('xy', 'disp')}
                    if combo is not None:                      # 08.10: what decided it, for checking and re-running later
                        who = str(e.get('bw') or e.get('w'))
                        e['tracks'] = [str(x) for x in getattr(combo, 'members', {}).get(who, [who])]
                        e['side_cfg'] = (combo.side or {}).get('at')
                    e = dict(e, clock=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['t'])), window=win.id)
                    try:                                       # 09.10: the CRM card's pictures; never stops the count
                        e['photo'], e['crop'] = snapshot(win, ticks, e)
                    except Exception as exc:
                        log('snapshot failed: %s' % str(exc)[:200])
                    told_all.append(e); new.append(e)
                    if clipper is not None:
                        clipper.event(e)
                    with open(events_path, 'a', encoding='utf-8') as fo:
                        fo.write(json.dumps(e) + '\n')
                    if roles is not None and e.get('tracks'):
                        role_q.put(('event', e))      # 09.10: the role comes a few seconds later, never holds the count
            if roles is not None:
                if bank is not None:
                    bank.dump()
                role_q.put(('feedback', final))
            t_events = time.time() - t_ev0
            if combo is not None and getattr(combo, 'raw', None):   # 08.10: every observation's votes, no pictures
                save_obs(win, combo)
            if clipper is not None:
                try:
                    if combo is not None and getattr(combo, 'raw', None):
                        clipper.disputes(combo, {x for e in told_all for x in e.get('tracks', [])} | {str(e.get('bw') or e.get('w')) for e in told_all})
                    clipper.random_minute(win.times[0], win.times[-1])
                    made = clipper.cut(win, final)
                    if made:
                        log('clips: %s' % ', '.join(made))
                except Exception as exc:                      # clips never stop the door
                    log('clips failed: %s' % str(exc)[:200])
            behind = time.time() - win.times[sess[1] - 1]
            if getattr(g, 'w0', None) is not None:          # a paced record: how far behind its own clock
                behind = time.time() - (g.w0 + win.times[sess[1] - 1] - g.tf0)
            n_people = len({qq['w'] for r in ticks[-SESSION:] for qq in r['p']}) if ticks else 0
            log('session %s %d-%d: SAM %.1f s (%.0f ms/tick, GPU peak %s GB), people %.1f s %s, events+role %.1f s, behind the camera %.0f s, people %d, new events %s' % (
                win.id, sess[0], sess[1], t_sam, 1000 * t_sam / (sess[1] - sess[0]), getattr(win, 'gpu_peak', {}).get(tuple(sess)), t_reid, getattr(win, 'timing', ''), t_events, behind, n_people,
                ', '.join('%s %s%s' % (e['kind'], e['clock'][11:], ' ' + e['role'] if e.get('role') else '') for e in new) or '-'))
            try:                                                # 08.10: the health log, one line per session
                (LIVE / 'records').mkdir(parents=True, exist_ok=True)
                with open(LIVE / 'records' / ('health_%s.jsonl' % time.strftime('%Y%m%d')), 'a', encoding='utf-8') as fh:
                    fh.write(json.dumps({'at': time.strftime('%H:%M:%S'), 'window': win.id, 'ticks': [sess[0], sess[1]],
                                         'sam_ms_tick': round(1000 * t_sam / max(1, sess[1] - sess[0])), 'reid_rule_s': round(t_reid, 1),
                                         'behind_s': round(behind), 'people': n_people, 'events': len(new),
                                         'skipping': writer.skipping, 'gpu_gb': getattr(win, 'gpu_peak', {}).get(tuple(sess))}) + '\n')
            except OSError:
                pass
            json.dump({'updated': time.strftime('%H:%M:%S'), 'window': win.id, 'ticks': win.n, 'behind_s': round(behind),
                       'ms_per_tick': round(1000 * t_sam / (sess[1] - sess[0])), 'events_total': len(told_all),
                       'in': sum(e['kind'] == 'in' for e in told_all), 'out': sum(e['kind'] == 'out' for e in told_all),
                       'skipping': writer.skipping},
                      open(LIVE / 'status_cam1.json', 'w'), indent=1)
        except Exception as exc:
            import traceback
            log('session post-processing failed: %s' % traceback.format_exc()[-600:])

    final = False
    while True:
        until = os.environ.get('RA_DOOR_UNTIL')        # 08.10: the shop closes -- the last session and its events, then exit
        closing = bool(until) and str(src).startswith('rtsp') and time.strftime('%H:%M') >= until
        if closing and not g.stop:                     # 09.10: no new frames; whatever is queued is still counted
            g.stop = True
            log('closing: the camera is off, %d ticks still to count' % writer.backlog())
        if closing and (not q.empty() or not writer.idle()):
            time.sleep(0.2)                            # the writer is still putting the last frames down
            continue
        ending = writer.done or (t_end and time.time() > t_end) or closing
        with writer.lock:
            win = writer.win
        if win is None:
            if ending:
                break
            time.sleep(0.2)
            continue
        sess = win.next_session()
        if sess is None and ending:
            # the end of a record / the closing: the last, short session too, and every event without waiting LAG
            if (not win.sessions and win.n < 2 * OVERLAP) or (win.sessions and win.n - (win.sessions[-1][1] - OVERLAP) <= 2 * OVERLAP):
                break
            s0 = win.sessions[-1][1] - OVERLAP if win.sessions else 0
            sess = (s0, win.n, OVERLAP if win.sessions else 0)
            final = True
        if sess is None:
            time.sleep(0.2)
            continue
        t0 = time.time()
        torch.cuda.reset_peak_memory_stats()
        segment(pred, win, sess)
        t_sam = time.time() - t0
        win.gpu_peak = getattr(win, 'gpu_peak', {})
        win.gpu_peak[tuple(sess)] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)   # 09.10: GB, the card has 12
        torch.cuda.empty_cache()
        pending = [f_ for f_ in pending if not f_.done()]
        while len(pending) >= 2:                       # back-pressure: the post thread is at most one session behind
            pending[0].result()
            pending = [f_ for f_ in pending if not f_.done()]
        pending.append(post_pool.submit(post_session, win, sess, t_sam, final))
        if len(win.sessions) >= WINDOW_SESSIONS and not final:
            # 08.10: the next window carries this one's last ~LAG + LEAD seconds (and whatever came after its last
            # session); it tells only crossings after what this one could tell
            bound = win.times[win.sessions[-1][1] - 1] - LAG
            c0 = next((j for j in range(win.n) if win.times[j] >= bound - LEAD), win.n)
            c0 = min(c0, win.sessions[-1][1] - OVERLAP)
            with writer.lock:
                nw = Window(win.times[c0] if c0 < win.n else time.time())
                nw.carry(win, c0, move_from=win.sessions[-1][1])
                nw.tell_from = bound
                writer.win = nw
            log('window %s -> %s: carried %d ticks, tells after %s' % (win.id, nw.id, nw.n, time.strftime('%H:%M:%S', time.localtime(bound))))
            win_old.append(win)
            while len(win_old) > KEEP_WINDOWS:
                gone = win_old.pop(0)
                shutil.rmtree(gone.dir.parent, ignore_errors=True)
                if combo is not None and hasattr(combo, 'forget'):
                    combo.forget(gone.id)
        if final:
            break
    for f_ in pending:
        f_.result()
    post_pool.shutdown(wait=True)
    if role_thread is not None:                       # the last roles before leaving
        role_q.put(None)
        role_thread.join(timeout=300)
    g.stop = True
    log('stop: %d events (%d in, %d out)' % (len(told_all), sum(e['kind'] == 'in' for e in told_all), sum(e['kind'] == 'out' for e in told_all)))
    os._exit(0)


if __name__ == '__main__':
    a = sys.argv[1:]
    kw = {}
    for flag in ('--source', '--minutes', '--stride'):
        if flag in a:
            i = a.index(flag)
            kw[flag[2:]] = a[i + 1]
            del a[i:i + 2]
    main(*a, **kw)
