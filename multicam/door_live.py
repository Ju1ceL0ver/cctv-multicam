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
LIVE = ROOT / 'data' / 'live'
TICK = 0.08
SESSION, OVERLAP = 240, 8
WINDOW_SESSIONS = 30
LAG = 20.0                       # a crossing is told once the window has this many seconds after it
KEEP_WINDOWS = 2
W, H, SAM_IN = 2176, 1224, 1008
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

    def __init__(self, source, out):
        super().__init__(daemon=True)
        self.source, self.out = source, out
        self.latest, self.lock, self.stop = None, threading.Lock(), False

    def run(self):
        import cv2
        file_mode = not str(self.source).startswith('rtsp')
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
                    if k % max(1, int(round(fps * TICK))) == 0:
                        t0 = os.environ.get('RA_SOURCE_T0')       # 07.10: a test on a record -- the record's own clock
                        self.out.put((float(t0) + k / fps if t0 else time.time(), f))
                        while self.out.qsize() > 600:
                            time.sleep(0.05)
                    k += 1
                else:
                    with self.lock:
                        self.latest = (time.time(), f)
            cap.release()
            if file_mode:
                self.out.put(None)
                return
            log('camera: stream ended, reopening')

    def ticker(self):
        """Live: one frame per tick from the newest one."""
        t_next = time.time()
        while not self.stop:
            t_next += TICK
            time.sleep(max(0.0, t_next - time.time()))
            with self.lock:
                f = self.latest
            if f is not None:
                self.out.put((t_next, f[1]))


class Window:
    def __init__(self, t0):
        self.id = time.strftime('w%Y%m%d_%H%M%S', time.localtime(t0))
        self.dir = LIVE / 'cam1' / self.id / 'cam1'
        (self.dir / 'frames').mkdir(parents=True, exist_ok=True)
        (self.dir / 'sam_in').mkdir(exist_ok=True)
        self.t0, self.n, self.sessions, self.told = t0, 0, [], []
        self.times = []
        self.seen = set()
        self.person_of = {}
        self._fr = {}

    def frame(self, k):
        """The tick's frame as RGB (2176 x 1224), a few cached."""
        import cv2
        if k not in self._fr:
            if len(self._fr) > 64:
                self._fr.pop(next(iter(self._fr)))
            f = cv2.imread(str(self.dir / 'frames' / ('%05d.jpg' % k)))
            self._fr[k] = None if f is None else f[:, :, ::-1].copy()
        return self._fr[k]

    def add(self, t, frame):
        import cv2
        g = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(self.dir / 'frames' / ('%05d.jpg' % self.n)), g, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(self.dir / 'sam_in' / ('%05d.jpg' % self.n)), cv2.resize(g, (SAM_IN, SAM_IN), interpolation=cv2.INTER_AREA),
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        self.times.append(t)
        self.n += 1

    def next_session(self):
        s = 0 if not self.sessions else self.sessions[-1][1] - OVERLAP
        if self.n >= s + SESSION:
            return (s, s + SESSION, 0 if not self.sessions else OVERLAP)
        return None


def segment(pred, win, sess):
    """One more session of the window with sam31_segment.label (it keeps the window's earlier sessions)."""
    import sam31_segment as SG
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
    static = DS.static_people(M, owned, person, win.n)
    t_link = time.time() - t_people
    if not hasattr(win, 'foot'):
        win.foot = {}                                   # 08.10: (k, row) -> foot, once (it was redone for the whole window)
    boxes_at = {}
    for p, rs in owned.items():
        for r_ in rs:
            boxes_at.setdefault(int(M.rows[r_, 1]), []).append((r_, M.rows[r_, 4:8].astype(float)))
    rows = {}
    for p, rs in owned.items():
        if (person.get(int(p)) or 0) in static:
            continue
        w = int(person.get(int(p)) or 0)
        for r_ in rs:
            k = int(M.rows[r_, 1])
            x1, y1, x2, y2 = M.rows[r_, 4:8].astype(int)
            if (k, r_) not in win.foot:
                win.foot[(k, r_)] = DS.foot_of(M.crop(r_))
            f = win.foot[(k, r_)]
            t = round(win.times[k], 2) if k < len(win.times) else round(win.t0 + k * TICK, 2)
            if combo is not None:
                win.person_of[(win.id, k, r_)] = '%s:%d' % (win.id, w)
            if combo is not None and (k, r_) not in win.seen:
                win.seen.add((k, r_))
                fr = win.frame(k)
                if fr is not None:
                    m = np.zeros((H, W), np.uint8)
                    c_ = M.crop(r_)
                    m[y1:y1 + c_.shape[0], x1:x1 + c_.shape[1]] = c_[:H - y1, :W - x1]
                    fx, fy = (x1 + f[0], y1 + f[1]) if f else ((x1 + x2) / 2, y2)
                    combo.add((win.id, k, r_), t, fr, m, [fx / W, fy / H])
                    if bank is not None:
                        import door_role
                        bx = M.rows[r_, 4:8].astype(float)
                        iso = all(door_role.box_iou(bx, ob) <= door_role.ISOLATED_IOU for orr, ob in boxes_at.get(k, []) if orr != r_)
                        cut = bx[0] <= 2 or bx[1] <= 2 or bx[2] >= W - 3 or bx[3] >= H - 3
                        sc = door_role.score(int(c_.sum()), iso, cut)
                        if bank.wants('%s:%d' % (win.id, w), sc):
                            import staff_masked as SM
                            bank.offer('%s:%d' % (win.id, w), sc,
                                       lambda: SM.crop_masked(fr[:, :, ::-1], m > 0, bx),
                                       [bx[0] * 1280 / W, bx[1] * 720 / H, bx[2] * 1280 / W, bx[3] * 720 / H])
            rows.setdefault(t, []).append({'w': w, 's': round(float(M.rows[r_, 3]), 3),
                                           'box': [round((x1 + x2) / 2 / 2176, 4), round((y1 + y2) / 2 / 1248, 4),
                                                   round((x2 - x1) / 2176, 4), round((y2 - y1) / 1248, 4)],
                                           'foot': [x1 + f[0], y1 + f[1]] if f else None, 'new': False, 'piece': int(p)})
    win.timing = {'link': round(t_link, 1), 'rows': round(time.time() - t_people - t_link, 1)}
    return [{'s': 0, 't': t, 'p': rows[t]} for t in sorted(rows)]


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


def main(student, heads, rule_name, source=None, minutes=None, stride=None):
    global TICK
    sf = ROOT / 'data' / 'door_v2' / 'side_final.json'
    if stride is None:                             # 08.10: the stride the side setting was chosen for
        stride = json.load(open(sf)).get('stride', 3) if sf.exists() else 3
    TICK = 0.08 * int(stride)                      # the rule was fitted on runs with every stride-th tick (door_micro --stride)
    global SESSION
    SESSION = int(os.environ.get('RA_LIVE_SESSION', max(60, 240 * 3 // int(stride))))   # 08.10: ~58 s of video per session
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
        try:
            bank = door_role.Bank(LIVE / 'role_views' / time.strftime('%Y%m%d_%H%M%S'))
            roles = door_role.Roles(bank, door_role.Worker())
            log('role: staff model loaded (threshold %.2f)' % roles.threshold)
        except Exception as exc:
            roles = bank = None
            log('role: off (%s)' % str(exc)[:200])
    clipper = None
    if os.environ.get('RA_DOOR_CLIPS', '1') == '1':      # 08.10: short clips for the night teacher and the owner
        import door_clips
        clipper = door_clips.Clipper()
    pred = DM.build(student, heads if heads not in ('-', 'none') else None)
    q = queue.Queue()
    src = source or camera_url()
    g = Grabber(src, q)
    g.start()
    if str(src).startswith('rtsp'):
        threading.Thread(target=g.ticker, daemon=True).start()
    log('start: student %s, heads %s, rule %s (threshold %.2f), combo %s, source %s' % (student, heads, rule_name, rule['thr'],
                                                                              'on' if combo else 'off',
                                                                              'camera 1' if str(src).startswith('rtsp') else src))
    win, old = None, []
    t_end = time.time() + 60 * float(minutes) if minutes else None
    told_all = []
    events_path = LIVE / 'events_cam1.jsonl'
    final = False
    while True:
        try:
            item = q.get(timeout=1.0)
        except queue.Empty:
            item = 'idle'
        if final:
            break
        if item is None or (t_end and time.time() > t_end):
            # 07.10: the end of a record -- the last, short session too, and every event without waiting LAG
            if win is None or (not win.sessions and win.n < 2 * OVERLAP) or \
                    (win.sessions and win.n - (win.sessions[-1][1] - OVERLAP) <= 2 * OVERLAP):
                break
            final = True                           # 08.10: also a record shorter than one session
            item = 'idle'
        if item != 'idle':
            t, f = item
            if win is None:
                win = Window(t)
            win.add(t, f)
        if win is None:
            continue
        sess = win.next_session()
        if final:
            s0 = win.sessions[-1][1] - OVERLAP if win.sessions else 0
            sess = (s0, win.n, OVERLAP if win.sessions else 0)
        if sess is None:
            continue
        t0 = time.time()
        segment(pred, win, sess)
        t_sam = time.time() - t0
        ticks = people(win, combo, bank)
        t_reid = time.time() - t0 - t_sam
        t_ev0 = time.time()
        new = []
        if ticks:
            last = ticks[-1]['t']
            alone = combo is not None and (os.environ.get('RA_COMBO_ALONE') == '1' or combo.cfg['lo'] > 1 or combo.side is not None)
            if not alone:
                D.add_io(ticks)
            evs = DR.apply(ticks, rule) if combo is None else combo.events(None if alone else DR.apply(ticks, dict(rule, thr=combo.cfg['lo'])), win.person_of)
            for e in evs:
                if e['t'] > last - LAG and not final:
                    continue
                if any(x['kind'] == e['kind'] and abs(x['t'] - e['t']) <= 2.0 for x in told_all):
                    continue
                e = {k_: v_ for k_, v_ in e.items() if k_ not in ('xy', 'disp')}
                if roles is not None and (e.get('bw') or e.get('w')) is not None:   # 08.10: the role by accumulation
                    who = str(e.get('bw') or e.get('w'))
                    e.update(roles.role(getattr(combo, 'members', {}).get(who, [who])))
                if combo is not None:                      # 08.10: what decided it, for checking and re-running later
                    who = str(e.get('bw') or e.get('w'))
                    e['tracks'] = [str(x) for x in getattr(combo, 'members', {}).get(who, [who])]
                    e['side_cfg'] = (combo.side or {}).get('at')
                e = dict(e, clock=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(e['t'])), window=win.id)
                told_all.append(e); new.append(e)
                if clipper is not None:
                    clipper.event(e)
                with open(events_path, 'a', encoding='utf-8') as fo:
                    fo.write(json.dumps(e) + '\n')
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
        n_people = len({qq['w'] for r in ticks[-SESSION:] for qq in r['p']}) if ticks else 0
        log('session %s %d-%d: SAM %.1f s (%.0f ms/tick), people %.1f s %s, events+role %.1f s, behind the camera %.0f s, people %d, new events %s' % (
            win.id, sess[0], sess[1], t_sam, 1000 * t_sam / (sess[1] - sess[0]), t_reid, getattr(win, 'timing', ''), t_events, behind, n_people,
            ', '.join('%s %s%s' % (e['kind'], e['clock'][11:], ' ' + e['role'] if e.get('role') else '') for e in new) or '-'))
        try:                                                # 08.10: the health log, one line per session
            (LIVE / 'records').mkdir(parents=True, exist_ok=True)
            with open(LIVE / 'records' / ('health_%s.jsonl' % time.strftime('%Y%m%d')), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps({'at': time.strftime('%H:%M:%S'), 'window': win.id, 'ticks': [sess[0], sess[1]],
                                     'sam_ms_tick': round(1000 * t_sam / max(1, sess[1] - sess[0])), 'reid_rule_s': round(t_reid, 1),
                                     'behind_s': round(behind), 'people': n_people, 'events': len(new)}) + '\n')
        except OSError:
            pass
        json.dump({'updated': time.strftime('%H:%M:%S'), 'window': win.id, 'ticks': win.n, 'behind_s': round(behind),
                   'ms_per_tick': round(1000 * t_sam / (sess[1] - sess[0])), 'events_total': len(told_all),
                   'in': sum(e['kind'] == 'in' for e in told_all), 'out': sum(e['kind'] == 'out' for e in told_all)},
                  open(LIVE / 'status_cam1.json', 'w'), indent=1)
        torch.cuda.empty_cache()
        if len(win.sessions) >= WINDOW_SESSIONS:
            old.append(win)
            win = None
            while len(old) > KEEP_WINDOWS:
                shutil.rmtree(old.pop(0).dir.parent, ignore_errors=True)
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
