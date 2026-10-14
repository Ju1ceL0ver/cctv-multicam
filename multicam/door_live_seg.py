"""14.10.2026, the owner: the live door on the new system -- doorseg_c (step14604) + its tracker + the side by the
geometry classifier on the owner's zones (zone_lab/lab4: inheritance only with overlapping boxes >= 0.3, door window
2 s), roles by the role model on the processor, the same events file and pictures the CRM sender reads.

Camera 1 at 12.5 ticks a second (door_live.Grabber); every tick the model sees t-5..t+5, so a tick is processed
0.4 s after it arrives. Every second the events of the last minutes are decided again; an event is told once LAG s
of ticks lie after it and never twice. Events -> data/live/events_cam1.jsonl (photo, crop for the CRM), roles ->
data/live/records/roles_<day>.jsonl, health -> data/live/records/health_seg_<day>.jsonl, log -> data/live/live_seg.log.

usage: venv_sam3/python door_live_seg.py [--until HH:MM] [--source FILE]"""
import collections, json, os, queue, sys, threading, time
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'zone_lab'))
LIVE = Path(os.environ.get('RA_LIVE_OUT') or ROOT / 'data' / 'live')
CK = ROOT / 'data/quality_backups/doorseg_c_kaggle_final_20261010/step14604_ce9b0a8498d3.pt'
TICK, LAG, KEEP_S = 0.08, 4.0, 90.0
CFG = dict(SM=0., thr=.6, inh_overlap=.3, door_s=2., pair_s=3.)


def log(msg):
    line = '%s %s' % (time.strftime('%m-%d %H:%M:%S'), msg)
    LIVE.mkdir(parents=True, exist_ok=True)
    with open(LIVE / 'live_seg.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


def main(until='21:00', source=None):
    import torch
    os.chdir(ROOT / 'zone_lab')                      # lab4 reads the teacher rows it learns from here
    import door_seg as S, door_seg_fast as F
    import lab4 as Q
    from door_live_bench import Net, Zones
    from door_seg_track import Tracker, Config
    import door_live as DLV
    import door_role, staff_masked as SM, live_events as LE
    os.environ.setdefault('RA_ROLE_DEVICE', 'cpu')
    torch.backends.cudnn.benchmark = True
    net = S.DoorSeg('cuda'); net.load(torch.load(CK, map_location='cpu', weights_only=False)['model'])
    net.yolo.eval(); net.off.eval(); net.yolo.fuse(verbose=False)
    M = Net(net, 'eager')
    asm = F.GPUChannels(S.background(time.strftime('%Y%m%d'), 'cam1'))
    tracker, zones = Tracker(Config(max_gap=25)), Zones()
    day = time.strftime('%Y%m%d')
    bank = door_role.Bank(LIVE / 'views_seg' / day)
    roles = door_role.Roles(bank, None, threshold=.45)
    role_q = queue.Queue()

    def role_loop():
        while True:
            e = role_q.get()
            if e is None:
                return
            try:
                if roles.worker is None or roles.worker.p.poll() is not None:
                    roles.worker = door_role.Worker()
                out = roles.role(list(e['tracks']))
                stream.roles[e['w']] = out.get('role')
                (LIVE / 'records').mkdir(parents=True, exist_ok=True)
                with open(LIVE / 'records' / ('roles_%s.jsonl' % time.strftime('%Y%m%d', time.localtime(e['t']))), 'a', encoding='utf-8') as fr:
                    fr.write(json.dumps(dict(out, key=LE.key_of(e), t=e['t'], kind=e['kind'], was=None, at=time.strftime('%H:%M:%S'))) + '\n')
            except Exception as exc:
                log('role failed: %s' % str(exc)[:200])
    threading.Thread(target=role_loop, daemon=True).start()

    stream = Stream(LIVE / 'stream')
    stream.shop = zones.m['shop']
    stream.start()
    q = queue.Queue()
    src = source or DLV.camera_url()
    g = DLV.Grabber(src, q, None)
    g.start()
    if str(src).startswith('rtsp'):
        threading.Thread(target=g.ticker, daemon=True).start()
    log('up: doorseg_c %s, side %s, camera %s, until %s' % (CK.name, CFG, 'camera 1' if str(src).startswith('rtsp') else src, until))

    frames = collections.deque()                       # (tick, t, frame 1280x720 BGR)
    recent = collections.deque()                       # (t, frame) for the snapshots, last ~20 s
    rows, seen, told, prev = [], {}, [], np.zeros((S.H, S.W), np.uint8)
    memo = {}
    boxes = {}                                         # (tick, pid) -> box in 1280 x 720
    k_in = k_done = 0
    t_of = {}
    last_ev, ms_sum, ms_n, last_health = time.time(), 0.0, 0, time.time()
    events_path = LIVE / 'events_cam1.jsonl'
    while True:
        if str(src).startswith('rtsp') and time.strftime('%H:%M') >= until:
            log('closing: %s reached' % until)
            break
        try:
            item = q.get(timeout=10)
        except queue.Empty:
            continue
        if item is None:
            break
        t, f = item
        f = cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA)
        frames.append((k_in, t, f)); t_of[k_in] = t; k_in += 1
        recent.append((t, f))
        while recent and recent[0][0] < t - 20:
            recent.popleft()
        if len(frames) < 11:
            continue
        k, tk, fk = frames[5]
        ctx = {o: frames[5 + o][2] for o in range(-5, 6)}
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            pred = M.predict(asm(ctx, prev))
        prev = F.previous_union([p for p in pred if p[2] >= .4])
        pred = [p for p in pred if not on_poster(p[0][:180])]          # 14.10 owner: the advertising stand is not a person
        out = [(pid, p[0][:180]) for pid, p in tracker.update(k, pred) if p[0][:180].any()]
        stream.put(k, fk, out)
        if k % 3 == 0 and out:
            rr = zones.rows(k, out)
            rows += rr
            for r_ in rr:
                boxes[(k, r_[1])] = r_[4:8]
            for pid, m in out:                            # views for the role
                ys, xs = np.nonzero(m)
                bx = [xs.min() * 4, ys.min() * 4, xs.max() * 4 + 4, ys.max() * 4 + 4]
                sc = door_role.score(int(m.sum()) * 16 * (2176 / 1280) ** 2, True, bx[1] <= 2 or bx[3] >= 716)
                if bank.wants(str(pid), sc):
                    full = cv2.resize(m.astype(np.uint8), (S.W, S.H), interpolation=cv2.INTER_NEAREST) > 0
                    bank.offer(str(pid), sc, lambda: SM.crop_masked(fk, full, bx), bx)
        ms_sum += 1000 * (time.perf_counter() - t0); ms_n += 1
        frames.popleft()
        k_done = k
        if time.time() - last_ev >= 1.0:
            last_ev = time.time()
            k_cut = k - int(KEEP_S / TICK)
            rows = [r_ for r_ in rows if r_[0] >= k_cut]
            for key in [x for x in boxes if x[0] < k_cut]:
                del boxes[key]
            for x in [x for x in t_of if x < k_cut]:
                del t_of[x]
            try:
                evs, sides = Q.events(rows, dict(Q.BASE4, memo=memo, **CFG))
                for key_ in [x for x in memo if x[1] < k_cut]:
                    del memo[key_]
                last_side = {}
                for (pp, kk), sv in sides.items():
                    if sv is not None and kk >= last_side.get(pp, (-1, None))[0]:
                        last_side[pp] = (kk, sv)
                stream.sides = {pp: v for pp, (kk, v) in last_side.items()}
            except Exception as exc:
                log('events failed: %s' % str(exc)[:200]); evs = []
            for e in sorted(evs, key=lambda e: e['t']):
                ke = int(round(e['t'] * 12.5))
                if ke > k - LAG / TICK:                     # not final yet
                    continue
                if any(x['kind'] == e['kind'] and str(x['w']) == str(e['w']) and abs(x['tick'] - ke) <= 750 for x in told) or \
                        any(x['kind'] == e['kind'] and abs(x['tick'] - ke) <= 25 for x in told):
                    continue
                tw = t_of.get(ke) or t_of.get(min(t_of, key=lambda x: abs(x - ke))) if t_of else time.time()
                ev = {'kind': e['kind'], 't': float(tw), 'w': str(e['w']), 'tracks': [str(e['w'])], 'tick': ke,
                      'clock': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(tw)), 'engine': 'doorseg_c+zones'}
                try:
                    ev['photo'], ev['crop'] = snapshot(ev, recent, boxes)
                except Exception as exc:
                    log('snapshot failed: %s' % str(exc)[:200]); ev['photo'] = ev['crop'] = ''
                told.append(dict(ev, w=str(e['w'])))
                with open(events_path, 'a', encoding='utf-8') as fo:
                    fo.write(json.dumps({x: v for x, v in ev.items() if x != 'tick'}) + '\n')
                role_q.put(ev)
                stream.events.append((time.time(), ev))
                log('event %s %s (track %s)' % (ev['kind'], ev['clock'][11:], ev['w']))
            told = [x for x in told if x['tick'] >= k_cut]
        if time.time() - last_health > 60:
            last_health = time.time()
            behind = time.time() - tk
            (LIVE / 'records').mkdir(parents=True, exist_ok=True)
            with open(LIVE / 'records' / ('health_seg_%s.jsonl' % time.strftime('%Y%m%d')), 'a', encoding='utf-8') as fh:
                fh.write(json.dumps({'at': time.strftime('%H:%M:%S'), 'ms_tick': round(ms_sum / max(1, ms_n), 1), 'behind_s': round(behind, 1),
                                     'queue': q.qsize(), 'people': len(out)}) + '\n')
            log('health: %.1f ms/tick, behind %.1f s, queue %d' % (ms_sum / max(1, ms_n), behind, q.qsize()))
            ms_sum = ms_n = 0
            bank.dump()
    g.stop = True
    role_q.put(None)
    log('stopped')


POSTER = (514, 35, 559, 158)                      # door_exclusions.POSTER on the 1280 x 720 frame (owner's stand)


def on_poster(m):
    """a mask (1/4 of 1280 x 720) with >= 65 % of its pixels on the advertising stand's box"""
    ys, xs = np.nonzero(m)
    if not len(xs):
        return False
    x1, y1, x2, y2 = (v / 4 for v in POSTER)
    return bool(((xs >= x1) & (xs <= x2) & (ys >= y1) & (ys <= y2)).mean() >= .65)


class Stream(threading.Thread):
    """14.10 owner: the annotated replay -- the newest processed tick drawn (masks green inside / magenta outside / grey
    undecided, number, role, the last crossings) -> <dir>/latest.jpg, 1280 x 720, at most 12.5 a second"""

    def __init__(self, folder):
        super().__init__(daemon=True)
        self.dir = Path(folder); self.dir.mkdir(parents=True, exist_ok=True)
        self.item, self.lock, self.cond = None, threading.Lock(), threading.Event()
        self.sides, self.roles, self.events = {}, {}, collections.deque(maxlen=6)
        self.shop = None

    def put(self, k, frame, out):
        with self.lock:
            self.item = (k, frame, list(out))
        self.cond.set()

    def run(self):
        from PIL import Image, ImageDraw, ImageFont
        def font(n):
            for f in ('arialbd.ttf', 'arial.ttf'):
                try:
                    return ImageFont.truetype(f, n)
                except OSError:
                    pass
            return ImageFont.load_default()
        f18, f30 = font(18), font(30)
        last = 0.0
        while True:
            self.cond.wait(); self.cond.clear()
            if time.time() - last < float(os.environ.get('RA_STREAM_DT', '0.2')):   # 5 pictures a second: the count first
                time.sleep(0.02); continue
            last = time.time()
            with self.lock:
                item = self.item
            if item is None:
                continue
            try:
                k, fr, out = item
                img = fr.copy(); labels = []
                for pid, m in out:
                    s = self.sides.get(str(pid))
                    if s is None and self.shop is not None:     # undecided yet: the zone rule, for the picture only
                        s = 1 if (m & self.shop).any() else 0
                    col = (60, 200, 60) if s == 1 else (200, 60, 200) if s == 0 else (170, 170, 170)
                    full = cv2.resize(m.astype(np.uint8), (1280, 720), interpolation=cv2.INTER_NEAREST)
                    img[full > 0] = (0.55 * img[full > 0] + 0.45 * np.array(col)).astype(np.uint8)
                    cn, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, cn, -1, col, 2)
                    ys, xs = np.nonzero(m)
                    r = self.roles.get(str(pid))
                    labels.append(((int(xs.min() * 4), max(0, int(ys.min() * 4) - 24)), '#%s %s%s' % (
                        pid, 'ВНУТРИ' if s == 1 else 'СНАРУЖИ' if s == 0 else '?',
                        ' сотр.' if r == 'staff' else ' покуп.' if r == 'customer' else ''), col))
                im = Image.fromarray(img[:, :, ::-1]); d = ImageDraw.Draw(im)
                for (x, y), t, col in labels:
                    l, tp, rr, b = d.textbbox((x, y), t, font=f18); d.rectangle([l - 3, tp - 2, rr + 3, b + 2], fill=(0, 0, 0))
                    d.text((x, y), t, fill=tuple(int(c) for c in col[::-1]), font=f18)
                now = time.time(); i = 0
                for tw, ev in list(self.events):
                    if now - tw > 8:
                        continue
                    r = self.roles.get(ev['w'])
                    t = '%s  #%s  %s  %s' % ('ВХОД' if ev['kind'] == 'in' else 'ВЫХОД', ev['w'], ev['clock'][11:],
                                            'сотрудник' if r == 'staff' else 'покупатель' if r == 'customer' else '')
                    l, tp, rr, b = d.textbbox((16, 14 + 44 * i), t, font=f30); d.rectangle([l - 6, tp - 4, rr + 6, b + 4], fill=(0, 0, 0))
                    d.text((16, 14 + 44 * i), t, fill=(60, 220, 60) if ev['kind'] == 'in' else (255, 150, 30), font=f30); i += 1
                d.text((16, 690), time.strftime('%H:%M:%S') + '  новая дверь (doorseg_c)', fill=(255, 255, 255), font=f18)
                buf = cv2.imencode('.jpg', np.asarray(im)[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 80])[1]
                tmp = self.dir / 'latest.tmp.jpg'
                tmp.write_bytes(buf.tobytes()); os.replace(tmp, self.dir / 'latest.jpg')
            except Exception as exc:
                log('stream failed: %s' % str(exc)[:200])
                time.sleep(1)


def snapshot(ev, recent, boxes):
    """the frame nearest to the crossing with the person boxed -> data/live/snapshots/<day>/<key>_full.jpg, the person -> _crop.jpg"""
    import live_events as LE
    ke, pid = ev['tick'], ev['w']
    cand = [(abs(k - ke), b) for (k, p), b in boxes.items() if str(p) == pid and abs(k - ke) <= 40]
    if not cand or not recent:
        return '', ''
    b = min(cand, key=lambda x: x[0])[1]
    t, fr = min(recent, key=lambda x: abs(x[0] - ev['t']))
    fr = fr.copy()
    x1, y1, x2, y2 = (int(v) for v in b)
    day = time.strftime('%Y%m%d', time.localtime(ev['t']))
    out = LIVE / 'snapshots' / day
    out.mkdir(parents=True, exist_ok=True)
    key = LE.key_of(ev)
    mx, my = int(.12 * (x2 - x1)) + 4, int(.08 * (y2 - y1)) + 4
    crop = fr[max(0, y1 - my):min(720, y2 + my), max(0, x1 - mx):min(1280, x2 + mx)]
    cv2.rectangle(fr, (x1, y1), (x2, y2), (60, 220, 60), 3)
    full, cp = out / (key + '_full.jpg'), out / (key + '_crop.jpg')
    cv2.imwrite(str(full), fr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    cv2.imwrite(str(cp), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return str(full), str(cp)


if __name__ == '__main__':
    a = sys.argv[1:]
    until = a[a.index('--until') + 1] if '--until' in a else '21:00'
    source = a[a.index('--source') + 1] if '--source' in a else None
    main(until, source)
