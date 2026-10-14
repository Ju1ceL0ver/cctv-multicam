"""Demo of the new door model (13.10.2026, the owner: run doorseg_c as if live on a short video, time every stage,
render it with masks, sides, staff/customer and events).

venv_sam3/python door_seg_demo.py CKPT TAG [SECONDS] [START_S]
-> data/demo/<tag>_doorseg.mp4 (1280x720, 12.5 fps) and data/demo/<tag>_doorseg.json (ms per stage).
Viewing only: no tuning, nothing written into the live data, the owner's labels or the CRM."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'demo'
STRIDE = 3


def main(ckpt, tag, seconds=60.0, start=0.0):
    import cv2
    import torch
    import door_seg as S
    import door_seg_fast as F
    import door_combo as C
    import door_role as DR
    import staff_masked as SM
    from door_seg_track import Config, Tracker
    from door_seg_run import contexts
    os.environ.setdefault('RA_ROLE_DEVICE', 'cpu')
    OUT.mkdir(parents=True, exist_ok=True)
    day = tag.split('_')[1]
    video = ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'video.mp4'
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(start * 12.5))
    n_max = int(seconds * 12.5)

    net = S.DoorSeg('cuda')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    net.load(state['model'])
    net.yolo.eval(); net.off.eval()
    torch.set_num_threads(2); cv2.setNumThreads(2)
    net.yolo.fuse(verbose=False)
    asm = F.GPUChannels(S.background(day, 'cam1'))
    tracker, live = Tracker(Config()), C.Live()
    bank = DR.Bank(OUT / ('%s_views' % tag))
    prof = {k: 0.0 for k in ('decode', 'channels', 'model', 'tracker', 'side', 'role_views', 'events_live')}
    first = {}
    frames_rgb, obs, person_of = [], {}, {}       # tick -> [(pid, small mask, box1280)]
    prev = np.zeros((S.H, S.W), np.uint8)

    def reader():
        k = 0
        while k < n_max:
            t = time.perf_counter()
            ok, f = cap.read()
            if not ok:
                return
            f = cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA)
            prof['decode'] += time.perf_counter() - t
            frames_rgb.append(f)
            k += 1
            yield f

    class Cap:                                    # contexts() wants .read()
        it = reader()
        def read(self):
            try:
                return True, next(self.it)
            except StopIteration:
                return False, None

    # warm-up: one pass on a black context so cuDNN choices do not land in the timing
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        z = {o: np.zeros((S.H, S.W, 3), np.uint8) for o in range(-5, 6)}
        for _ in range(3):
            F.predict(net, asm(z, prev), thr=.4)
        torch.cuda.synchronize()
    ticks = 0
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for tick, ctx in contexts(Cap()):
            t = time.perf_counter()
            img = asm(ctx, prev)
            torch.cuda.synchronize(); prof['channels'] += time.perf_counter() - t; t = time.perf_counter()
            pred = F.predict(net, img, thr=.4)
            prof['model'] += time.perf_counter() - t; t = time.perf_counter()
            prev = F.previous_union([p for p in pred if p[2] >= .4])
            out = list(tracker.update(tick, pred))
            prof['tracker'] += time.perf_counter() - t
            rgb = ctx[0][:, :, ::-1].copy()
            rows = []
            for pid, p in out:
                m = p[0][:S.H // 4].astype(np.uint8)
                ys, xs = np.nonzero(m)
                if not len(xs):
                    continue
                box = [xs.min() * 4, ys.min() * 4, xs.max() * 4 + 4, ys.max() * 4 + 4]
                rows.append((pid, m, box, p[2]))
            obs[tick] = rows
            if tick % STRIDE == 0:
                t = time.perf_counter()
                for pid, m, (x1, y1, x2, y2), _ in rows:
                    full = cv2.resize(m, (S.W, S.H), interpolation=cv2.INTER_NEAREST)
                    crop = full[y1:y2, x1:x2]
                    cols = np.nonzero(crop[-1])[0]
                    fx = x1 + cols.mean() if len(cols) else (x1 + x2) / 2
                    key = (tag, tick, pid)
                    live.add(key, tick * .08, rgb, (crop, x1, y1), [fx / S.W, y2 / S.H])
                    person_of[key] = pid
                prof['side'] += time.perf_counter() - t
                t = time.perf_counter()
                for pid, m, (x1, y1, x2, y2), _ in rows:
                    iso = all(DR.box_iou([x1, y1, x2, y2], b) <= DR.ISOLATED_IOU for q, _, b, _ in rows if q != pid)
                    cut = x1 <= 2 or y1 <= 2 or x2 >= S.W - 3 or y2 >= S.H - 3
                    sc = DR.score(int(m.sum() * 16 * (2176 / S.W) ** 2), iso, cut)
                    if bank.wants(pid, sc):
                        full = cv2.resize(m, (S.W, S.H), interpolation=cv2.INTER_NEAREST) > 0
                        bank.offer(pid, sc, lambda: SM.crop_masked(ctx[0], full, [x1, y1, x2, y2]), [x1, y1, x2, y2])
                prof['role_views'] += time.perf_counter() - t
            if tick % 12 == 11:                       # as live: events asked about once a second
                t = time.perf_counter()
                for e in live.events(None, dict(person_of)):
                    first.setdefault((e['kind'], str(e['w']), round(e['t'], 2)), tick)
                prof['events_live'] += time.perf_counter() - t
            ticks += 1
    cap.release()
    t = time.perf_counter()
    events = live.events(None, person_of)
    prof['events'] = time.perf_counter() - t
    t = time.perf_counter()
    roles = DR.Roles(bank, DR.Worker())
    prof['role_worker_start'] = time.perf_counter() - t
    t = time.perf_counter()
    role_of = {}
    for e in events:
        pid = int(str(e['w']).split(':')[-1]) if not isinstance(e['w'], (int, np.integer)) else int(e['w'])
        if pid not in role_of:
            role_of[pid] = roles.role([pid])
        e['pid'], e['role'] = pid, role_of[pid].get('role')
    prof['role_model'] = time.perf_counter() - t
    roles.worker.p.kill()

    # side of every observation (the model's p, the shop line) for the colours
    side, pmod = {}, {}                           # the live track state (flip_events: conf trusted votes in a row)
    sd = live.side or {}
    conf, thr = sd.get('conf', live.cfg['conf']), live.cfg['thr']
    bypid = {}
    for key, pid in person_of.items():
        if key in live.obs:
            bypid.setdefault(pid, []).append((live.obs[key][0], live.obs[key][1], key))
        if key in live.raw:
            pmod[key] = live.raw[key][5]
    for pid, xs in bypid.items():
        st, pend, n = None, None, 0
        for t_, v_, key in sorted(xs):
            v = 1 if v_ >= thr else (0 if v_ <= 1 - thr else None)
            if v is not None and v != st:
                if pend != v:
                    pend, n = v, 0
                n += 1
                if n >= conf:
                    st, pend, n = v, None, 0
            elif v == st:
                pend, n = None, 0
            side[key] = st
    last, lastp = {}, {}
    path = OUT / ('%s_doorseg.mp4' % tag)
    wr = cv2.VideoWriter(str(path.with_suffix('.tmp.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (S.W, S.H))
    banners = []
    for k, f in enumerate(frames_rgb[:ticks]):
        img = f.copy()
        for e in events:
            if abs(e['t'] / .08 - k) < .5:
                banners.append((k, e))
        for pid, m, (x1, y1, x2, y2), s in obs.get(k, []):
            if (tag, k, pid) in side:
                last[pid] = side[(tag, k, pid)]
                lastp[pid] = pmod.get((tag, k, pid))
            ins = last.get(pid)
            col = (60, 200, 60) if ins == 1 else (200, 60, 200) if ins == 0 else (180, 180, 180)
            full = cv2.resize(m, (S.W, S.H), interpolation=cv2.INTER_NEAREST)
            img[full > 0] = (0.55 * img[full > 0] + 0.45 * np.array(col)).astype(np.uint8)
            cnts, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, 2)
            r = role_of.get(pid, {}).get('role')
            lab = '#%d %s%s' % (pid, 'ВНУТРИ' if ins == 1 else 'СНАРУЖИ' if ins == 0 else '?',
                                ' сотр.' if r == 'staff' else ' покуп.' if r == 'customer' else '') + \
                ('  p=%.2f' % lastp[pid] if lastp.get(pid) is not None else '')
            img = put(img, lab, (x1, max(0, y1 - 26)), col)
        live_b = [e for k0, e in banners if k - k0 < 25]
        for i, e in enumerate(live_b):
            r = e.get('role')
            txt = '%s  #%d  %s' % ('ВХОД' if e['kind'] == 'in' else 'ВЫХОД', e['pid'],
                                   'сотрудник' if r == 'staff' else 'покупатель' if r == 'customer' else '')
            img = put(img, txt, (20, 20 + 46 * i), (60, 220, 60) if e['kind'] == 'in' else (255, 140, 30), 34, True)
        img = put(img, 'doorseg_c  %.1f s' % (k * .08), (S.W - 260, S.H - 40), (255, 255, 255), 24)
        wr.write(img)
    wr.release()
    os.replace(path.with_suffix('.tmp.mp4'), path)
    lat = []
    for e in events:
        k0 = first.get((e['kind'], str(e['w']), round(e['t'], 2)))
        if k0 is not None:                            # reported at tick k0, which needs 5 future frames; + its compute
            lat.append(round((k0 + 5) * .08 - e['t'], 2))
    ms = {k: round(v / max(1, ticks) * 1000, 2) for k, v in prof.items()}
    per_tick = sum(ms[k] for k in ('decode', 'channels', 'model', 'tracker', 'side', 'role_views', 'events_live'))
    rep = dict(tag=tag, start=start, ticks=ticks, ms_per_tick=ms, ms_per_tick_live=round(per_tick, 2),
               fps=round(1000 / per_tick, 1), events=[{k: e.get(k) for k in ('kind', 't', 'pid', 'role')} for e in events],
               event_delay_s=lat, roles={str(k): v for k, v in role_of.items()}, tracks=len({p for r in obs.values() for p, *_ in r}),
               role_once_s={k: round(prof[k], 2) for k in ('events', 'role_worker_start', 'role_model')})
    (OUT / ('%s_doorseg.json' % tag)).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
    print(json.dumps(rep, ensure_ascii=False, default=float))


def put(img, text, xy, col, size=22, box=False):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img[:, :, ::-1])
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype('arialbd.ttf', size)
    except OSError:
        font = ImageFont.load_default()
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=font)
    d.rectangle([l - 4, t - 3, r + 4, b + 3], fill=(0, 0, 0))
    d.text((x, y), text, fill=tuple(int(c) for c in col[::-1]), font=font)
    return np.asarray(im)[:, :, ::-1].copy()


if __name__ == '__main__':
    a = sys.argv[1:]
    main(a[0], a[1], float(a[2]) if len(a) > 2 else 60.0, float(a[3]) if len(a) > 3 else 0.0)
