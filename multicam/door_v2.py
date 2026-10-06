"""Entry and exit by the v2 model on camera 1 alone, against the owner's /door answers (03.10.2026).

  run DAY CKPT [PAD_S]  every counter event of the day +- PAD_S (default 25 s) seconds, merged into stretches; each stretch
                        goes through the model + tracker (v2_track's rules, one camera) tick by tick (0.08 s), from the
                        raw recording on the film's clock -> data/door_v2/<day>_<run name>.jsonl.gz: per tick the people
                        (track's world number, person-ness, box, floor place, zone probabilities, state)
  The stretches are where the counter saw something; the counter finds ~95 % of real crossings (section 30), so
  nearly every real crossing is inside them -- but a false crossing of the model outside them is not counted.

Backgrounds as in training: the long one is the 15-minute file's empty hall (nearest file of the day), the recent
one the median of 15 frames of the 5 minutes before, people cut out (here by the model's own masks).

usage (cctv_base, the card): door_v2.py run 20260918 runs/v2_m_clips/best.pt"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'door_v2'
TICK = 0.08
BG_EVERY = 20.0         # s between samples of the recent hall ...
BG_SPAN = 300.0         # ... over the last 5 minutes
BG_W, BG_H = 1088, 612


def stretches(day, pad=25.0):
    import day_movie
    import door_review
    start, _ = day_movie.clock(day, str(ROOT))
    ts = sorted(e['unix_ms'] / 1000 - start for e in door_review.day_events(day, str(ROOT)))
    out = []
    for t in ts:
        a, b = t - pad, t + pad
        if out and a <= out[-1][1]:
            out[-1][1] = b
        else:
            out.append([a, b])
    return out


class Raw:
    """Raw frames of camera 1 by tick, read forward; the first frame of a stretch is found by its time stamp."""

    def __init__(self, day, cam='cam1'):
        import day_player
        from rawsource import segments
        self.day, self.cam = day, cam
        self.paths = {os.path.basename(p): p for p, _ in segments(cam, day)}
        self.times = {s.name: s.times() for s in day_player.segments(day, str(ROOT))[cam] if s.ready and not s.broken}
        self.cap, self.cur, self.pos = None, None, -1
        self.off = 0                      # frames whose stamp did not match (a wrong frame would be a wrong answer)

    def _open(self, name, k):
        if self.cap is not None:
            self.cap.release()
        self.cap, self.cur, self.pos = cv2.VideoCapture(self.paths[name]), name, -1
        if k > 50:                                                # seek a little before and walk by the stamps
            target = float(self.times[name][k])
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, (target - 3.0) * 1000))
            while self.cap.grab():
                ms = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                j = int(np.searchsorted(self.times[name], ms - 0.005))
                if j >= k - 1:
                    self.pos = j
                    break

    def get(self, tk):
        if tk is None:
            return None
        name, k, _ = tk
        if name != self.cur or k <= self.pos - 1 or k > self.pos + 200:
            self._open(name, k)
        img = None
        if self.pos == k:
            ok, img = self.cap.retrieve()
        while self.pos < k:
            if not self.cap.grab():
                return None
            self.pos += 1
            if self.pos == k:
                ok, img = self.cap.retrieve()
        if img is not None:
            ms = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if abs(ms - float(self.times[name][k])) > 0.03:
                self.off += 1
        return img


def bg_long(day, sec_of_day, cache={}, cam='cam1'):
    """The empty hall of the nearest 15-minute file of the day (data/seg_datasets/backgrounds)."""
    bgs = []
    for p in (ROOT / 'data' / 'seg_datasets' / 'backgrounds').glob('%s_%s_*.jpg' % (day, cam)):
        hms, seg = p.stem.split('_')[2:4]
        bgs.append((int(hms[:2]) * 3600 + int(hms[2:4]) * 60 + int(hms[4:]) + 900 * int(seg) + 450, p))
    p = min(bgs, key=lambda b: abs(b[0] - sec_of_day))[1]
    if p not in cache:
        cache[p] = cv2.imread(str(p))
    return cache[p]


class OneCam:
    """v2_track.Runtime for camera 1 alone (the other camera's slots empty, as in the teacher test)."""

    def __init__(self, model, dev, cam='cam1'):
        import v2_data as VD
        self.cam = cam
        import v2_track as VT
        self.VT = VT
        self.model, self.dev = model, dev
        self.asm = VD.Assembler(dev, getattr(model, 'rgb_size', None))
        self.reset()

    def reset(self):
        self.track, self.track_world, self.hidden_since, self.scene = None, [], [], None
        self.world = self.VT.World()

    def step(self, t, f, tick):
        import torch
        import slot_v2 as V
        import v2_eval
        VT, model, dev = self.VT, self.model, self.dev
        ac = lambda: torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda')
        with torch.no_grad(), ac():
            rgb, bgv, sta, cam_ids = self.asm([f], (self.cam,), False)
            bg_sem = model.body.vit(bgv)[0]
            m = model.maps(rgb, sta, cam_ids, bg_sem, self.asm.last_world)
            tb = model.stack_tracks([self.track], dev, m['mem'].dtype)
            sc = self.scene if self.scene is not None else model.scene.start(cam_ids)
            r = model.decode(m, cam_ids, tb, sc)
            r, _ = model.cross_cameras(r, v2_eval.empty_like(r))
            logits = model.full_masks(m, r)[0]
            pool = model.pooled(m, r, logits.float().sigmoid()[None].to(logits.dtype))
        p = r['obj'][0].float().sigmoid().cpu().numpy()
        pad = r['pad'][0].cpu().numpy()
        state = r['state'][0].float().softmax(-1).cpu().numpy()
        place = r['place'][0].float().cpu().numpy()
        ident = pool['ident'][0].float().cpu().numpy()
        zone = pool['zone'][0].float().softmax(-1).cpu().numpy()
        T = r['tracks']
        ntr = len(self.track_world)
        masks = (logits > 0).cpu().numpy()
        from door_exclusions import excluded
        excluded_slots = np.array([excluded(mask, self.cam) for mask in masks])
        area = masks.reshape(len(masks), -1).sum(1)
        keep = np.zeros(len(p), bool)
        world = [None] * len(p)
        for s_ in range(min(T, ntr)):
            hidden = state[s_, 1] > max(state[s_, 0], state[s_, 2]) and (tick - self.hidden_since[s_]) * TICK < VT.HIDDEN_S
            if not excluded_slots[s_] and (p[s_] >= VT.KEEP_TRACK or hidden):
                keep[s_] = True; world[s_] = self.track_world[s_]
        new = [s_ for s_ in range(T, len(p)) if not pad[s_] and not excluded_slots[s_] and p[s_] >= VT.KEEP and area[s_] > 0]
        order = [s_ for s_ in range(len(p)) if keep[s_] and p[s_] >= VT.KEEP_TRACK] + sorted(new, key=lambda s_: -p[s_])
        kept_masks = []
        cos = VT.Runtime._cos
        for s_ in order:
            inter = [np.logical_and(masks[s_], masks[q]).sum() / max(1, np.logical_or(masks[s_], masks[q]).sum()) for q in kept_masks]
            if inter and max(inter) >= VT.DUP_IOU:
                q = kept_masks[int(np.argmax(inter))]
                if s_ < T and q < T and max(inter) >= VT.DUP_TRACK_IOU and cos(ident[s_], ident[q]) >= 1 - VT.DUP_TRACK_APP:
                    keep[s_] = False
                elif s_ >= T:
                    continue
                else:
                    kept_masks.append(s_)
                    continue
                continue
            kept_masks.append(s_)
        new = [s_ for s_ in new if s_ in kept_masks]
        xy = place[:, :2] * V.XY_S + np.array(V.XY_C)
        var = np.exp(place[:, 3:5]).sum(1) * V.XY_S ** 2
        taken = {w for w in world if w is not None}
        rt = self                                                     # assign_new needs .world, ._same, .dev
        got = VT.Runtime.assign_new(rt, [ident[s_] for s_ in new], [xy[s_] for s_ in new], [var[s_] for s_ in new], tick, 0, taken)
        for s_, w in zip(new, got):
            keep[s_] = True; world[s_] = w
        people, cut = [], np.zeros(masks.shape[1:], bool)
        self.last_masks = []                                          # aligned with people (door_video_model.py draws them)
        for s_ in range(len(p)):
            if keep[s_] and world[s_] is not None and p[s_] >= VT.KEEP_TRACK and s_ in kept_masks:
                self.world.see(world[s_], ident[s_], xy[s_], var[s_], tick, 0)
                ys, xs = np.nonzero(masks[s_])
                foot = [float(xs.mean()) * 4, float(ys.max()) * 4] if len(ys) else None
                people.append({'w': int(world[s_]), 's': round(float(p[s_]), 3), 'box': [round(float(v), 4) for v in r['box'][0, s_].float().cpu().tolist()],
                               'xy': [round(float(v), 3) for v in xy[s_]], 'z': [round(float(v), 3) for v in zone[s_]],
                               'st': int(state[s_].argmax()), 'foot': foot, 'new': bool(s_ >= T)})
                self.last_masks.append(masks[s_])
            if p[s_] >= 0.3 and not pad[s_]:
                cut |= masks[s_]
        kt = torch.as_tensor(keep, device=dev)[None]
        with torch.no_grad(), ac():
            self.track, idx = model.next_tracks(r, kt, tb)
            self.scene = model.scene.update(sc, model.norm(r['q']), r['pad'])
        kept = idx[0].tolist()
        prev_hidden = self.hidden_since
        self.track_world = [world[s_] for s_ in kept]
        self.hidden_since = [(prev_hidden[s_] if s_ < ntr and p[s_] < VT.KEEP_TRACK else tick) for s_ in kept]
        return people, cut

    def _same(self, a_ident, b_ident, ctx):
        return self.VT.Runtime._same(self, a_ident, b_ident, ctx)

    @property
    def torch(self):
        import torch
        return torch


def run(day, ckpt, pad=25.0):
    import torch
    import day_movie
    import sam31_segment as SS
    import slot_v2 as V
    dev = 'cuda'
    model = V.load(str(ROOT / ckpt)).to(dev).eval()
    cam = os.environ.get('RA_DOOR_CAM', 'cam1')          # camera 2 sees the same door from inside the shop
    oc = OneCam(model, dev, cam)
    raw = Raw(day, cam)
    start, _ = day_movie.clock(day, str(ROOT))
    lt = time.localtime(start)
    sod0 = lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec              # film second 0 as the second of the day
    name = Path(ckpt).parent.name + '_' + Path(ckpt).stem + ('' if cam == 'cam1' else '_' + cam) + os.environ.get('RA_DOOR_SUFFIX', '')
    OUT.mkdir(parents=True, exist_ok=True)
    out_p = OUT / ('%s_%s.jsonl.gz' % (day, name))
    st_p = OUT / ('%s_%s.status.json' % (day, name))
    spans = stretches(day, pad)[:int(os.environ.get('RA_DOOR_SPANS', '100000'))]   # RA_DOOR_SPANS: a smoke run
    total = sum(b - a for a, b in spans)
    samples = {}                                                       # film second -> recent-hall sample (float, nan = person)
    done, t_run = 0.0, time.time()
    kept = []                                                          # RA_DOOR_RESUME=1: whole stretches of an earlier run stay
    if os.environ.get('RA_DOOR_RESUME') == '1' and out_p.exists():
        try:
            for l in gzip.open(out_p, 'rt'):
                kept.append(json.loads(l))
        except (EOFError, json.JSONDecodeError, OSError):
            pass
        have = sorted({r['s'] for r in kept[1:]})
        whole = set(have[:-1])                                         # the last one may be cut
        kept = [r for r in kept[1:] if r['s'] in whole]
    skip = {r['s'] for r in kept}
    with gzip.open(out_p, 'wt') as fo:
        fo.write(json.dumps({'day': day, 'cam': cam, 'ckpt': ckpt, 'film_start': start, 'pad': pad, 'spans': spans, 'tick': TICK}) + '\n')
        for r in kept:
            fo.write(json.dumps(r) + '\n')
        for si, (a, b) in enumerate(spans):
            if si in skip:
                done += b - a
                continue
            # the recent hall before the stretch: samples every BG_EVERY s in the 5 minutes before (cut by a cold look)
            want = [a - BG_SPAN + k * BG_EVERY for k in range(int(BG_SPAN / BG_EVERY))]
            for ts in want:
                key = round(ts / BG_EVERY) * BG_EVERY
                if key in samples or key < 0:
                    continue
                tk = SS.tick_frames(day, cam, key, 1)[0]
                img = raw.get(tk)
                if img is None:
                    continue
                g = cv2.resize(img, (2176, 1224), interpolation=cv2.INTER_AREA)
                bl = bg_long(day, sod0 + key, cam=cam)
                oc.reset()
                _, cut = oc.step(key, {'img': g, 'bg_long': bl, 'bg_now': cv2.resize(bl, (BG_W, BG_H))}, 0)
                samples[key] = _sample(g, cut)
            oc.reset()
            n = min(int((b - a) / TICK), int(float(os.environ.get('RA_DOOR_MAX_S', '1e9')) / TICK))
            ticks = SS.tick_frames(day, cam, a, n)
            for i, tk in enumerate(ticks):
                t = a + i * TICK
                if i % 125 == 0 and (9, 40) <= time.localtime()[3:5] < (21, 0):
                    raise SystemExit('09:40: the card goes back to the live counter')
                img = raw.get(tk)
                if img is None:
                    continue
                g = cv2.resize(img, (2176, 1224), interpolation=cv2.INTER_AREA)
                use = [v for k, v in samples.items() if t - BG_SPAN <= k <= t]
                if not use:
                    use = [samples[max(samples)]] if samples else []
                bl = bg_long(day, sod0 + t, cam=cam)
                if i % int(BG_EVERY / TICK) == 0 or i == 0:
                    now = np.nan_to_num(np.nanmedian(np.stack(use), 0), nan=127).astype(np.uint8) if use else cv2.resize(bl, (BG_W, BG_H))
                people, cut = oc.step(i, {'img': g, 'bg_long': bl, 'bg_now': now}, i)
                key = round(t / BG_EVERY) * BG_EVERY
                if abs(t - key) < TICK / 2 and key not in samples:
                    samples[key] = _sample(g, cut)
                fo.write(json.dumps({'s': si, 't': round(t, 2), 'p': [dict(q, w=si * 10000 + q['w']) for q in people]}) + '\n')
                if i % 125 == 0:
                    el = time.time() - t_run
                    json.dump({'span': si, 'spans': len(spans), 'film_s': round(t, 1), 'done_s': round(done + i * TICK), 'total_s': round(total),
                               'elapsed_s': round(el), 'x_realtime': round((done + i * TICK) / max(1, el), 2), 'stamp_mismatch': raw.off,
                               'updated': time.strftime('%H:%M:%S')}, open(st_p, 'w'))
            done += b - a
            for k in [k for k in samples if k < a - BG_SPAN]:
                del samples[k]
    json.dump({'finished': time.strftime('%H:%M:%S'), 'total_s': round(total), 'elapsed_s': round(time.time() - t_run), 'stamp_mismatch': raw.off},
              open(st_p, 'w'))


def _sample(g, cut):
    """A recent-hall sample: the frame at 1088 x 612, the model's people (dilated) as nan."""
    img = cv2.resize(g, (BG_W, BG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
    m = cv2.resize(cut[:1224 // 4].astype(np.uint8), (BG_W, BG_H), interpolation=cv2.INTER_NEAREST)
    m = cv2.dilate(m, np.ones((9, 9), np.uint8))
    img[m > 0] = np.nan
    return img


# ------------------------------------------------------------------ scoring against the owner's answers

SMOOTH_S = 0.4          # zone: majority over +- this
MIN_SIDE_S = 0.5        # a side counts when the person stayed on it this long
DITHER_S = 20.0         # out and back in (or in and back out) by one person within this: one visit goes on (door_clean's rule)


def door_side(xy, margin):
    """1 inside / 0 outside by the signed distance to the door line (plan_door.json), 2 within +-margin of it."""
    d = json.load(open(ROOT / 'data' / 'plan_door.json'))
    a, b, inside = (np.array(d[k], float) for k in ('door_a', 'door_b', 'inside'))
    ab = b - a
    cross = lambda p: (ab[0] * (p[..., 1] - a[1]) - ab[1] * (p[..., 0] - a[0])) / np.linalg.norm(ab)
    sd = cross(xy) * np.sign(cross(inside))
    return np.where(sd > margin, 1, np.where(sd < -margin, 0, 2))


def io_model():
    """The /inout geometry boosting (section 37: 97.6 % leave-one-day-out) fitted on every owner answer; no 18.09
    there, so 18.09 stays honest. Cached in data/door_v2/io_geom.pkl."""
    import pickle
    import inout
    import inout_train as IT
    cache = OUT / 'io_geom.pkl'
    if cache.exists():
        return pickle.load(open(cache, 'rb'))
    S, L = inout.samples(str(ROOT)), inout.labels(str(ROOT))
    G, y, days = [], [], set()
    for sid, a in L.items():
        sm = S.get(sid)
        k = a.get('label') if isinstance(a, dict) else a
        if sm is None or k not in IT.CLASS:
            continue
        G.append(IT.geometry(sm)); y.append(IT.CLASS[k]); days.add(sm['day'])
    m = IT.boost().fit(np.array(G, np.float32), np.array(y))
    m.days_ = sorted(days)
    OUT.mkdir(parents=True, exist_ok=True)
    pickle.dump(m, open(cache, 'wb'))
    return m


def run_cam(path):
    try:
        return json.loads(gzip.open(path, 'rt').readline()).get('cam', 'cam1')
    except Exception:
        return 'cam1'


def add_io(ticks, path=None):
    """q['io'] = (outside, inside, doorway) of every person at every tick, from where they are in the frame."""
    import inout
    m = io_model()
    cam = run_cam(path) if path else 'cam1'
    dist = inout.floor_distance(cam)
    k = 1280 / 2176
    G, refs = [], []
    for r in ticks:
        for q in r['p']:
            cx, cy, w, h = q['box']
            x1, y1, x2, y2 = (cx - w / 2) * 2176 * k, (cy - h / 2) * 1248 * k, (cx + w / 2) * 2176 * k, (cy + h / 2) * 1248 * k
            fx, fy = (q['foot'][0] * k, q['foot'][1] * k) if q.get('foot') else ((x1 + x2) / 2, y2)
            d = float(dist[int(np.clip(fy, 0, 719)), int(np.clip(fx, 0, 1279))])
            G.append([float(cam == 'cam2'), x1 / 1280, y1 / 720, x2 / 1280, y2 / 720, (x2 - x1) / 1280, (y2 - y1) / 720, fx / 1280, fy / 720, d / 100.0])
            refs.append(q)
    if G:
        P = m.predict_proba(np.array(G, np.float32))
        for q, pr in zip(refs, P):
            q['io'] = pr.tolist()
    dep_p = Path(str(path).replace('.jsonl.gz', '.depth.json.gz')) if path else None
    if dep_p is not None and dep_p.exists() and os.environ.get('RA_DOOR_NODEPTH') != '1':             # door_depth.py: distance from the camera, every 2nd tick
        dep = json.load(gzip.open(dep_p, 'rt'))
        for r in ticks:
            for q in r['p']:
                v = dep.get('%d|%.2f|%d' % (r['s'], r['t'], q['w']))
                if v is not None and v[0] is not None:
                    q['dep'] = v
    lk_p = Path(str(path).replace('.jsonl.gz', '.lk.json.gz')) if path else None
    if lk_p is not None and lk_p.exists() and os.environ.get('RA_DOOR_NOLK') != '1':   # door_lk.py: do its points follow it
        lk = json.load(gzip.open(lk_p, 'rt'))
        for r in ticks:
            for q in r['p']:
                v = lk.get('%d|%.2f|%d' % (r['s'], r['t'], q['w']))
                if v is not None and v[1] is not None:
                    q['lk'] = v
    tap_p = Path(str(path).replace('.jsonl.gz', '.tap.json.gz')) if path else None
    if tap_p is not None and tap_p.exists() and os.environ.get('RA_DOOR_NOTAP') != '1':   # door_tap.py: TAPNext++ points
        tap = json.load(gzip.open(tap_p, 'rt'))
        for r in ticks:
            for q in r['p']:
                v = tap.get('%d|%.2f|%d' % (r['s'], r['t'], q['w']))
                if v is not None:
                    q['tap'] = v
    cnn_p = Path(str(path).replace('.jsonl.gz', '.io_cnn.json.gz')) if path else None
    if cnn_p is not None and cnn_p.exists() and os.environ.get('RA_DOOR_NOCNN') != '1':   # door_io_crops.py: the crop net
        cnn = json.load(gzip.open(cnn_p, 'rt'))
        for r in ticks:
            for q in r['p']:
                c = cnn.get('%d|%.2f|%d' % (r['s'], r['t'], q['w']))
                if c is not None:
                    q['io_cnn'] = c
                    q['io_mix'] = ((np.array(c) + np.array(q['io'])) / 2).tolist()
    return ticks


def crossings(ticks, how='zone', min_in=MIN_SIDE_S, min_out=MIN_SIDE_S, min_move=0.0):
    """Per world person: the side of the door at each tick (0 outside, 1 inside, 2 between), smoothed; a crossing is
    a change of side ('between' ignored) with both sides held MIN_SIDE_S. how:
      zone      the zone head, doorway between
      zone_hd   the zone head, doorway counted as outside (the model calls the people at the glass 'doorway')
      place     the place head against the door line, v2_prep.zones' rule (0.6 m doorway between)
      place_h   the place head, signed distance to the door line with +-0.3 m between"""
    import v2_prep
    net = how.endswith('_net')
    how = how.replace('_net', '')
    by = {}
    for row in ticks:
        for q in row['p']:
            by.setdefault(q['w'], []).append((row['t'], q))
    out = []
    k = max(1, int(round(SMOOTH_S / TICK)))
    for w, seq in by.items():
        ts = np.array([t for t, _ in seq])
        if how in ('io', 'io_cnn', 'io_mix'):
            seq = [(t, q) for t, q in seq if how in q]
            if not seq:
                continue
            ts = np.array([t for t, _ in seq])
            z = np.array([int(np.argmax(q[how])) for _, q in seq])
        elif how in ('zone', 'zone_hd'):
            z = np.array([int(np.argmax(q['z'])) for _, q in seq])
            if how == 'zone_hd':
                z = np.where(z == 2, 0, z)
        elif how == 'place':
            z = v2_prep.zones(np.array([q['xy'] for _, q in seq], float), np.ones(len(seq), bool)).astype(int)
        else:
            z = door_side(np.array([q['xy'] for _, q in seq], float), 0.3)
        sm = z.copy()
        for i in range(len(z)):
            sm[i] = np.bincount(z[max(0, i - k):i + k + 1], minlength=3).argmax()
        runs = []                                    # [side, t_first, t_last] of in/out runs, doorway skipped
        for t, v in zip(ts, sm):
            if v == 2:
                continue
            if runs and runs[-1][0] == v and t - runs[-1][2] < 3.0:
                runs[-1][2] = t
            else:
                runs.append([int(v), t, t])
        runs = [r_ for r_ in runs if r_[2] - r_[1] >= (min_in if r_[0] == 1 else min_out)]
        merged = []
        for r_ in runs:
            if merged and merged[-1][0] == r_[0]:
                merged[-1][2] = r_[2]
            else:
                merged.append(list(r_))
        if net:                      # the whole track: came from outside -> one entry; left to outside -> one exit
            if len(merged) >= 2 and merged[0][0] == 0:
                b = next(r_ for r_ in merged if r_[0] == 1)
                out.append({'w': w, 'kind': 'in', 't': round(b[1], 2)})
            if len(merged) >= 2 and merged[-1][0] == 0:
                a = [r_ for r_ in merged if r_[0] == 1][-1]
                out.append({'w': w, 'kind': 'out', 't': round(a[2], 2)})
            continue
        for a, b in zip(merged, merged[1:]):
            t = (a[2] + b[1]) / 2
            if min_move > 0 and moved(seq, t) < min_move:
                continue                       # the stand, somebody sitting: the side flickers, the feet stay
            out.append({'w': w, 'kind': 'in' if (a[0], b[0]) == (0, 1) else 'out', 't': round(t, 2)})
    return sorted(out, key=lambda c: c['t'])


def moved(seq, t, near=0.5, far=2.0):
    """px (2176 frame) between the median feet in [t-far, t-near] and in [t+near, t+far]."""
    a = [q['foot'] for s_, q in seq if q.get('foot') and t - far <= s_ <= t - near]
    b = [q['foot'] for s_, q in seq if q.get('foot') and t + near <= s_ <= t + far]
    if not a or not b:
        return 1e9                             # the track starts or ends at the crossing: it came or went
    return float(np.linalg.norm(np.median(a, 0) - np.median(b, 0)))


def undither(cs):
    """Drop an out followed by the same person's in within DITHER_S (and in then out): the visit goes on."""
    cs = sorted(cs, key=lambda c: c['t'])
    drop = set()
    for i, c in enumerate(cs):
        if i in drop:
            continue
        for j in range(i + 1, len(cs)):
            d = cs[j]
            if d['t'] - c['t'] > DITHER_S:
                break
            if j not in drop and d['w'] == c['w'] and d['kind'] != c['kind']:
                drop |= {i, j}
                break
    return [c for i, c in enumerate(cs) if i not in drop]


def truth(day, physical=True):
    """The owner's answers as crossings on the film clock: kind in/out (direction known), 'any' (staff without a
    direction on 17-18.09: a crossing, direction not said). physical: an answer Claude turned into 'none' by the
    dither rule counts as the owner first gave it -- somebody did cross the line."""
    import day_movie
    import door_review
    start, _ = day_movie.clock(day, str(ROOT))
    lab = door_review.labels(day, str(ROOT))['labels']
    out = []
    for e in door_review.day_events(day, str(ROOT)):
        v = lab.get(e['event_id'])
        if v is None:
            continue
        if physical and v.get('fixed') == 'claude' and v.get('kind') == 'none' and 'owner' in v:
            v = v['owner']
        kind = v.get('kind')
        if kind in ('in', 'out', 'staff'):
            out.append({'t': e['unix_ms'] / 1000 - start, 'kind': kind if kind != 'staff' else 'any',
                        'staff': kind == 'staff' or v.get('role') == 'staff', 'id': e['event_id']})
    # the owner's /doorcheck answers where the model and the counter disagreed (door_check.py): a crossing the
    # counter never wrote joins the truth on the counter's clock (the model's moment + door_learn.SHIFT); 'none'
    # adds nothing (the model's one is then plainly false); 'unsure' is set aside like a staff crossing
    import door_check
    import door_learn
    ans = door_check.answers(day)
    for it in door_check.items(day):
        a = ans.get(it['id'], {}).get('answer')
        kind = {'in': 'in', 'out': 'out', 'staff_in': 'in', 'staff_out': 'out', 'unsure': 'any'}.get(a)
        if kind:
            out.append({'t': it['film'] + door_learn.SHIFT, 'kind': kind, 'staff': a.startswith('staff'), 'id': 'check:' + it['id']})
    # the owner's own marks on the stretch videos (door_mark.py, /doormark): crossings no answer holds
    import door_mark
    for m in door_mark.truth_marks(day):
        if not any(o['kind'] in (m['kind'], 'any') and abs(o['t'] - m['t']) <= 2.0 for o in out):
            out.append(m)
    return out


def counter(day):
    """The live counter's own crossings (its event direction), as predictions."""
    import day_movie
    import door_review
    start, _ = day_movie.clock(day, str(ROOT))
    out = []
    for e in door_review.day_events(day, str(ROOT)):
        ev = str(e.get('event', '')).lower()
        kind = 'in' if ev in ('in', 'entry', 'enter') else 'out' if ev in ('out', 'exit', 'leave') else None
        if kind:
            out.append({'w': e.get('global_id'), 'kind': kind, 't': e['unix_ms'] / 1000 - start, 'role': e.get('role')})
    return out


def match(pred, true, tol):
    """One to one by time within tol. Returns precision/recall per direction; a prediction at an 'any' crossing
    (staff, no direction) is neither right nor wrong; a prediction matched to the opposite direction is wrong."""
    from scipy.optimize import linear_sum_assignment
    res = {}
    for kind in ('in', 'out'):
        P = [p for p in pred if p['kind'] == kind]
        T = [t for t in true if t['kind'] == kind]
        hit = 0
        used_p = set()
        if P and T:
            C = np.array([[abs(p['t'] - t['t']) for t in T] for p in P])
            r, c = linear_sum_assignment(np.where(C <= tol, C, 1e6))   # most pairs within tol first, then the closest
            for i, j in zip(r, c):
                if C[i, j] <= tol:
                    hit += 1; used_p.add(i)
        rest = [P[i] for i in range(len(P)) if i not in used_p]
        neutral = [t for t in true if t['kind'] == 'any']
        opposite = [t for t in true if t['kind'] not in (kind, 'any')]
        n_neu = n_opp = 0
        used_n = set()
        for p in rest:
            near = [i for i, t in enumerate(neutral) if abs(t['t'] - p['t']) <= tol and i not in used_n]
            if near:
                used_n.add(near[0]); n_neu += 1
            elif any(abs(t['t'] - p['t']) <= tol for t in opposite):
                n_opp += 1
        false = len(P) - hit - n_neu
        R = hit / max(1, len(T)); Pr = hit / max(1, hit + false)
        res[kind] = {'true': len(T), 'pred': len(P), 'hit': hit, 'false': false, 'wrong_direction': n_opp, 'at_staff_no_dir': n_neu,
                     'recall': round(R, 3), 'precision': round(Pr, 3), 'f1': round(2 * R * Pr / max(1e-9, R + Pr), 3)}
    return res


def grid(day, path, tols=(3.0, 6.0)):
    """The zone_hd rule over held times of the inside and the outside side: F1 of entries and exits (physical)."""
    ticks, spans = _read(path)
    add_io(ticks, path)
    hows = ('io', 'io_cnn', 'io_mix') if any('io_cnn' in q for r in ticks for q in r['p']) else ('io',)
    inside = lambda t: any(a <= t <= b for a, b in spans)
    tr = [t for t in truth(day, True) if inside(t['t'])]
    out = []
    import itertools
    for how, mi, mo, mm in itertools.product(hows, (0.5, 1.0, 2.0, 3.0), (0.5, 1.0, 2.0, 3.0), (0, 60, 120, 200)):
        cs = crossings(ticks, how, mi, mo, mm)
        for tol in tols:
            m = match(cs, tr, tol)
            out.append({'how': how, 'min_in': mi, 'min_out': mo, 'min_move': mm, 'tol': tol, 'in_f1': m['in']['f1'], 'out_f1': m['out']['f1'],
                        'f1': round((m['in']['f1'] + m['out']['f1']) / 2, 3), 'in': m['in'], 'out': m['out']})
    return out


def _read(path):
    ticks, spans = _read_raw(path)
    if os.environ.get('RA_DOOR_STITCH') == '1':              # door_stitch.py: one person, one number (06.10)
        import door_stitch
        door_stitch.stitch(ticks)
    return ticks, spans


def _read_raw(path):
    rows = []
    try:
        for l in gzip.open(path, 'rt'):
            rows.append(json.loads(l))
    except (EOFError, json.JSONDecodeError):
        pass
    head, ticks = rows[0], rows[1:]
    if 'sam_done' in head:                                   # door_sam.py: exactly the stretches SAM has finished
        keep = set(head['sam_done'])
        return [r for r in ticks if r['s'] in keep], [head['spans'][i] for i in sorted(keep)]
    done = sorted({r['s'] for r in ticks})
    n = len(done) if len(done) == len(head['spans']) and 'end' in head else len(done) - 1
    keep = set(done[:max(0, n)]) if len(done) < len(head['spans']) else set(done)
    return [r for r in ticks if r['s'] in keep], [head['spans'][i] for i in sorted(keep)]


def score(day, path, min_in=MIN_SIDE_S, min_out=MIN_SIDE_S, min_move=0.0):
    rows = []
    try:                                                    # a run still writing: read up to its last whole line
        for l in gzip.open(path, 'rt'):
            rows.append(json.loads(l))
    except (EOFError, json.JSONDecodeError):
        pass
    head, ticks = rows[0], rows[1:]
    done = sorted({r['s'] for r in ticks})
    spans = [head['spans'][i] for i in done[:-1]] if len(done) < len(head['spans']) else head['spans']   # a run still going: whole stretches only
    ticks = [r for r in ticks if r['s'] in set(done[:len(spans)])]
    add_io(ticks, path)
    inside = lambda t: any(a <= t <= b for a, b in spans)
    rep = {'day': day, 'ckpt': head['ckpt'], 'minutes': round(sum(b - a for a, b in spans) / 60, 1), 'ticks': len(ticks),
           'world_people': len({q['w'] for r in ticks for q in r['p']})}
    for phys in (True, False):
        tr = [t for t in truth(day, phys) if inside(t['t'])]
        part = rep.setdefault('physical' if phys else 'visits', {})
        for how in ('io', 'io_cnn', 'io_mix', 'zone_hd'):
            cs = crossings(ticks, how, min_in, min_out, min_move)
            if not phys:
                cs = undither(cs)
            for tol in (3.0, 6.0):
                part['%s_tol%d' % (how, tol)] = match(cs, tr, tol)
        part['counter_tol1'] = match([c for c in counter(day) if inside(c['t'])], tr, 1.0)
        part['truth'] = {'in': sum(t['kind'] == 'in' for t in tr), 'out': sum(t['kind'] == 'out' for t in tr), 'staff_no_dir': sum(t['kind'] == 'any' for t in tr)}
    return rep


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        for day in sys.argv[2].split(','):
            run(day, sys.argv[3], float(sys.argv[4]) if len(sys.argv) > 4 else 25.0)
    elif sys.argv[1] == 'score':
        print(json.dumps(score(sys.argv[2], sys.argv[3]), indent=1))
