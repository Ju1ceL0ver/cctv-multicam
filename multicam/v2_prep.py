"""Training targets of the v2 slot model from SAM 3.1 windows (data/sam31_seg/<tag>/<cam>/, sam31_jobs.py).

For every window and camera, into data/v2/<tag>/<cam>/:
  rows.npz   one entry per mask the window keeps (a seam's shared frames keep the earlier chunk's masks):
             r (row in chunks.npz), tick, person (the window's ReID-joined person), piece (the seam-linked SAM track),
             prob, box (x1 y1 x2 y2, video pixels 2176 x 1224), foot / head (raw 2560 x 1440 pixels, the bottom and
             top of the mask's main blob), xy (metres, the common floor frame -- camera 2's, person3d), height (m, nan
             when the feet are hidden), foot_vis, zone (0 outside, 1 inside, 2 doorway, -1 unknown: feet hidden)
  bg_recent.npy  the hall a moment ago: every 30 s the median of 15 frames of the 5 minutes before, people cut out
             by their SAM masks (1088 x 612); bg_ticks.npy their ticks
  meta.json  day, cam, film start, the segment file of every tick range (for the long, empty-hall background)
and, with `emb` (venv_rfdetr: the ReID teachers), emb.npz: per piece the mean TransReID (clothes) and CSCI (body
shape) vectors of its cleanest views, as sam31_reid picks them.

usage: v2_prep.py [TAG ...]            (all merged windows when none; the exam day never)
       v2_prep.py emb [TAG ...]"""
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SEG = ROOT / 'data' / 'sam31_seg'
OUT = ROOT / 'data' / 'v2'
EXAM = '20260918'
RAW_W, RAW_H = 2560, 1440
BG_W, BG_H = 1088, 612
BG_EVERY = 375          # ticks: 30 s
BG_SPAN = 3750          # ticks: 5 min
BG_SAMPLES = 15
DOOR_M = 0.6            # a foot this close to the door line is in the doorway


def windows(tags=None):
    """(tag, cam) of every merged window camera (report.json written), the exam day excluded."""
    out = []
    for tag in sorted(os.listdir(SEG)):
        if tag.startswith(EXAM) or (tags and tag not in tags):
            continue
        for cam in ('cam1', 'cam2'):
            if (SEG / tag / cam / 'report.json').exists():
                out.append((tag, cam))
    return out


def owned_rows(base):
    """Rows kept, their piece and person (the merge's ReID join)."""
    import sam31_reid
    info = json.load(open(base / 'info.json'))
    M = sam31_reid.Masks(base / 'chunks.npz')
    overlap = {int(s): int(sh) for s, e, sh in info['sessions']}
    owned, _ = sam31_reid.link_seams(M, overlap)
    rep = json.load(open(base / 'report.json'))
    pop = {int(k): int(v) for k, v in rep['person_of_piece'].items()}
    r, piece = [], []
    for p, rs in owned.items():
        r += rs; piece += [p] * len(rs)
    r, piece = np.array(r, np.int64), np.array(piece, np.int64)
    order = np.argsort(r)
    r, piece = r[order], piece[order]
    person = np.array([pop.get(int(p), -1) for p in piece], np.int64)
    return M, info, r, piece, person


def ends(M, r, W, H):
    """Foot (bottom centre) and head (top centre) of the mask's main blob, raw pixels."""
    import sam31_reid
    (x1, y1, x2, y2), _ = sam31_reid.main_blob(M, int(r))
    sx, sy = RAW_W / W, RAW_H / H
    cx = (x1 + x2) / 2 * sx
    return (cx, y2 * sy), (cx, y1 * sy)


def zones(xy, vis):
    """0 outside, 1 inside, 2 doorway, -1 unknown -- by the door line on the common floor (plan_door.json)."""
    d = json.load(open(ROOT / 'data' / 'plan_door.json'))
    a, b, inside = (np.array(d[k], float) for k in ('door_a', 'door_b', 'inside'))
    ab = b - a
    side = lambda p: np.sign(ab[0] * (p[..., 1] - a[1]) - ab[1] * (p[..., 0] - a[0]))
    t = np.clip(((xy - a) @ ab) / (ab @ ab), 0, 1)
    dist = np.linalg.norm(xy - (a + t[:, None] * ab), axis=1)
    z = np.where(side(xy) == side(inside), 1, 0)
    z = np.where(dist < DOOR_M, 2, z)
    return np.where(vis & np.isfinite(xy).all(1), z, -1).astype(np.int8)


PRIOR_H = 1.68         # m: the usual height, for a person never seen whole


def geometry(cam, foot, head, boxes_raw, person, tick):
    """Where each person stands (x, y m on the common floor) and how sure the target is (weight):
    1.0  the bottom of the mask on open floor, not cut by the frame's edge -- the feet;
    0.7  the top of the mask (the head) at this person's own height (median over the window's frames where the
         feet and the head are both seen), when the head is not cut by the frame's top;
    0.3  the head at the usual height (a person never seen whole in the window);
    then every frame without its own target (only an arm, a shoulder) takes its person's place interpolated
    between the nearest targeted frames within 3 s, weight 0.5. The network does not look for feet: it learns
    the place from whatever it sees; these are only its answers."""
    import person3d
    import rooms
    calib = json.load(open(ROOT / 'data' / 'calib_final.json'))
    reg = json.load(open(ROOT / 'data' / 'cam1_to_cam2_affine.json'))
    c = person3d.Camera(cam, calib)
    floor = rooms.mask(cam)
    n = len(foot)
    vis = np.zeros(n, bool)
    head_ok = np.zeros(n, bool)
    for i, (f, b) in enumerate(zip(foot, boxes_raw)):
        x, y = int(round(f[0])), int(round(f[1]))
        vis[i] = 0 <= x < RAW_W and 0 <= y < RAW_H and b[3] <= RAW_H - 12 and b[1] >= 4 and floor[y, x] > 0
        head_ok[i] = b[1] >= 4
    if not n:
        z = np.zeros((0, 2))
        return z, np.zeros(0), vis, np.zeros(0, np.float32)
    U = person3d.UNIT
    xy = c.on_plane(foot) * U
    h = np.where(vis & head_ok, c.height(foot, head), np.nan)
    h = np.where((h > 0.9) & (h < 2.2), h, np.nan)
    own = {}
    for p in set(person.tolist()):
        v = h[person == p]
        v = v[np.isfinite(v)]
        own[p] = float(np.median(v)) if len(v) >= 3 else None
    weight = np.where(vis, 1.0, 0.0).astype(np.float32)
    for p, hp in own.items():
        idx = np.nonzero((person == p) & ~vis & head_ok)[0]
        if not len(idx):
            continue
        xh = c.on_plane(head[idx], z=(hp or PRIOR_H) / U) * U
        ok = np.isfinite(xh).all(1)
        xy[idx[ok]] = xh[ok]
        weight[idx[ok]] = 0.7 if hp else 0.3
    for p in set(person.tolist()):                 # the rest: between the person's targeted frames
        idx = np.nonzero(person == p)[0]
        idx = idx[np.argsort(tick[idx])]
        good = idx[weight[idx] > 0]
        if not len(good):
            continue
        for i in idx[weight[idx] == 0]:
            t = tick[i]
            j = int(np.searchsorted(tick[good], t))
            a = good[j - 1] if j > 0 else None
            b = good[j] if j < len(good) else None
            near = [k for k in (a, b) if k is not None and abs(tick[k] - t) * 0.08 <= 3.0]
            if len(near) == 2:
                w = (t - tick[a]) / max(1, tick[b] - tick[a])
                xy[i] = xy[a] + w * (xy[b] - xy[a]); weight[i] = 0.5
            elif len(near) == 1:
                xy[i] = xy[near[0]]; weight[i] = 0.3
    height = np.array([own.get(int(p)) or np.nan for p in person], np.float32)
    height = np.where(np.isfinite(h), h, height)
    if cam == 'cam1':
        xy = person3d.map_cam1(xy, reg)
    xy = np.where(weight[:, None] > 0, xy, np.nan)
    return xy, height, vis, weight


def recent_backgrounds(base, M, r, tick):
    """The hall a moment ago, people cut out: every BG_EVERY ticks the median of BG_SAMPLES frames of the
    BG_SPAN before (pixels under a SAM mask left out)."""
    info = json.load(open(base / 'info.json'))
    W, H = info['size']
    n = info['ticks']
    step = max(1, BG_SPAN // BG_SAMPLES)
    want = set(range(0, n, step))
    by_tick = {}
    for i, t in zip(r, tick):
        if int(t) in want:
            by_tick.setdefault(int(t), []).append(int(i))
    cap = cv2.VideoCapture(str(base / 'video.mp4'))
    samples = {}
    k = -1
    while k < n - 1 and cap.grab():
        k += 1
        if k not in want:
            continue
        img = cv2.resize(cap.retrieve()[1], (BG_W, BG_H), interpolation=cv2.INTER_AREA).astype(np.float32)
        m = np.zeros((H, W), np.uint8)
        for i in by_tick.get(k, []):
            x1, y1, x2, y2 = M.rows[i, 4:8].astype(int)
            m[y1:y2, x1:x2] |= M.crop(i).astype(np.uint8)
        m = cv2.dilate(cv2.resize(m, (BG_W, BG_H), interpolation=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8))
        img[m > 0] = np.nan
        samples[k] = img
    cap.release()
    ticks = list(range(0, n, BG_EVERY))
    out = []
    fill = np.nanmedian(np.stack(list(samples.values())), 0)            # a pixel always covered: the whole window's
    for t in ticks:
        use = [samples[s] for s in sorted(samples) if t - BG_SPAN <= s <= max(t, step)]
        if not use:
            use = [samples[min(samples)]]
        med = np.nanmedian(np.stack(use), 0)
        med = np.where(np.isfinite(med), med, fill)
        out.append(np.nan_to_num(med, nan=127).astype(np.uint8))
    return np.stack(out), np.array(ticks, np.int64)


ROWS_VERSION = 3


def build(tag, cam):
    """Whatever of rows / backgrounds / meta is missing or old."""
    import warnings
    warnings.filterwarnings('ignore', 'All-NaN slice')
    base = SEG / tag / cam
    out = OUT / tag / cam
    out.mkdir(parents=True, exist_ok=True)
    done = []
    old = not (out / 'rows.npz').exists() or int(np.load(out / 'rows.npz').get('version', 1)) < ROWS_VERSION
    need_bg = not (out / 'bg_recent.npy').exists()
    M = info = r = tick = person = None
    if old or need_bg:
        M, info, r, piece, person = owned_rows(base)
        tick = M.rows[r, 1].astype(np.int64)
    if old:
        W, H = info['size']
        fh = [ends(M, i, W, H) for i in r]
        foot = np.array([f for f, _ in fh], np.float64).reshape(-1, 2)
        head = np.array([h for _, h in fh], np.float64).reshape(-1, 2)
        sx, sy = RAW_W / W, RAW_H / H
        boxes = M.rows[r, 4:8]
        xy, height, vis, weight = geometry(cam, foot, head, boxes * [sx, sy, sx, sy], person, tick)
        np.savez_compressed(out / 'rows.npz', version=ROWS_VERSION, r=r, tick=tick, person=person, piece=piece,
                            prob=M.rows[r, 3].astype(np.float32), box=boxes.astype(np.float32), foot=foot.astype(np.float32),
                            head=head.astype(np.float32), xy=xy.astype(np.float32), place_w=weight,
                            height=height.astype(np.float32), foot_vis=vis, zone=zones(xy, weight > 0))
        done.append('rows %d, %d people, a place for %.0f %% (feet %.0f %%)' % (len(r), len(set(person.tolist())),
                    100 * (weight > 0).mean() if len(r) else 0, 100 * vis.mean() if len(r) else 0))
    if need_bg:
        bg, bt = recent_backgrounds(base, M, r, tick)
        np.save(out / 'bg_recent.npy', bg)
        np.save(out / 'bg_ticks.npy', bt)
        done.append('backgrounds')
    if not (out / 'meta.json').exists():
        info = info or json.load(open(base / 'info.json'))
        ticks = json.load(open(base / 'ticks.json'))['ticks']
        seg, last = [], None
        for i, tk in enumerate(ticks):
            if not tk:                                   # a tick the camera did not record
                continue
            name = tk[0]
            if name != last:
                seg.append([i, name]); last = name
        z = np.load(out / 'rows.npz')
        json.dump({'tag': tag, 'day': info['day'], 'cam': cam, 'film_start': info['film_start'], 'ticks': info['ticks'],
                   'size': info['size'], 'segments': seg, 'people': int(len(set(z['person'].tolist()))), 'rows': int(len(z['r']))},
                  open(out / 'meta.json', 'w'), indent=1)
        done.append('meta')
    return ', '.join(done) or 'kept'


def embeddings(tag, cam, k=8):
    """Per piece: the mean teacher vectors of its k cleanest views (sam31_reid.views)."""
    import sam31_reid
    import track_emb
    base = SEG / tag / cam
    out = OUT / tag / cam
    if (out / 'emb.npz').exists():
        return 'kept'
    M, info, r, piece, person = owned_rows(base)
    W, H = info['size']
    owned = {}
    for i, p in zip(r, piece):
        owned.setdefault(int(p), []).append(int(i))
    pick = sam31_reid.views(M, owned, k, W, H)
    need = sorted({int(M.rows[i, 1]) for p in pick for i, _ in pick[p]})
    got, cap, f = {}, cv2.VideoCapture(str(base / 'video.mp4')), -1
    for t in need:
        while f < t and cap.grab():
            f += 1
        if f == t:
            got[t] = cap.retrieve()[1]
    cap.release()
    crops, owner = [], []
    for p in sorted(pick):
        for i, (x1, y1, x2, y2) in pick[p]:
            img = got.get(int(M.rows[i, 1]))
            if img is None:
                continue
            w, h = x2 - x1, y2 - y1
            c = img[max(0, int(y1 - 0.05 * h)):min(H, int(y2 + 0.05 * h)), max(0, int(x1 - 0.05 * w)):min(W, int(x2 + 0.05 * w))]
            if c.size:
                crops.append(c.copy()); owner.append(p)
    embed = track_emb.teachers()
    v1, v2 = embed(crops)
    owner = np.array(owner)
    ids = sorted(set(owner.tolist()))
    norm = lambda m: m / max(1e-8, np.linalg.norm(m))
    cloth = np.stack([norm(v1[owner == p].mean(0)) for p in ids]).astype(np.float16)
    shape = np.stack([norm(v2[owner == p].mean(0)) for p in ids]).astype(np.float16)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / 'emb.npz', piece=np.array(ids, np.int64), cloth=cloth, shape=shape)
    return 'emb %d pieces' % len(ids)


def main():
    args = sys.argv[1:]
    emb = args[:1] == ['emb']
    tags = args[1:] if emb else args
    for tag, cam in windows(tags or None):
        try:
            print(tag, cam, (embeddings if emb else build)(tag, cam), flush=True)
        except Exception as e:
            import traceback
            print(tag, cam, 'FAILED', repr(e)[:300], flush=True)
            traceback.print_exc()


if __name__ == '__main__':
    main()
