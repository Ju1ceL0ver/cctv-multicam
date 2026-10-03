"""Clips of the SAM 3.1 windows for the v2 slot model (targets from v2_prep.py).

A sample is T moments (ticks, 12.5 a second) of one window, both cameras. The workers return small uint8
pieces -- the frame (2176 x 1224 as stored), the empty hall and the hall a moment ago -- and the targets; the
network's inputs are put together on the GPU (assemble): ViT RGB 1280 x 720, STA 10 channels at 2176 x 1248.

Targets per camera and moment: masks at stride 4 of the STA frame (544 x 312), boxes (cx cy w h of the padded
frame), the window's person of each, piece (for the teachers' vectors), place (x, y m, height), whether the
feet are seen, zone; and the people hidden right now (seen before and after within HIDE_S) with their place
interpolated, for the tracks' state and place."""
import collections
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SEG = ROOT / 'data' / 'sam31_seg'
V2 = ROOT / 'data' / 'v2'
BGS = ROOT / 'data' / 'seg_datasets' / 'backgrounds'
SCENE = ROOT / 'data' / 'scene'
FW, FH = 2176, 1224               # stored frame
PH = 1248                         # padded to a multiple of 32
GW, GH = FW // 4, PH // 4         # stride-4 grid: 544 x 312
VW, VH = 1280, 720                # ViT input
TICK = 0.08
HIDE_S = 25.0                     # a person gone for less than this and back is hidden, not gone
MIN_PX = 40                       # masks smaller than this (stride-4 pixels) are not people to learn


MAX_OPEN = 4
_OPEN = collections.OrderedDict()   # open video decoders of this process, oldest first


def window_cams(exclude=()):
    """(tag, cam) with prepared targets."""
    out = []
    for tag in sorted(os.listdir(V2)) if V2.exists() else []:
        if any(tag.startswith(e) for e in exclude):
            continue
        for cam in ('cam1', 'cam2'):
            if all((V2 / tag / cam / f).exists() for f in ('rows.npz', 'bg_recent.npy', 'meta.json')):
                out.append((tag, cam))
    return out


_STAND = {}


def _stand_hit(cam, m):
    """RA_V2_NOPOSTER=1: does this window mask lie on the advertising stand (poster.npz of the camera, IoU >= 0.4)?"""
    if os.environ.get('RA_V2_NOPOSTER', '') != '1':
        return False
    if not _STAND:
        _STAND.update(dict(np.load(POSTER)) if POSTER.exists() else {'-': None})
    p = _STAND.get(cam)
    if p is None:
        return False
    return (m & p).sum() / max(1, (m | p).sum()) >= 0.4


class Window:
    """One camera of one window: targets in memory, masks and frames read on demand."""

    def __init__(self, tag, cam):
        import sam31_reid
        self.tag, self.cam = tag, cam
        d = V2 / tag / cam
        z = np.load(d / 'rows.npz')
        self.t = {k: z[k] for k in z.files}
        if 'place_w' not in self.t:                  # rows of an older v2_prep: feet only
            self.t['place_w'] = self.t['foot_vis'].astype(np.float32)
        self.meta = json.load(open(d / 'meta.json'))
        self.masks = sam31_reid.Masks(SEG / tag / cam / 'chunks.npz', mmap=True)   # shared by all workers (to_npy)
        self.bg_recent = np.load(d / 'bg_recent.npy', mmap_mode='r')
        self.bg_ticks = np.load(d / 'bg_ticks.npy')
        e = d / 'emb.npz'
        self.emb = None
        if e.exists():
            z = np.load(e)
            self.emb = {int(p): (c.astype(np.float32), s.astype(np.float32)) for p, c, s in zip(z['piece'], z['cloth'], z['shape'])}
        tick = self.t['tick']
        order = np.argsort(tick, kind='stable')
        self.order, self.sorted_ticks = order, tick[order]
        # per person: its ticks and places (sorted), for hidden people and 'gone'
        self.track = {}
        for i in order:
            self.track.setdefault(int(self.t['person'][i]), []).append(int(i))
        self.video = SEG / tag / cam / 'video.mp4'
        self.cap, self.pos = None, -1
        self._bg_long = {}

    def rows_at(self, tick):
        a, b = np.searchsorted(self.sorted_ticks, [tick, tick + 1])
        return self.order[a:b]

    def frame(self, tick):
        """The stored frame of a tick (reads forward when it can, seeks when it must). At most MAX_OPEN decoders stay
        open per process (~0.2 GB each): the least recently used one is closed."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.video))
            self.pos = -1
        _OPEN.pop(id(self), None)
        _OPEN[id(self)] = self
        while len(_OPEN) > MAX_OPEN:
            _, old = _OPEN.popitem(last=False)
            if old.cap is not None:
                old.cap.release()
            old.cap, old.pos = None, -1
        if tick <= self.pos or tick > self.pos + 30:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, tick)
            self.pos = tick - 1
        img = None
        while self.pos < tick:
            ok, img = self.cap.read()
            if not ok:
                return None
            self.pos += 1
        return img

    def bg_long(self, tick):
        """The empty hall of the tick's 15-minute file (1280 x 720), else the nearest file's."""
        seg = self.meta['segments']
        name = seg[0][1]
        for start, n in seg:
            if start <= tick:
                name = n
        name = name.replace('.mp4', '')
        if name not in self._bg_long:
            p = BGS / ('%s_%s_%s.jpg' % (self.meta['day'], self.cam, name))
            if not p.exists():
                cands = sorted(BGS.glob('%s_%s_*.jpg' % (self.meta['day'], self.cam)))
                p = min(cands, key=lambda q: abs(int(q.stem.split('_')[2]) - int(name.split('_')[0]))) if cands else None
            self._bg_long[name] = cv2.imread(str(p)) if p is not None else None
        return self._bg_long[name]

    def bg_now(self, tick):
        i = max(0, int(np.searchsorted(self.bg_ticks, tick, side='right')) - 1)
        return np.asarray(self.bg_recent[i])

    def targets(self, tick):
        t = self.t
        idx = [i for i in self.rows_at(tick)]
        masks, boxes, keep = [], [], []
        for i in idx:
            r = int(t['r'][i])
            x1, y1, x2, y2 = self.masks.rows[r, 4:8].astype(int)
            gx1, gy1 = x1 // 4, y1 // 4
            gx2, gy2 = max(gx1 + 1, min(GW, -(-x2 // 4))), max(gy1 + 1, min(GH, -(-y2 // 4)))
            crop = self.masks.crop(r).astype(np.float32)
            small = np.zeros((GH, GW), np.float32)
            small[gy1:gy2, gx1:gx2] = cv2.resize(crop, (gx2 - gx1, gy2 - gy1), interpolation=cv2.INTER_AREA)
            small = small >= 0.5
            if small.sum() < MIN_PX:
                continue
            if _stand_hit(self.cam, small):                  # RA_V2_NOPOSTER: the advertising stand is not a person
                continue
            masks.append(small)
            keep.append(i)
            boxes.append([(x1 + x2) / 2 / FW, (y1 + y2) / 2 / PH, (x2 - x1) / FW, (y2 - y1) / PH])
        keep = np.array(keep, np.int64)
        out = {'masks': np.stack(masks) if masks else np.zeros((0, GH, GW), bool),
               'boxes': np.array(boxes, np.float32).reshape(-1, 4),
               'person': t['person'][keep] if len(keep) else np.zeros(0, np.int64),
               'piece': t['piece'][keep] if len(keep) else np.zeros(0, np.int64),
               'xy': t['xy'][keep] if len(keep) else np.zeros((0, 2), np.float32),
               'height': t['height'][keep] if len(keep) else np.zeros(0, np.float32),
               'foot_vis': t['foot_vis'][keep] if len(keep) else np.zeros(0, bool),
               'place_w': t['place_w'][keep] if len(keep) else np.zeros(0, np.float32),
               'zone': t['zone'][keep].astype(np.int64) if len(keep) else np.zeros(0, np.int64)}
        cloth, shape, has = [], [], []
        for p in out['piece']:
            e = self.emb.get(int(p)) if self.emb else None
            has.append(e is not None)
            cloth.append(e[0] if e is not None else np.zeros(3840, np.float32))
            shape.append(e[1] if e is not None else np.zeros(1024, np.float32))
        out['cloth'] = np.stack(cloth) if cloth else np.zeros((0, 3840), np.float32)
        out['shape'] = np.stack(shape) if shape else np.zeros((0, 1024), np.float32)
        out['has_reid'] = np.array(has, bool)
        # hidden now: seen before and after within HIDE_S, not seen at this tick
        here = set(out['person'].tolist())
        hid_p, hid_xy, gone_p = [], [], []
        for p, rows in self.track.items():
            if p in here:
                continue
            ticks = t['tick'][rows]
            j = int(np.searchsorted(ticks, tick))
            if j == 0:
                continue                                   # not yet in the window
            before = rows[j - 1]
            if j < len(rows) and (ticks[j] - ticks[j - 1]) * TICK <= HIDE_S:
                after = rows[j]
                a, b = t['xy'][before], t['xy'][after]
                w = (tick - ticks[j - 1]) / max(1, ticks[j] - ticks[j - 1])
                ok = np.isfinite(a).all() and np.isfinite(b).all()
                hid_p.append(p); hid_xy.append(a + w * (b - a) if ok else [np.nan, np.nan])
            elif (tick - ticks[j - 1]) * TICK <= HIDE_S:
                gone_p.append(p)                              # recently seen, not coming back: gone
        out['hidden'] = np.array(hid_p, np.int64)
        out['hidden_xy'] = np.array(hid_xy, np.float32).reshape(-1, 2)
        out['gone'] = np.array(gone_p, np.int64)
        return out


class Clips:
    """Random clips: T moments, stride 1..max_stride ticks, both cameras of a window."""

    def __init__(self, tags, T=1, max_stride=3, seed=0, busy=0.9):
        self.T, self.max_stride, self.busy = T, max_stride, busy
        by = {}
        for tag, cam in window_cams():
            if tag in tags:
                by.setdefault(tag, []).append(cam)
        self.tags = [t for t, cams in by.items() if len(cams) == 2]
        self.win = {}
        self.rng = random.Random(seed)

    def w(self, tag, cam):
        k = (tag, cam)
        if k not in self.win:
            self.win[k] = Window(tag, cam)
        return self.win[k]

    def sample(self):
        tag = self.rng.choice(self.tags)
        w1, w2 = self.w(tag, 'cam1'), self.w(tag, 'cam2')
        n = w1.meta['ticks']
        stride = self.rng.randint(1, self.max_stride)
        span = (self.T - 1) * stride
        if self.rng.random() < self.busy:                     # a moment with people
            t0 = int(self.rng.choice(np.concatenate([w1.sorted_ticks, w2.sorted_ticks])))
            t0 = min(max(0, t0 - self.rng.randint(0, span)), n - 1 - span)
        else:
            t0 = self.rng.randint(0, n - 1 - span)
        ticks = [t0 + i * stride for i in range(self.T)]
        frames = []
        for t in ticks:
            per = []
            for w in (w1, w2):
                img = w.frame(t)
                if img is None:
                    return self.sample()
                per.append({'img': img, 'bg_long': w.bg_long(t), 'bg_now': w.bg_now(t), 'tg': w.targets(t)})
            frames.append(per)
        return {'tag': tag, 'ticks': ticks, 'frames': frames}


def worker_stream(tags, T, max_stride, seed, q):
    """A data worker: samples into a queue forever."""
    c = Clips(tags, T, max_stride, seed)
    while True:
        q.put(c.sample())


# ---------------------------------------------------------------- single frames of the rough drafts
# The 13 thousand frames of 9 days (yolo26x boxes -> SAM 2.1 L masks, seg_datasets/pseudo_20260925*): far more
# people and situations than the SAM 3.1 windows. They teach finding people and their masks only -- no tracks, no
# identity, no place. SAM 3.1 calls the advertising stand by the door a person in every frame, the drafts do not:
# the stand's mask (poster.npz, from the windows) is added to every draft where nobody covers it, so both teachers
# say the same thing.
DSETS = ROOT / 'data' / 'seg_datasets'
DRAFT_DIRS = ('pseudo_20260925', 'pseudo_20260925_more', 'pseudo_20260925_random')
POSTER = V2 / 'poster.npz'


SAM31_STILLS = ROOT / 'data' / 'sam31_stills' / 'drafts'
SAM31_CLEAN = ROOT / 'data' / os.environ.get('RA_V2_CLEAN_DIR', 'sam31_stills_clean')          # decide.py: the keep/delete network's cleaning of the same maps


def sam31_drafts():
    """RA_V2_DRAFTS=sam31: the same frames with SAM 3.1's labels (sam31_stills.py) instead of yolo26x + SAM 2.1 L."""
    return os.environ.get('RA_V2_DRAFTS', '') in ('sam31', 'sam31clean')


def sam31_clean():
    """RA_V2_DRAFTS=sam31clean: SAM 3.1 maps cleaned by decide.py (the stand and other not-people are background). A frame that still holds
    an ignored person (value 254) is left out whole until the loss can ignore a place."""
    return os.environ.get('RA_V2_DRAFTS', '') == 'sam31clean'


def _ignore_frames():
    p = SAM31_CLEAN / '_ignore_frames.json'
    return set(json.load(open(p))) if p.exists() else set()


def draft_list(exclude_days=('20260918', '20260923')):
    """(label png, image jpg, day, cam, segment) of every draft, the exam day and the held-out day left out."""
    out = []
    skip = _ignore_frames() if sam31_clean() else set()
    for d in DRAFT_DIRS:
        for p in sorted((DSETS / d / 'drafts').glob('*.png')):
            if sam31_drafts():
                p = (SAM31_CLEAN if sam31_clean() else SAM31_STILLS) / p.name
                if not p.exists() or p.stem in skip:
                    continue
            parts = p.stem.split('_')
            if parts[0] in exclude_days:
                continue
            img = next((q for q in (DSETS / d / s / 'images' / (p.stem + '.jpg') for s in ('train', 'test')) if q.exists()), None)
            if img is not None:
                out.append((str(p), str(img), parts[0], parts[1], '_'.join(parts[2:4])))
    return out


def make_poster(tag='20260917_09005', ticks=400):
    """The stand's mask per camera (stride-4 grid): the window's person seen at >= 95 % of the live ticks whose box
    never moves, averaged over `ticks` ticks, kept where >= half."""
    out = {}
    for cam in ('cam1', 'cam2'):
        w = Window(tag, cam)
        live = len(np.unique(w.t['tick']))
        best = None
        for p, rows in w.track.items():
            if len(rows) < 0.95 * live:
                continue
            b = w.t['box'][rows]
            spread = float(np.median(np.abs(b - np.median(b, 0))))
            if best is None or spread < best[0]:
                best = (spread, p, rows)
        _, p, rows = best
        acc = np.zeros((GH, GW), np.float32)
        n = 0
        for i in rows[:: max(1, len(rows) // ticks)]:
            tg = w.targets(int(w.t['tick'][i]))
            k = np.nonzero(tg['person'] == p)[0]
            if len(k):
                acc += tg['masks'][k[0]]
                n += 1
        out[cam] = acc / max(1, n) >= 0.5
        print(cam, 'person', p, 'seen', len(rows), 'of', live, 'ticks, mask', int(out[cam].sum()), 'cells', flush=True)
    np.savez_compressed(POSTER, **out)
    return out


def draft_targets(label, cam, poster=None):
    """A draft's label map (720 x 1280, 0 = nobody) -> the same targets as Window.targets (fields the draft cannot
    know are empty: no place, no zone, no identity, nobody hidden)."""
    masks = []
    for v in np.unique(label):
        if not v:
            continue
        g = cv2.resize((label == v).astype(np.float32), (GW, FH // 4), interpolation=cv2.INTER_AREA) >= 0.5
        if g.sum() < MIN_PX:
            continue
        full = np.zeros((GH, GW), bool)
        full[:FH // 4] = g
        masks.append(full)
    if poster is not None:
        anyone = np.any(masks, 0) if masks else np.zeros((GH, GW), bool)
        covered = [(m & poster).sum() / max(1, (m | poster).sum()) for m in masks]
        free = poster & ~anyone
        if free.sum() >= MIN_PX and not any(c > 0.3 for c in covered):
            masks.append(free)
    n = len(masks)
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        x1, x2, y1, y2 = xs.min() * 4, (xs.max() + 1) * 4, ys.min() * 4, (ys.max() + 1) * 4
        boxes.append([(x1 + x2) / 2 / FW, (y1 + y2) / 2 / PH, (x2 - x1) / FW, (y2 - y1) / PH])
    return {'masks': np.stack(masks) if n else np.zeros((0, GH, GW), bool), 'boxes': np.array(boxes, np.float32).reshape(-1, 4),
            'person': np.arange(n, dtype=np.int64), 'piece': np.full(n, -1, np.int64), 'xy': np.zeros((n, 2), np.float32),
            'height': np.full(n, np.nan, np.float32), 'foot_vis': np.zeros(n, bool), 'place_w': np.zeros(n, np.float32),
            'zone': np.full(n, -1, np.int64), 'cloth': np.zeros((n, 3840), np.float32), 'shape': np.zeros((n, 1024), np.float32),
            'has_reid': np.zeros(n, bool), 'hidden': np.zeros(0, np.int64), 'hidden_xy': np.zeros((0, 2), np.float32),
            'gone': np.zeros(0, np.int64)}


class Drafts:
    """Random pairs of draft frames (two unrelated frames, each with its own camera)."""

    def __init__(self, seed=0):
        self.items = draft_list()
        self.rng = random.Random(seed)
        self.poster = dict(np.load(POSTER)) if POSTER.exists() else {}
        self._bg = collections.OrderedDict()
        # the owner's corrected training frames (/fix, taken): this share of the draft frames (RA_V2_GOLD, 0 = none)
        self.gold_share = float(os.environ.get('RA_V2_GOLD', '0') or 0)
        self.gold, self._win = [], collections.OrderedDict()
        if self.gold_share > 0:
            import v2_teacher_test as TT
            self.gold = [(it, str(p)) for it, p in TT.fix_items('take', 'train')]

    def gold_one(self):
        it, lab_p = self.rng.choice(self.gold)
        if it.get('src') == 'still':                     # a single varied frame (live.py): its file's background
            img = cv2.imread(str(Path(lab_p).parent / ('%s.jpg' % it['id'])))
            lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
            parts = it['id'].split('_')
            bg = self.bg(parts[0], parts[1], '_'.join(parts[2:4]))
            if img is None or lab is None or bg is None:
                return None
            lab = lab[:, :, 0] if lab.ndim == 3 else lab
            return it['cam'], {'img': img, 'bg_long': bg, 'bg_now': bg, 'tg': draft_targets(lab, it['cam'], None)}
        k = (it['tag'], it['cam'])
        if k not in self._win:
            self._win[k] = Window(*k)
            while len(self._win) > 8:
                self._win.popitem(last=False)
        w = self._win[k]
        img = cv2.imread(str(Path(lab_p).parent / ('%s.jpg' % it['id'])))
        lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
        if img is None or lab is None:
            return None
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        return it['cam'], {'img': img, 'bg_long': w.bg_long(it['tick']), 'bg_now': np.array(w.bg_now(it['tick'])),
                           'tg': draft_targets(lab, it['cam'], None)}

    def bg(self, day, cam, seg):
        k = (day, cam, seg)
        if k not in self._bg:
            self._bg[k] = cv2.imread(str(BGS / ('%s_%s_%s.jpg' % (day, cam, seg))))
            while len(self._bg) > 64:
                self._bg.popitem(last=False)
        return self._bg[k]

    def one(self):
        if self.gold and self.rng.random() < self.gold_share:
            return self.gold_one()
        lab_p, img_p, day, cam, seg = self.rng.choice(self.items)
        img = cv2.imread(img_p)
        lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
        bg = self.bg(day, cam, seg)
        if img is None or lab is None or bg is None:
            return None
        poster = None if sam31_drafts() else self.poster.get(cam)        # SAM 3.1 outlines the stand itself
        return cam, {'img': img, 'bg_long': bg, 'bg_now': bg, 'tg': draft_targets(lab, cam, poster)}

    def sample(self, k=2):
        out = []
        while len(out) < k:
            x = self.one()
            if x is not None:
                out.append(x)
        return {'cams': tuple(c for c, _ in out), 'frames': [f for _, f in out], 'draft': True}


def draft_stream(seed, q, k=2):
    d = Drafts(seed)
    while True:
        q.put(d.sample(k))


# ---------------------------------------------------------------- on the GPU
IMNET_M = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_S = np.array([0.229, 0.224, 0.225], np.float32)


class Assembler:
    """uint8 pieces -> the network's inputs, on the GPU, with the augmentations: colour per clip (frame and
    backgrounds apart, so a lighting change is learned), backgrounds shifted by a few pixels (BSUV 2.0)."""

    def __init__(self, dev, rgb_size=None):
        """rgb_size: (w, h) of the RGB backbone's input (the model's rgb_size; default VW x VH)."""
        import torch
        self.dev = dev
        self.vw, self.vh = rgb_size or (VW, VH)
        self.depth = {}
        for c in ('cam1', 'cam2'):
            d = np.load(SCENE / ('depth_%s_da2_large.npy' % c)).astype(np.float32)
            d = (d - d.min()) / max(1e-6, d.max() - d.min())
            d = cv2.resize(d, (FW, FH), interpolation=cv2.INTER_LINEAR)
            self.depth[c] = torch.from_numpy(np.pad(d, ((0, PH - FH), (0, 0)))).to(dev)[None, None] - 0.5
        self.world = {}
        for c in ('cam1', 'cam2'):
            w = SCENE / ('world_%s.npy' % c)
            self.world[c] = torch.from_numpy(np.load(w)).permute(2, 0, 1).to(dev) if w.exists() else torch.zeros(3, 90, 160, device=dev)
        self.m = torch.tensor(IMNET_M, device=dev)[:, None, None]
        self.s = torch.tensor(IMNET_S, device=dev)[:, None, None]

    def tensor(self, img, size):
        import torch
        import torch.nn.functional as F
        x = torch.from_numpy(img if img.flags.c_contiguous else np.ascontiguousarray(img)).to(self.dev, non_blocking=True).flip(2).permute(2, 0, 1).float() / 255   # BGR->RGB on the card, not a copy on the CPU
        if x.shape[1:] != (size[1], size[0]):
            x = F.interpolate(x[None], size=(size[1], size[0]), mode='bilinear', align_corners=False)[0]
        return x

    @staticmethod
    def jitter(x, g):
        """g: (brightness, contrast, saturation) factors."""
        b, c, s = g
        x = x * b
        mean = x.mean()
        x = (x - mean) * c + mean
        gray = x.mean(0, keepdim=True)
        return ((x - gray) * s + gray).clamp(0, 1)

    def __call__(self, frames, cams=('cam1', 'cam2'), train=True, rng=None):
        """frames: [ {img, bg_long, bg_now} per camera ] of one moment -> rgb (2 x 3 x 720 x 1280), bg_rgb (the
        long background for the ViT), sta (2 x 10 x 1248 x 2176), cam ids."""
        import torch
        import torch.nn.functional as F
        rng = rng or random.Random()
        rgb, bgv, sta = [], [], []
        for f, cam in zip(frames, cams):
            gf = (rng.uniform(0.75, 1.25), rng.uniform(0.8, 1.2), rng.uniform(0.8, 1.2)) if train else (1, 1, 1)
            gb = (gf[0] * rng.uniform(0.9, 1.1), gf[1], gf[2]) if train else gf
            img = self.jitter(self.tensor(f['img'], (FW, FH)), gf)
            bl = f['bg_long'] if f['bg_long'] is not None else f['bg_now']
            long_ = self.jitter(self.tensor(bl, (FW, FH)), gb)
            now = self.jitter(self.tensor(f['bg_now'], (FW, FH)), gb)
            if train:
                dx, dy = rng.randint(-5, 5), rng.randint(-5, 5)
                long_ = torch.roll(long_, (dy, dx), (1, 2))
                dx, dy = rng.randint(-5, 5), rng.randint(-5, 5)
                now = torch.roll(now, (dy, dx), (1, 2))
            pad = lambda x: F.pad(x, (0, 0, 0, PH - FH))
            nrm = lambda x: (x - self.m) / self.s
            rgb.append(nrm(F.interpolate(img[None], size=(self.vh, self.vw), mode='bilinear', align_corners=False)[0]))
            bgv.append(nrm(F.interpolate(long_[None], size=(self.vh, self.vw), mode='bilinear', align_corners=False)[0]))
            sta.append(torch.cat([pad(nrm(img)), pad(nrm(long_)), pad(nrm(now)), self.depth[cam][0]], 0))
        cam_ids = torch.tensor([0 if c == 'cam1' else 1 for c in cams], device=self.dev)
        self.last_world = torch.stack([self.world[c] for c in cams])
        cl = torch.channels_last
        return torch.stack(rgb).contiguous(memory_format=cl), torch.stack(bgv).contiguous(memory_format=cl), torch.stack(sta).contiguous(memory_format=cl), cam_ids
