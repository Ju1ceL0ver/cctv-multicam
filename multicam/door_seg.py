"""A fast segmenter that sees time: YOLO26s-seg on 17 channels + a head of displacements (09.10.2026).

The cameras never move, the hall never changes, the live door already runs a minute behind the camera -- so the
network gets more than one frame (decided with the owner, AGENTS 64):

  input 1280 x 736 (720 padded), 17 channels:
    RGB of the frame t                                          3
    the empty hall (the day's background, data/seg_datasets/backgrounds)   3
    |grey t - grey t-k|, k = 1..5   (the past, every frame)     5
    |grey t+k - grey t|, k = 1..5   (the future: the live door is a minute late anyway, 5 frames = 0.4 s)  5
    the people of t-1 (a mask: the teacher's, spoiled, in training; its own answer live)   1
  output: the people of t (boxes + masks, YOLO26's own end-to-end head, no NMS)
          + per stride-8 cell, where this person's centre was at t-1 and will be at t+1 (dx, dy, dx, dy):
            the tracker links people from the network, not from guesses of overlap (CenterTrack's idea, both ways).

Targets: the SAM 3.1 teacher (sam31_seg windows and sam31_door stretches, numbers of people consistent inside a
session) -- without 23.09 (the test) and the door 18.09, as sam31_e2e_det.sources(). A tick is 0.08 s (12.5 a second).

usage (venv_sam3, the card):
  door_seg.py smoke                 one batch: shapes, losses, a backward pass
  door_seg.py train RUN [HH:MM]     trains until HH:MM (default 20:50), resumes from runs/RUN/last.pt, saves every
                                    10 minutes; evaluates against the teacher on 23.09 every EVAL_MIN minutes
  door_seg.py eval CKPT             found / precision / F1 / mask IoU / links against the teacher on 23.09, ms per frame
"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
W, H, HP = 1280, 720, 736
MW, MH = W // 4, HP // 4                 # the masks: stride 4 (320 x 184), as YOLO's proto
GW, GH = W // 8, HP // 8                 # the displacements: stride 8 (160 x 92)
PAST, FUT = (1, 2, 3, 4, 5), (1, 2, 3, 4, 5)   # the owner: every frame from t-5 to t+5 (0.4 s each way)
C = 3 + 3 + len(PAST) + len(FUT) + 1
OFF_SCALE = 16.0                         # input pixels per unit of the displacement head
MAX_STEP = 64.0                          # px of 1280 in one tick (0.08 s): more is two people under one number
DIFF_GAIN = 4                            # differences are small: x4, clipped to 255
BGS = ROOT / 'data' / 'seg_datasets' / 'backgrounds'
EVAL_MIN = 30
TEST_DAY = '20260923'


def say(run, text):
    line = '%s  %s' % (time.strftime('%d.%m %H:%M'), text)
    run.mkdir(parents=True, exist_ok=True)
    with open(run / 'progress.md', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


# ------------------------------------------------------------------ data

def sources(test=False):
    if test:
        return sorted(p for p in (ROOT / 'data' / 'sam31_seg').glob(TEST_DAY + '*/cam*')
                      if (p / 'video.mp4').exists() and (p / 'chunks.npz').exists())
    import sam31_e2e_det as E
    return E.sources()


_BG = {}


def background(day, cam):
    """the day's empty hall of the camera (median of its 15-minute backgrounds), 1280 x 720 BGR; another day's when
    the day has none"""
    key = (day, cam)
    if key not in _BG:
        fs = sorted(BGS.glob('%s_%s_*.jpg' % (day, cam))) or sorted(BGS.glob('*_%s_*.jpg' % cam))
        ims = [cv2.resize(cv2.imread(str(f)), (W, H), interpolation=cv2.INTER_AREA) for f in fs[:: max(1, len(fs) // 9)]]
        ims = [i for i in ims if i is not None]
        _BG[key] = np.median(np.stack(ims), 0).astype(np.uint8) if ims else np.zeros((H, W, 3), np.uint8)
    return _BG[key]


def _grey(f):
    return cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)


def channels(frames, bg, prev_mask):
    """frames: {offset: 1280 x 720 BGR} for 0, -PAST, +FUT; bg 1280 x 720 BGR; prev_mask 720 x 1280 uint8 (0/1) ->
    C x 736 x 1280 uint8"""
    out = np.zeros((C, HP, W), np.uint8)
    f0 = frames[0]
    out[0:3, :H] = f0[:, :, ::-1].transpose(2, 0, 1)
    out[3:6, :H] = bg[:, :, ::-1].transpose(2, 0, 1)
    g0 = _grey(f0).astype(np.int16)
    c = 6
    for k in PAST:
        out[c, :H] = np.clip(np.abs(g0 - _grey(frames[-k]).astype(np.int16)) * DIFF_GAIN, 0, 255)
        c += 1
    for k in FUT:
        out[c, :H] = np.clip(np.abs(_grey(frames[k]).astype(np.int16) - g0) * DIFF_GAIN, 0, 255)
        c += 1
    out[c, :H] = prev_mask * 255
    return out


class Source:
    """one camera of one SAM 3.1 window or door stretch"""

    def __init__(self, d):
        import sam31_reid as R
        self.d = d
        self.M = R.Masks(d / 'chunks.npz', mmap=True)
        rows = self.M.rows
        self.sess = {}
        for i in range(len(rows)):
            self.sess.setdefault(int(rows[i, 0]), {}).setdefault(int(rows[i, 1]), []).append(i)
        cap = cv2.VideoCapture(str(d / 'video.mp4'))
        self.n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.vw, self.vh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        # ticks where the frame and its neighbours exist and a session holds t-1, t, t+1
        self.starts = [(s, k) for s, by in self.sess.items() for k in by
                       if k - max(PAST) >= 0 and k + max(FUT) < self.n and (k - 1) in by and (k + 1) in by]
        name = d.parent.name
        self.day, self.cam = name[:8], d.name
        self.cap = None
        self.ident = glued_ids(d, self.M) if os.environ.get('RA_DS_GLUED') == '1' else None
        if self.ident is not None:                 # numbers across sessions: any row of a tick, one per person
            self.by_tick = {}
            for i in range(len(rows)):
                if i in self.ident:
                    self.by_tick.setdefault(int(rows[i, 1]), []).append(i)
            self.starts = [(None, k) for k in self.by_tick
                           if k - max(PAST) >= 0 and k + max(FUT) < self.n and (k - 1) in self.by_tick and (k + 1) in self.by_tick]

    def read(self, a, b):
        """{tick: 1280 x 720 BGR} for ticks a .. b-1, one seek and one pass"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(str(self.d / 'video.mp4'))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, a)
        out = {}
        for j in range(a, b):
            ok, f = self.cap.read()
            if not ok:
                break
            out[j] = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        return out

    def frames(self, k, cache=None):
        """{offset: 1280 x 720 BGR} for t-5 .. t+5 (from cache: {tick: frame} when given)"""
        want = [0] + [-p for p in PAST] + list(FUT)
        got = cache if cache is not None and all(k + o in cache for o in want) else self.read(k - max(PAST), k + max(FUT) + 1)
        if not all(k + o in got for o in want):
            return None
        return {o: got[k + o] for o in want}

    def people(self, s, k):
        """{obj: (mask 720 x 1280 uint8, box x1 y1 x2 y2 in 1280 x 720)} of session s at tick k; with glued numbers
        (RA_DS_GLUED) {person: ...} of every session at k, one row per person"""
        out = {}
        sx, sy = W / self.vw, H / self.vh
        rows_k = self.sess[s].get(k, []) if self.ident is None else self.by_tick.get(k, [])
        for r in rows_k:
            key = int(self.M.rows[r, 2]) if self.ident is None else self.ident[r]
            if key in out:
                continue
            x1, y1, x2, y2 = [float(v) for v in self.M.rows[r, 4:8]]
            c = self.M.crop(r).astype(np.uint8)
            X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
            cw, ch = max(1, int(round(c.shape[1] * sx))), max(1, int(round(c.shape[0] * sy)))
            m = np.zeros((H, W), np.uint8)
            cs = cv2.resize(c, (cw, ch), interpolation=cv2.INTER_NEAREST)[:H - Y1, :W - X1]
            m[Y1:Y1 + cs.shape[0], X1:X1 + cs.shape[1]] = cs
            if m.sum() < 30:
                continue
            ys, xs = np.nonzero(m)
            out[key] = (m, np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float32))
        return out

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def glued_ids(d, M):
    """{row of chunks.npz: person} for a source whose people are known across SAM's sessions, else None:
    - a door stretch the owner went through on /doorside with the role 'train': his people -- SAM's tracks linked over
      the session seams (door_side.outlines), cut where he cut (door_side.eff), merged where he merged (root_of);
      tracks he marked 'not a person' have no row here (they are no one to learn);
    - a SAM 3.1 window with v2 targets: the window's person of each row (ReID glued over sessions, v2_prep)."""
    TICK = 0.08
    if d.parent.parent.name == 'sam31_door':
        tag = d.parent.name
        f = ROOT / 'data' / 'door_side' / ('%s.json' % tag.split('_')[1])
        if not f.exists():
            return None
        st = json.load(open(f, encoding='utf-8')).get(tag)
        if not st or not st.get('done') or st.get('role') != 'train':
            return None
        import door_side as DSD
        import sam31_reid as R
        info = json.load(open(d / 'info.json'))
        owned, _ = R.link_seams(M, {int(s_): int(sh) for s_, e_, sh in info['sessions']})
        nop = {str(x) for x in st.get('noperson', [])}
        out = {}
        for piece, rs in owned.items():
            for r in rs:
                e = DSD.eff(st.get('cuts'), piece, int(M.rows[r, 1]) * TICK)
                root = DSD.root_of(st.get('merge', {}), e)
                if e in nop or root in nop or str(piece) in nop:
                    continue
                out[int(r)] = 'o' + root
        return out
    v2 = ROOT / 'data' / 'v2' / d.parent.name / d.name / 'rows.npz'
    if v2.exists():
        z = np.load(v2)
        return {int(r): 'p%d' % int(p) for r, p in zip(z['r'], z['person'])}
    return None


def spoil(masks, rng):
    """the people of t-1 as the network will see its own answer live: some missing, a ghost, shifted, ragged"""
    out = np.zeros((H, W), np.uint8)
    keep = [m for m in masks if rng.random() > 0.15]
    for m in keep:
        out |= m
    if masks and rng.random() < 0.1:                          # a ghost: someone's shape somewhere else
        m = masks[rng.randrange(len(masks))]
        out |= np.roll(m, (rng.randint(-200, 200), rng.randint(-300, 300)), (0, 1))
    if rng.random() < 0.7:
        out = np.roll(out, (rng.randint(-10, 10), rng.randint(-10, 10)), (0, 1))
    if rng.random() < 0.5:
        k = np.ones((rng.choice((3, 5, 7)),) * 2, np.uint8)
        out = cv2.dilate(out, k) if rng.random() < 0.5 else cv2.erode(out, k)
    return out


def sample(src, rng, train=True, at=None, cache=None):
    """one training example: img C x 736 x 1280 uint8, idx 184 x 320 (0 = nobody, j+1 = the j-th person),
    boxes N x 4 (cx cy w h of 1280 x 736), off N x 4 (to t-1, to t+1; input px), offv N x 2"""
    s, k = at or rng.choice(src.starts)
    fr = src.frames(k, cache)
    if fr is None:
        return None
    now, before, after = src.people(s, k), src.people(s, k - 1), src.people(s, k + 1)
    prev = spoil([m for m, _ in before.values()], rng) if train else \
        (np.bitwise_or.reduce([m for m, _ in before.values()]) if before else np.zeros((H, W), np.uint8))
    img = channels(fr, background(src.day, src.cam), prev)
    if train and rng.random() < 0.5:                           # light: the frame and the hall both, the same way
        g = rng.uniform(0.8, 1.2)
        img[:6] = np.clip(img[:6].astype(np.float32) * g, 0, 255).astype(np.uint8)
    order = sorted(now, key=lambda o: -int(now[o][0].sum()))   # big first, small painted over them
    idx = np.zeros((MH, MW), np.uint8)
    boxes, off, offv = [], [], []
    for j, o in enumerate(order[:254]):
        m, b = now[o]
        small = cv2.resize(m, (MW, H // 4), interpolation=cv2.INTER_NEAREST) > 0
        idx[:H // 4][small] = j + 1
        c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])
        boxes.append([c[0] / W, c[1] / HP, (b[2] - b[0]) / W, (b[3] - b[1]) / HP])
        o4, v = np.zeros(4, np.float32), [False, False]
        for i, other in enumerate((before, after)):
            if o in other:
                bb = other[o][1]
                dd = [(bb[0] + bb[2]) / 2 - c[0], (bb[1] + bb[3]) / 2 - c[1]]
                if max(abs(dd[0]), abs(dd[1])) <= MAX_STEP:      # a jump is a wrong glue, not a step: no target
                    o4[2 * i:2 * i + 2] = dd
                    v[i] = True
        off.append(o4)
        offv.append(v)
    return {'img': img, 'idx': idx, 'boxes': np.array(boxes, np.float32).reshape(-1, 4),
            'off': np.array(off, np.float32).reshape(-1, 4), 'offv': np.array(offv, bool).reshape(-1, 2)}


def stream(seed, q, test=False):
    """a worker: examples into q forever"""
    rng = random.Random(seed)
    cv2.setNumThreads(1)                                   # 6+ workers x 32 OpenCV threads fought over the processor
    srcs = []
    for d in sources(test):
        try:
            s = Source(d)
        except Exception:
            continue
        if s.starts:
            srcs.append(s)
    weights = [math.sqrt(len(s.starts)) * (3.0 if s.ident is not None and s.d.parent.parent.name == 'sam31_door' else 1.0)
               for s in srcs]
    last = None
    while True:
        src = rng.choices(srcs, weights)[0]
        if last is not None and last is not src:
            last.close()
        last = src
        try:                                               # one pass of 26 frames -> up to 8 examples, 2 ticks apart
            s, k0 = rng.choice(src.starts)
            cache = src.read(k0 - max(PAST), k0 + 15 + max(FUT) + 1)
            by = src.sess[s] if src.ident is None else src.by_tick
            for k in range(k0, k0 + 16, 2):
                if (k - 1) in by and k in by and (k + 1) in by:
                    x = sample(src, rng, at=(s, k), cache=cache)
                    if x is not None:
                        q.put(x)
        except Exception as exc:
            print('sample failed', src.d, exc, flush=True)
            continue


def collate(xs, dev):
    import torch
    img = torch.from_numpy(np.stack([x['img'] for x in xs])).to(dev, non_blocking=True)
    bi, bx, of, ov = [], [], [], []
    for i, x in enumerate(xs):
        n = len(x['boxes'])
        bi.append(np.full(n, i, np.float32))
        bx.append(x['boxes']); of.append(x['off']); ov.append(x['offv'])
    t = lambda a, dt=torch.float32: torch.from_numpy(np.concatenate(a)).to(dev, dtype=dt)
    batch = {'img': img, 'batch_idx': t(bi), 'cls': torch.zeros(sum(len(b) for b in bx), 1, device=dev),
             'bboxes': t(bx), 'masks': torch.from_numpy(np.stack([x['idx'] for x in xs])).to(dev, dtype=torch.long)}
    batch['sem_masks'] = torch.zeros_like(batch['masks'])        # YOLO26's semantic branch: one class, 'person'
    return batch, t(of), t(ov, torch.bool)


# ------------------------------------------------------------------ the model

def build(weights='yolo26s-seg.pt', cfg='yolo26s-seg.yaml'):
    """YOLO26s-seg with C input channels, one class, COCO weights: the first convolution keeps RGB and starts the
    other channels at zero (the start does not spoil what COCO knows); the class layers keep COCO's person."""
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    m = SegmentationModel(cfg, ch=C, nc=1, verbose=False)
    src = YOLO(weights).model.state_dict()
    own = m.state_dict()
    put = {}
    for k_, v in src.items():
        if k_ not in own:
            continue
        o = own[k_]
        if v.shape == o.shape:
            put[k_] = v
        elif v.dim() == 4 and v.shape[1] == 3 and o.shape[1] == C and v.shape[0] == o.shape[0]:
            w = torch.zeros_like(o)
            w[:, :3] = v
            put[k_] = w
        elif v.shape[0] == 80 and o.shape[0] == 1 and v.shape[1:] == o.shape[1:]:
            put[k_] = v[:1]
    m.load_state_dict(put, strict=False)
    m.args = get_cfg(overrides={'overlap_mask': True, 'epochs': 30})
    m.n_loaded = (len(put), len(own))
    return m


class DoorSeg:
    """the segmenter + the displacement head on its stride-8 features (taken on the way into YOLO's head)"""

    def __init__(self, dev, weights='yolo26s-seg.pt'):
        import torch
        import torch.nn as nn
        self.yolo = build(weights).to(dev)
        self.feat = None
        self.yolo.model[-1].register_forward_pre_hook(lambda mod, a: setattr(self, 'feat', a[0][0]))
        with torch.no_grad():
            self.yolo.eval()
            self.yolo(torch.zeros(1, C, HP, W, device=dev))
        c3 = self.feat.shape[1]
        self.off = nn.Sequential(nn.Conv2d(c3, 96, 3, padding=1), nn.SiLU(), nn.Conv2d(96, 96, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(96, 4, 1)).to(dev)
        nn.init.zeros_(self.off[-1].weight)
        nn.init.zeros_(self.off[-1].bias)

    def parameters(self):
        return list(self.yolo.parameters()) + list(self.off.parameters())

    def state(self):
        return {'yolo': self.yolo.state_dict(), 'off': self.off.state_dict()}

    def load(self, st):
        self.yolo.load_state_dict(st['yolo'])
        self.off.load_state_dict(st['off'])


def offset_loss(pred, batch, off, offv):
    """smooth L1 of the displacements on the stride-8 cells of each person (the cell's person from the mask index)"""
    import torch
    import torch.nn.functional as F
    idx8 = batch['masks'][:, ::2, ::2]                             # 92 x 160: the person of each cell
    B = idx8.shape[0]
    bi = batch['batch_idx'].long()
    starts = torch.zeros(B + 1, dtype=torch.long, device=bi.device)
    starts.scatter_add_(0, bi + 1, torch.ones_like(bi))
    starts = starts.cumsum(0)
    b_, y_, x_ = torch.nonzero(idx8 > 0, as_tuple=True)
    if len(b_) == 0:
        return pred.sum() * 0
    g = starts[b_] + idx8[b_, y_, x_] - 1                           # the global target of the cell
    tgt = off[g] / OFF_SCALE                                        # cells x 4
    val = offv[g].repeat_interleave(2, 1).float()
    p = pred[b_, :, y_, x_].float()
    l = F.smooth_l1_loss(p, tgt, reduction='none') * val
    return l.sum() / val.sum().clamp(min=1)


def predict(net, img, thr=0.4):
    """img 1 x C x 736 x 1280 float -> list of (mask 184 x 320 bool, box xyxy, score, displacement to t-1 (dx dy), to t+1)"""
    import torch
    out = net.yolo(img)
    det, proto = out[0][0][0], out[0][1][0]                       # 300 x 38, 32 x 184 x 320
    off = net.off(net.feat)[0].float() * OFF_SCALE                 # 4 x 92 x 160
    keep = det[:, 4] >= thr
    det = det[keep].float()
    if not len(det):
        return []
    m = (det[:, 6:] @ proto.float().flatten(1)).view(-1, MH, MW).sigmoid()
    res = []
    for i in range(len(det)):
        x1, y1, x2, y2 = det[i, :4].tolist()
        mm = torch.zeros(MH, MW, dtype=torch.bool, device=m.device)
        a1, b1, a2, b2 = max(0, int(x1 / 4)), max(0, int(y1 / 4)), min(MW, int(math.ceil(x2 / 4))), min(MH, int(math.ceil(y2 / 4)))
        mm[b1:b2, a1:a2] = m[i, b1:b2, a1:a2] > 0.5
        cells = mm[::2, ::2]
        d = off[:, cells].mean(1) if cells.any() else off[:, min(GH - 1, int((y1 + y2) / 16)), min(GW - 1, int((x1 + x2) / 16))]
        res.append((mm.cpu().numpy(), np.array([x1, y1, x2, y2]), float(det[i, 4]), d[:2].cpu().numpy(), d[2:].cpu().numpy()))
    return res


def _match(pred, truth, thr=0.5):
    """greedy by IoU: [(pred i, truth j, iou)]"""
    pairs = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            inter = np.logical_and(p, t).sum()
            if inter:
                u = np.logical_or(p, t).sum()
                pairs.append((inter / u, i, j))
    out, ui, uj = [], set(), set()
    for iou, i, j in sorted(pairs, reverse=True):
        if iou < thr:
            break
        if i in ui or j in uj:
            continue
        ui.add(i); uj.add(j); out.append((i, j, iou))
    return out


def evaluate(net, n=150, seed=7):
    """against the SAM 3.1 teacher on 23.09 (never trained on): every tick predicted twice in a row (t-1 then t), the
    mask of the people of t-1 is the network's own answer, as live"""
    import torch
    rng = random.Random(seed)
    srcs = [s for s in (Source(d) for d in sources(test=True)) if s.starts]
    picks = [(src, src.starts[rng.randrange(len(src.starts))]) for src in rng.choices(srcs, k=n)]
    net.yolo.eval()
    tot = dict(truth=0, pred=0, hit=0, iou=0.0, links=0, links_ok=0)
    times = []
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for src, (s, k) in picks:
            fr0, fr1 = src.frames(k - 1), src.frames(k)
            if fr0 is None or fr1 is None or k - 2 < 0:
                continue
            bg = background(src.day, src.cam)
            before2 = src.people(s, k - 2)
            prev0 = np.bitwise_or.reduce([m for m, _ in before2.values()]) if before2 else np.zeros((H, W), np.uint8)
            x0 = torch.from_numpy(channels(fr0, bg, prev0))[None].cuda().float() / 255
            p0 = predict(net, x0)
            prev1 = np.zeros((H, W), np.uint8)
            for m, _, _, _, _ in p0:
                prev1 |= cv2.resize(m[:H // 4].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
            x1 = torch.from_numpy(channels(fr1, bg, prev1))[None].cuda().float() / 255
            torch.cuda.synchronize(); t0 = time.time()
            p1 = predict(net, x1)
            torch.cuda.synchronize(); times.append(time.time() - t0)
            tr1, tr0 = src.people(s, k), src.people(s, k - 1)
            small = lambda m: cv2.resize(m, (MW, H // 4), interpolation=cv2.INTER_NEAREST) > 0
            ids1, ids0 = list(tr1), list(tr0)
            T1 = [np.pad(small(tr1[o][0]), ((0, MH - H // 4), (0, 0))) for o in ids1]
            T0 = [np.pad(small(tr0[o][0]), ((0, MH - H // 4), (0, 0))) for o in ids0]
            M1, M0 = _match([p[0] for p in p1], T1), _match([p[0] for p in p0], T0)
            tot['truth'] += len(T1); tot['pred'] += len(p1); tot['hit'] += len(M1); tot['iou'] += sum(x[2] for x in M1)
            who0 = {i: ids0[j] for i, j, _ in M0}
            for i, j, _ in M1:                                     # the link: where the displacement points at t-1
                if ids1[j] not in set(ids0) or not p0:
                    continue
                b = p1[i][1]
                c = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]) + p1[i][3]
                cs = np.array([[(q[1][0] + q[1][2]) / 2, (q[1][1] + q[1][3]) / 2] for q in p0])
                near = int(np.argmin(((cs - c) ** 2).sum(1)))
                tot['links'] += 1
                tot['links_ok'] += who0.get(near) == ids1[j]
            src.close()
    net.yolo.train()
    f = tot['hit'] / max(1, tot['truth']); p = tot['hit'] / max(1, tot['pred'])
    return {'found': round(f, 4), 'precision': round(p, 4), 'f1': round(2 * f * p / max(1e-9, f + p), 4),
            'mask_iou': round(tot['iou'] / max(1, tot['hit']), 4), 'links': round(tot['links_ok'] / max(1, tot['links']), 4),
            'n_links': tot['links'], 'truth': tot['truth'], 'ms_per_frame': round(1000 * float(np.median(times)), 1) if times else None}


def train(name, stop='20:50', batch=6, workers=8, lr=2e-4, total=60000):
    import multiprocessing as mp
    import torch
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = 'cuda'
    run = ROOT / 'runs' / name
    run.mkdir(parents=True, exist_ok=True)
    net = DoorSeg(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    step = 0
    if (run / 'last.pt').exists():
        st = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        net.load(st['model']); opt.load_state_dict(st['opt']); step = st['step']
        say(run, 'продолжаю с шага %d' % step)
    elif os.environ.get('RA_DS_INIT'):                          # the next version: the previous one's weights, a fresh optimizer
        net.load(torch.load(os.environ['RA_DS_INIT'], map_location='cpu', weights_only=False)['model'])
        lr = float(os.environ.get('RA_DS_LR', lr))
        say(run, 'старт от %s, lr %g, склеенные номера: %s, до %s' % (os.environ['RA_DS_INIT'], lr,
                                                                    os.environ.get('RA_DS_GLUED') == '1', stop))
    else:
        say(run, 'старт: YOLO26s-seg на %d каналах (кадр, пустой зал, разницы t-5..t+5, люди t-1) 1280x736 + смещения к t-1/t+1, '
                 'учитель SAM 3.1, пачка %d, до %s' % (C, batch, stop))
    q = mp.Queue(maxsize=64)
    ps = [mp.Process(target=stream, args=(1000 * step + i, q), daemon=True) for i in range(workers)]
    for p in ps:
        p.start()
    hh, mm = map(int, stop.split(':'))
    t_end = time.time() + ((hh * 60 + mm) - (time.localtime().tm_hour * 60 + time.localtime().tm_min)) % 1440 * 60
    t_save, t_eval, acc, pool, POOL = time.time(), time.time(), {}, [], 96
    crit_updates = step // 2000
    while time.time() < t_end and not (run / 'STOP').exists():
        while len(pool) < POOL:                              # examples of one pass are neighbours: mix them
            pool.append(q.get())
        xs = []
        for _ in range(batch):
            i = random.randrange(len(pool))
            xs.append(pool[i]); pool[i] = q.get()
        b, off, offv = collate(xs, dev)
        f = min(1.0, (step + 1) / 500) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total))))
        for g in opt.param_groups:
            g['lr'] = lr * f
        with torch.autocast('cuda', dtype=torch.bfloat16):
            img = b['img'].float() / 255
            preds = net.yolo(img)
            loss, items = net.yolo.loss(b, preds)
            lo = offset_loss(net.off(net.feat), b, off, offv)
        total_loss = loss.sum() + 2.0 * batch * lo
        opt.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
        opt.step()
        step += 1
        while step // 2000 > crit_updates:                       # YOLO26: one-to-many weight decays as epochs pass
            net.yolo.criterion.update(); crit_updates += 1
        for k_, v in dict(items, off=lo.detach()).items():
            acc[k_] = acc.get(k_, 0.0) + float(v)
        acc['n'] = acc.get('n', 0) + 1
        if step % 100 == 0:
            with open(run / 'log.jsonl', 'a') as fo:
                fo.write(json.dumps(dict({k_: round(v / acc['n'], 4) for k_, v in acc.items() if k_ != 'n'}, step=step,
                                         lr=round(lr * f, 7), t=round(time.time()))) + '\n')
            acc = {}
        if time.time() - t_save > 600:
            torch.save({'model': net.state(), 'opt': opt.state_dict(), 'step': step}, run / 'last.tmp')
            os.replace(run / 'last.tmp', run / 'last.pt')
            t_save = time.time()
        if time.time() - t_eval > EVAL_MIN * 60:
            r = evaluate(net)
            say(run, 'шаг %d: против SAM 3.1 на 23.09 %s' % (step, json.dumps(r)))
            torch.save({'model': net.state(), 'step': step, 'eval': r}, run / ('step%d.pt' % step))
            t_eval = time.time()
    torch.save({'model': net.state(), 'opt': opt.state_dict(), 'step': step}, run / 'last.tmp')
    os.replace(run / 'last.tmp', run / 'last.pt')
    r = evaluate(net)
    say(run, 'стоп на шаге %d: против SAM 3.1 на 23.09 %s' % (step, json.dumps(r)))
    for p in ps:
        p.terminate()


def smoke():
    import torch
    dev = 'cuda'
    rng = random.Random(0)
    srcs = [Source(d) for d in sources()[:3]]
    for s in srcs:
        print(s.d, s.vw, s.vh, s.n, len(s.starts), s.day, s.cam)
    t0 = time.time()
    xs = []
    while len(xs) < 4:
        x = sample(rng.choice([s for s in srcs if s.starts]), rng)
        if x is not None:
            xs.append(x)
    print('4 samples %.2f s' % (time.time() - t0), xs[0]['img'].shape, [len(x['boxes']) for x in xs], xs[0]['off'][:2], xs[0]['offv'][:2])
    cv2.imwrite(str(ROOT / 'data' / 'logs' / 'door_seg_smoke.jpg'),
                np.concatenate([np.concatenate([xs[0]['img'][min(i, C - 1)] for i in range(j, j + 6)], 1) for j in (0, 6, 12)], 0)[::2, ::2])
    net = DoorSeg(dev)
    print('loaded %d of %d tensors, P3 %s' % (net.yolo.n_loaded + (tuple(net.feat.shape),)))
    net.yolo.train()
    batch, off, offv = collate(xs, dev)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        img = batch['img'].float() / 255
        preds = net.yolo(img)
        print('preds', type(preds), list(preds.keys()) if isinstance(preds, dict) else [type(p) for p in preds])
        loss, items = net.yolo.loss(batch, preds)
        lo = offset_loss(net.off(net.feat), batch, off, offv)
    print('loss', loss, items, 'offsets', lo)
    (loss.sum() + lo).backward()
    print('peak %.2f GB' % (torch.cuda.max_memory_allocated() / 1e9))


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'smoke':
        smoke()
    elif a[0] == 'train':
        train(a[1], *(a[2:3]))
    elif a[0] == 'eval':
        import torch
        net = DoorSeg('cuda')
        net.load(torch.load(a[1], map_location='cpu', weights_only=False)['model'])
        print(json.dumps(evaluate(net)), flush=True)
    elif a[0] == 'eval0':                                  # the untrained start (COCO weights), for the record
        print(json.dumps(evaluate(DoorSeg('cuda'), n=int(a[1]) if len(a) > 1 else 40)), flush=True)
