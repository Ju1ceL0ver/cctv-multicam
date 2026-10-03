"""Frames and targets for the slot model (slot_model.py).

A sample is one frame of the teacher+SAM drafts (1280x720) with what the model has to copy:
  x        5 x 608 x 1088: RGB and two background heat maps, clipped and normalised (norm.json)
  masks    N x 152 x 272 (stride 4), one per person of the draft's label map (>= MIN_PX pixels)
  boxes    N x 4, cx cy w h in 0..1 of the output frame
  cloth    N x 3840, shape N x 1024: the ReID teachers' vectors (TransReID global + 4 local, CSCI); has_reid N
  inout    N: 0 outside, 1 inside, 2 doorway, -1 unknown (the owner's /inout answers)
  radio    512 x 38 x 68 (stride 16): C-RADIOv4-H PCA-512 features of the frame; radio_valid 38 x 68

Heat maps are made from the frame before any colour change: d_file = |frame - background of its
15-minute file|, z_days = |frame - multi-day median| / multi-day spread (other days of the camera;
'all' for days never trained on). Augmentation is one affine (scale, shift) applied to everything
alike -- no flip: the place in the frame says inside or outside -- plus brightness/contrast on RGB."""
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
SEG = DATA / 'seg_datasets'
W, H = 1088, 608
SW, SH = 1280, 720                  # the drafts' frame size
MIN_PX = 60
CLIP = {'d_file': 75.0, 'z_days': 28.0}     # 99.9 % of the training pixels (norm.json)
WL = np.array([0.5, 1.0, 1.0], np.float32)
INOUT = {2: 0, 0: 0, 1: 1, 3: 2}   # owner's key -> class; 4 reflection, 5 not a person: not learned


def lab(img):
    import cv2
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)


def heat(frame_bgr, bg_bgr, mu, sd):
    """The two background heat maps at the frame's own size."""
    L = lab(frame_bgr)
    d1 = np.sqrt((((L - lab(bg_bgr)) * WL) ** 2).sum(2))
    d2 = np.sqrt(((((L - mu) / sd) * WL) ** 2).sum(2))
    return np.minimum(d1, CLIP['d_file']), np.minimum(d2, CLIP['z_days'])


class Frames:
    """Everything the loader needs to find, by frame id (day_cam_HHMMSS_NNNN_sec)."""

    def __init__(self, root=DATA):
        root = Path(root)
        self.root = root
        norm = json.load(open(root / 'slot_prep' / 'norm.json'))
        self.mean = np.array(norm['mean'], np.float32)
        self.std = np.array(norm['std'], np.float32)
        # taken before clipping at 99.9 %: close enough for the clipped channels
        self.drafts = {}
        for d in (root / 'seg_datasets').glob('*/drafts'):
            for p in d.glob('*.png'):
                self.drafts.setdefault(p.stem, p)
        self.ratings = {}
        rp = root / 'rate' / 'ratings.json'
        if rp.exists():
            r = json.load(open(rp))
            self.ratings = {k: (v['score'] if isinstance(v, dict) else v) for k, v in r.get('ratings', r).items()}
        self.inout = {}
        ip = root / 'inout' / 'labels.json'
        if ip.exists():
            lab_ = json.load(open(ip))
            for sid, v in lab_.get('labels', lab_).items():
                k = INOUT.get(v.get('label'))
                if k is not None:
                    frame, pv = sid.rsplit('_p', 1)
                    self.inout.setdefault(frame, {})[int(pv)] = k
        self._md = {}
        self.teacher_dims = (3840, 1024)            # TransReID ViT: global + 4 local (JPM) = 5 x 768; CSCI 1024
        probe = next((root / 'teacher_emb').glob('*.npz'), None) if (root / 'teacher_emb').exists() else None
        if probe is not None:
            z = np.load(probe)
            self.teacher_dims = (int(z['cloth'].shape[1]), int(z['shape'].shape[1]))

    def split(self, name):
        return sorted(p.stem for p in (self.root / 'seg_datasets' / 'rf_20260925' / name / 'images').glob('*.jpg'))

    def image(self, ident):
        for s in ('train', 'valid'):
            p = self.root / 'seg_datasets' / 'rf_20260925' / s / 'images' / (ident + '.jpg')
            if p.exists():
                return p
        if not hasattr(self, '_images'):                   # frames kept outside the training set (test400, excluded ...)
            self._images = {}
            for p in (self.root / 'seg_datasets').glob('*/**/images/*.jpg'):
                self._images.setdefault(p.stem, p)
        return self._images.get(ident)

    def background(self, ident):
        return self.root / 'seg_datasets' / 'backgrounds' / (ident.rsplit('_', 1)[0] + '.jpg')

    def multiday(self, cam, day):
        key = '%s_%s' % (cam, day)
        if key not in self._md:
            p = self.root / 'slot_prep' / 'multiday' / (key + '.npz')
            if not p.exists():
                p = self.root / 'slot_prep' / 'multiday' / ('%s_all.npz' % cam)
            z = np.load(p)
            import cv2
            self._md[key] = tuple(cv2.resize(z[k].astype(np.float32), (SW, SH), interpolation=cv2.INTER_LINEAR) for k in ('mu', 'sd'))
        return self._md[key]


def affine(rng, train):
    """Source (1280x720) -> output (1088x608) pixels: resize, and when training a random zoom and shift."""
    s = min(W / SW, H / SH)
    if not train:
        return np.array([[s, 0, (W - SW * s) / 2], [0, s, (H - SH * s) / 2]], np.float32)
    z = s * rng.uniform(0.85, 1.25)
    tx = (W - SW * z) / 2 + rng.uniform(-0.06, 0.06) * W
    ty = (H - SH * z) / 2 + rng.uniform(-0.06, 0.06) * H
    return np.array([[z, 0, tx], [0, z, ty]], np.float32)


def load(frames, ident, train=False, rng=None, radio_dir=None, emb_dir=None, image=None, label=None, background=None, cam_day=None):
    """One sample as numpy arrays (see the module text). image/label/background override the lookups
    (the owner's /paint frames)."""
    import cv2
    rng = rng or np.random.default_rng()
    img = cv2.imread(str(image or frames.image(ident)))
    if img.shape[:2] != (SH, SW):
        img = cv2.resize(img, (SW, SH), interpolation=cv2.INTER_AREA)
    day, cam = cam_day or ident.split('_')[:2]
    bgp = background or frames.background(ident)
    mu, sd = frames.multiday(cam, day)
    bg = cv2.imread(str(bgp)) if bgp and os.path.exists(bgp) else None
    if bg is None:                                                  # no background of its own: the multi-day median
        bg = cv2.cvtColor(np.clip(mu, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    elif bg.shape[:2] != (SH, SW):
        bg = cv2.resize(bg, (SW, SH), interpolation=cv2.INTER_AREA)
    d1, d2 = heat(img, bg, mu, sd)
    lp = label or frames.drafts.get(ident)
    L = cv2.imread(str(lp), cv2.IMREAD_UNCHANGED) if lp else np.zeros((SH, SW), np.uint8)
    if L.ndim == 3:
        L = L[:, :, 0]
    if L.shape != (SH, SW):
        L = cv2.resize(L, (SW, SH), interpolation=cv2.INTER_NEAREST)
    M = affine(rng, train)
    rgb = img[:, :, ::-1].astype(np.float32) / 255.0
    if train:                                                       # colour on RGB only, after the heat maps
        rgb = np.clip((rgb - 0.5) * rng.uniform(0.8, 1.2) + 0.5 + rng.uniform(-0.08, 0.08), 0, 1)
    stack = np.concatenate([rgb, d1[..., None], d2[..., None]], 2)
    x = cv2.warpAffine(stack, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    inside = cv2.warpAffine(np.ones((SH, SW), np.float32), M, (W, H), flags=cv2.INTER_NEAREST)
    x = (x - frames.mean) / frames.std
    x[inside < 0.5] = 0.0
    Lw = cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_NEAREST)
    values = [v for v in np.unique(L) if v and (L == v).sum() >= MIN_PX]
    masks, boxes, keep = [], [], []
    for v in values:
        m = (Lw == v)
        if m.sum() < 20:                                            # pushed out of the frame by the zoom
            continue
        ys, xs = np.nonzero(m)
        x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        boxes.append([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
        masks.append(cv2.resize(m.astype(np.float32), (W // 4, H // 4), interpolation=cv2.INTER_AREA))
        keep.append(int(v))
    n = len(keep)
    out = {'x': x.transpose(2, 0, 1).astype(np.float32),
           'masks': np.stack(masks).astype(np.float32) if n else np.zeros((0, H // 4, W // 4), np.float32),
           'boxes': np.array(boxes, np.float32).reshape(-1, 4), 'values': np.array(keep, np.int32),
           'cloth': np.zeros((n, frames.teacher_dims[0]), np.float32), 'shape': np.zeros((n, frames.teacher_dims[1]), np.float32), 'has_reid': np.zeros(n, bool),
           'inout': np.full(n, -1, np.int64)}
    io = frames.inout.get(ident, {})
    for k, v in enumerate(keep):
        out['inout'][k] = io.get(v, -1)
    ep = Path(emb_dir or frames.root / 'teacher_emb') / (ident + '.npz')
    if ep.exists():
        z = np.load(ep)
        at = {int(v): k for k, v in enumerate(z['labels'])}
        for k, v in enumerate(keep):
            if v in at:
                out['cloth'][k] = z['cloth'][at[v]]; out['shape'][k] = z['shape'][at[v]]; out['has_reid'][k] = True
    rp = Path(radio_dir or frames.root / 'radio_feats') / (ident + '.npy')
    gh, gw = H // 16, W // 16
    if rp.exists():
        R = np.load(rp).astype(np.float32)                          # 72 x 128 x 512 over the whole frame
        # radio cell (u, v) centre -> source px -> output px -> output cell
        S1 = np.array([[SW / R.shape[1], 0, SW / R.shape[1] / 2 - 0.5], [0, SH / R.shape[0], SH / R.shape[0] / 2 - 0.5], [0, 0, 1]], np.float32)
        S2 = np.array([[1 / 16, 0, -0.5 + 0.5 / 16], [0, 1 / 16, -0.5 + 0.5 / 16], [0, 0, 1]], np.float32)
        Mr = (S2 @ np.vstack([M, [0, 0, 1]]) @ S1)[:2]
        out['radio'], out['radio_valid'] = resample(R, Mr, gw, gh)
    else:
        out['radio'] = np.zeros((512, gh, gw), np.float16)
        out['radio_valid'] = np.zeros((gh, gw), bool)
    return out


def resample(R, M, gw, gh):
    """R (h x w x C) seen through the affine M (R's cells -> output cells), bilinear, as C x gh x gw
    float16 plus the mask of output cells that fall inside R. numpy only: OpenCV 5 refuses > 128 channels."""
    A = np.linalg.inv(np.vstack([M, [0, 0, 1]]))[:2]
    yy, xx = np.mgrid[0:gh, 0:gw].astype(np.float32)
    sx = A[0, 0] * xx + A[0, 1] * yy + A[0, 2]
    sy = A[1, 0] * xx + A[1, 1] * yy + A[1, 2]
    h, w = R.shape[:2]
    valid = (sx >= -0.5) & (sx <= w - 0.5) & (sy >= -0.5) & (sy <= h - 0.5)
    sx = np.clip(sx, 0, w - 1); sy = np.clip(sy, 0, h - 1)
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    x1 = np.minimum(x0 + 1, w - 1); y1 = np.minimum(y0 + 1, h - 1)
    fx = (sx - x0)[..., None]; fy = (sy - y0)[..., None]
    out = (R[y0, x0] * (1 - fx) * (1 - fy) + R[y0, x1] * fx * (1 - fy) + R[y1, x0] * (1 - fx) * fy + R[y1, x1] * fx * fy)
    return out.transpose(2, 0, 1).astype(np.float16), valid


def train_ids(frames, min_rating=2):
    """Training frames: rf_20260925/train with a draft, minus those the owner rated below min_rating."""
    return [i for i in frames.split('train') if i in frames.drafts and frames.ratings.get(i, 5) >= min_rating]


class Dataset:
    def __init__(self, frames, ids, train):
        self.frames, self.ids, self.train = frames, ids, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        rng = np.random.default_rng() if self.train else np.random.default_rng(k)
        return load(self.frames, self.ids[k], self.train, rng)


class Paint:
    """The owner's painted frames (/paint) outside the exam day: his masks are the targets. Their recording
    file's background when there is one, else the multi-day median of the camera."""

    def __init__(self, frames, train, exclude_day='20260918', exclude_ids=()):
        import json as _json
        paint = frames.root / 'paint'
        state = _json.load(open(paint / 'state.json', encoding='utf-8'))
        man = _json.load(open(paint / 'manifest.json', encoding='utf-8'))['items']
        self.items = [it for it in man if state.get(it['id'], {}).get('done') and it.get('day') != exclude_day
                      and it['id'] not in set(exclude_ids) and (paint / ('%s_mask.png' % it['id'])).exists()]
        self.frames, self.train, self.paint = frames, train, paint

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        it = self.items[k]
        rng = np.random.default_rng() if self.train else np.random.default_rng(k)
        seg = it.get('segment')
        bg = self.frames.root / 'seg_datasets' / 'backgrounds' / ('%s_%s_%s.jpg' % (it['day'], it['cam'], Path(seg).stem)) if seg else None
        return load(self.frames, 'paint_' + it['id'], self.train, rng, image=self.paint / ('%s.jpg' % it['id']),
                    label=self.paint / ('%s_mask.png' % it['id']), background=bg, cam_day=(it['day'], it['cam']))


def collate(batch):
    import torch
    x = torch.from_numpy(np.stack([b['x'] for b in batch]))
    radio = torch.from_numpy(np.stack([b['radio'] for b in batch]))
    valid = torch.from_numpy(np.stack([b['radio_valid'] for b in batch]))
    targets = [{k: torch.from_numpy(v) for k, v in b.items() if k not in ('x', 'radio', 'radio_valid')} for b in batch]
    return x, radio, valid, targets
