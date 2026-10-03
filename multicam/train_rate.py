"""A light model that scores a teacher+SAM draft 1..5 the way the owner does on /rate, so only
good drafts go into distillation.

Input, 11 channels at 640x360: the frame (RGB), the draft (people mask and the outlines between
people), the three background differences of rate_features.py (5-min selective, 30-min
median, other days' median over their MAD spread) and the difference to the prebuilt background
of the frame's 15-minute file (data/seg_datasets/backgrounds), plus two maps of where the draft and
the backgrounds disagree (see `disagreement`). All are cheap to keep up to date
in real time. Score 0 ("no people / not judgeable") is left out: the owner's scale is 1..5.

Backbone: MobileNetV3-Large from torchvision's LR-ASPP, pretrained for segmentation on COCO
(VOC classes, person among them) -- light (~3 M) and trained on the task next to ours. The first
convolution takes 11 channels: the RGB weights are kept, the new channels start small.

Split: by 15-minute recording file (neighbouring frames of one file look alike), fixed seed.
Loss: MSE on the score. Every epoch is printed and written to <out>/log.txt and epochs.jsonl.

  python train_rate.py [--task reg|cls] [--out data/rate/model] [--epochs 60] [--channels rgb,draft,diff] [--backbone lraspp|student]
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
W, H = 640, 360
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
DIFF_NAMES = ('short_5min', 'long_30min', 'days_mad', 'file_bg')
BG_DIR = os.path.join(HERE, 'data', 'seg_datasets', 'backgrounds')


def group_of(ident):
    """The 15-minute file a frame comes from: day_cam_HHMMSS_NNNN."""
    return ident.rsplit('_', 1)[0]


def split(ids, scores, test_frac=0.2, seed=0):
    """Whole files go to test until it holds test_frac of the frames; files are taken in a fixed
    shuffled order, and a file is skipped when its scores would overfill a score class in test."""
    groups = {}
    for i in ids:
        groups.setdefault(group_of(i), []).append(i)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    total = {s: sum(1 for i in ids if scores[i] == s) for s in set(scores[i] for i in ids)}
    want = {s: test_frac * n for s, n in total.items()}
    have = {s: 0 for s in total}
    test = set()
    for g in order:
        if len(test) >= test_frac * len(ids):
            break
        add = {}
        for i in groups[g]:
            add[scores[i]] = add.get(scores[i], 0) + 1
        if all(have[s] + n <= want[s] + 1 for s, n in add.items()):
            test.update(groups[g])
            for s, n in add.items():
                have[s] += n
    return sorted(i for i in ids if i not in test), sorted(test)


def draft_channels(label):
    """uint8 label map (0 = nobody) -> people mask and the outlines of every person, both 0/255."""
    import cv2
    k = np.ones((3, 3), np.uint8)
    edge = cv2.dilate(label, k) != cv2.erode(label, k)
    return ((label > 0) * 255).astype(np.uint8), (edge * 255).astype(np.uint8)


def file_background(ident):
    """The background of the frame's 15-minute file built on 25-26.09 (bg_build: the student's
    people cut out, frames of the whole file averaged), 640x360 BGR, or None."""
    import cv2
    p = os.path.join(BG_DIR, group_of(ident) + '.jpg')
    return cv2.resize(cv2.imread(p), (W, H), interpolation=cv2.INTER_AREA) if os.path.exists(p) else None


def load_item(image, draft, feat, ident=None):
    """-> uint8 (H, W, 5) [B, G, R, mask, outlines] and float16 (4, H, W) differences:
    the three of rate_features.py and |frame - the file's prebuilt background| (zeros if none)."""
    import rate_features
    import cv2
    img = cv2.resize(cv2.imread(str(image)), (W, H), interpolation=cv2.INTER_AREA)
    lab = cv2.imread(str(draft), cv2.IMREAD_UNCHANGED)
    if lab.ndim == 3:
        lab = lab[:, :, 0]
    lab = cv2.resize(lab, (W, H), interpolation=cv2.INTER_NEAREST)
    m, e = draft_channels(lab)
    d = np.load(feat)['d'].astype(np.float32) if feat else np.zeros((3, H, W), np.float32)
    bg = file_background(ident) if ident else None
    fb = rate_features.dist(rate_features.lab(img), rate_features.lab(bg)) if bg is not None else np.zeros((H, W), np.float32)
    sh, sl = student_channels(ident) if ident else (np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8))
    return np.dstack([img, m, e, sh, sl]), np.concatenate([d, fb[None]]).astype(np.float16)


def student_channels(ident):
    """What our student yolo26n-seg sees on the frame (cached by rate_boost.student_view):
    people it is sure of (>= 0.25) and the doubtful ones (0.10-0.25), both 0/255. Disagreeing with
    the draft is the strongest sign of a bad draft the boosting found (26.09)."""
    import cv2
    png, js = (os.path.join(HERE, 'data', 'rate', 'student', ident + e) for e in ('.png', '.json'))
    if not os.path.exists(png):
        z = np.zeros((H, W), np.uint8)
        return z, z
    lab, confs = cv2.imread(png, cv2.IMREAD_UNCHANGED), json.load(open(js))
    sure = [k for k, c in enumerate(confs, 1) if c >= 0.25]
    hi = np.isin(lab, sure)
    return (hi * 255).astype(np.uint8), (((lab > 0) & ~hi) * 255).astype(np.uint8)


def augment(u8, d, rng):
    """The same mirror flip for all channels; colour, gamma, blur, noise, JPEG and grey on the frame
    only; gain, noise and blur on the differences. Nothing moves or is cut off, so the person a
    score is about always stays in the frame."""
    import cv2
    if rng.random() < 0.5:
        u8, d = u8[:, ::-1], d[:, :, ::-1]
    img = u8[:, :, :3].astype(np.float32)          # no crop, turn, zoom or shift: the owner asked for none
    md = np.ascontiguousarray(u8[:, :, 3:])
    d = d.astype(np.float32)
    # frame: brightness, contrast, saturation, gamma, blur, noise
    img = img * rng.uniform(0.75, 1.25) + rng.uniform(-20, 20)
    mean = img.mean()
    img = (img - mean) * rng.uniform(0.75, 1.25) + mean
    grey = img.mean(2, keepdims=True)
    img = grey + (img - grey) * rng.uniform(0.7, 1.3)
    img = np.clip(img, 0, 255)
    img = 255.0 * (img / 255.0) ** rng.uniform(0.8, 1.25)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 1.5))
    img = np.clip(img + rng.normal(0, rng.uniform(0, 6), img.shape), 0, 255)
    if rng.random() < 0.3:                          # compression, as the camera stream sometimes looks
        ok, buf = cv2.imencode('.jpg', img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(35, 90))])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR).astype(np.float32)
    if rng.random() < 0.2:                          # grey frame: the score is about outlines, not colour
        img = np.repeat(img.mean(2, keepdims=True), 3, 2)
    # differences: gain per map and a little noise
    if rng.random() < 0.3:                          # a smoother or sharper background estimate
        d = np.stack([cv2.GaussianBlur(c, (0, 0), rng.uniform(0.6, 2.0)) for c in d])
    d = d * rng.uniform(0.85, 1.15, (len(d), 1, 1)) + rng.normal(0, 0.03, d.shape) * d.std(axis=(1, 2), keepdims=True)
    return img, md, np.maximum(d, 0)


def to_tensor(img, md, d, dstat, channels):
    """-> float32 (C, H, W): ImageNet-normalised RGB, mask and outlines in -1..1, standardised differences."""
    parts = []
    if 'rgb' in channels:
        parts.append(((img[:, :, ::-1] / 255.0 - MEAN) / STD).transpose(2, 0, 1))
    if 'draft' in channels:
        parts.append(md[:, :, :2].transpose(2, 0, 1) / 127.5 - 1.0)
    if 'student' in channels:
        parts.append(md[:, :, 2:4].transpose(2, 0, 1) / 127.5 - 1.0)
    if 'diff' in channels:
        parts.append((np.log1p(d) - dstat[0][:, None, None]) / dstat[1][:, None, None])
    if 'diff' in channels and 'draft' in channels:
        parts.append(disagreement(md[:, :, 0] > 127, d))
    return np.concatenate(parts).astype(np.float32)


def disagreement(mask, d):
    """Where the draft and the backgrounds disagree, both in 0..1: `missed` -- differs from the
    5-minute, 30-minute or file background and no person drawn there (a missed person, the most
    common reason for a low score); `empty` -- a person drawn where nothing differs (a false one,
    a stain). A missed person is 1-2% of the frame; spelled out, the network need not find it."""
    import cv2
    fg = np.clip(np.max(d[[0, 1, 3]], axis=0) / 24.0, 0, 1)
    near = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    return np.stack([fg * ~near, mask * (1 - fg)]).astype(np.float32) * 2 - 1


STUDENT = os.path.join(HERE, 'runs', 'student_seg_all_n', 'weights', 'best.pt')


def _widen(old, n_in):
    """A first convolution for n_in channels: RGB weights kept, the new channels start small."""
    import torch
    import torch.nn as nn
    new = nn.Conv2d(n_in, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
    with torch.no_grad():
        new.weight.zero_()
        new.weight[:, :3] = old.weight
        if n_in > 3:                                 # like one more grey channel
            new.weight[:, 3:] = old.weight.mean(1, keepdim=True) * 0.3
    return new


def lraspp_body(n_in, pretrained=True):
    """MobileNetV3-Large of torchvision's LR-ASPP (COCO/VOC segmentation): {'low': s8, 'high': s16}."""
    from torchvision.models.segmentation import lraspp_mobilenet_v3_large, LRASPP_MobileNet_V3_Large_Weights
    body = lraspp_mobilenet_v3_large(weights=LRASPP_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None).backbone
    body['0'][0] = _widen(body['0'][0], n_in)
    return body


def student_body(n_in, weights=STUDENT):
    """The backbone of our own student yolo26n-seg (layers 0-10, a plain chain): trained on people
    of this very shop, both cameras. {'low': layer 4, s8; 'high': layer 10, s32}. It expects RGB in
    0..1, the input here is ImageNet-normalised, so the RGB channels are turned back first."""
    import torch
    import torch.nn as nn
    from ultralytics import YOLO
    layers = YOLO(weights).model.model[:11]
    layers[0].conv = _widen(layers[0].conv, n_in)

    class Body(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = layers
            self.register_buffer('mean', torch.tensor(MEAN).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(STD).view(1, 3, 1, 1))

        def forward(self, x):
            x = torch.cat([x[:, :3] * self.std + self.mean, x[:, 3:]], 1)
            out = {}
            for i, layer in enumerate(self.layers):
                x = layer(x)
                if i == 4:
                    out['low'] = x
            out['high'] = x
            return out
    body = Body()
    for p in body.parameters():                      # ultralytics loads its weights frozen off; train them all
        p.requires_grad_(True)
    return body


def build_model(n_in, pretrained=True, backbone='lraspp'):
    import torch
    import torch.nn as nn
    body = student_body(n_in) if backbone == 'student' else lraspp_body(n_in, pretrained)
    with torch.no_grad():
        f = body.eval()(torch.zeros(1, n_in, H, W))
    c_lo, c_hi = f['low'].shape[1], f['high'].shape[1]

    class Scorer(nn.Module):
        """The body's features become K maps of 'something wrong here' at stride 8; each map is
        pooled by the mean of its worst 1% cells and by its plain mean, so one missed person in
        a large frame still moves the score (a plain mean over the frame averaged it away)."""
        K = 8

        def __init__(self):
            super().__init__()
            self.body = body
            self.reduce = nn.Sequential(nn.Conv2d(c_hi, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
            self.maps = nn.Sequential(nn.Conv2d(64 + c_lo + n_in, 48, 3, padding=1, bias=False), nn.BatchNorm2d(48),
                                      nn.ReLU(inplace=True), nn.Conv2d(48, self.K, 1))
            self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2 * self.K, 1))

        def error_maps(self, x):
            f = self.body(x)
            lo = f['low']
            hi = nn.functional.interpolate(self.reduce(f['high']), size=lo.shape[2:], mode='bilinear', align_corners=False)
            xs = nn.functional.adaptive_avg_pool2d(x, lo.shape[2:])       # the inputs themselves, at stride 8
            return self.maps(torch.cat([hi, lo, xs], 1))

        def forward(self, x):
            m = self.error_maps(x).flatten(2)
            k = max(1, m.shape[2] // 100)
            z = torch.cat([m.topk(k, dim=2).values.mean(2), m.mean(2)], 1)
            return 3.0 + self.head(z).squeeze(1)      # centred on the middle of the scale

    return Scorer()


def metrics(y, p):
    """Everything the log prints about predictions p against scores y (1..5)."""
    from scipy.stats import pearsonr, spearmanr
    pc = np.clip(p, 1, 5)
    out = {'mse': float(np.mean((pc - y) ** 2)), 'rmse': float(np.sqrt(np.mean((pc - y) ** 2))),
           'mae': float(np.mean(np.abs(pc - y))), 'exact_round': float(np.mean(np.round(pc) == y)),
           'within_1': float(np.mean(np.abs(pc - y) <= 1.0))}
    out['spearman'] = float(spearmanr(y, pc)[0]) if len(set(y)) > 1 else 0.0
    out['pearson'] = float(pearsonr(y, pc)[0]) if len(set(y)) > 1 else 0.0
    good = y >= 4
    if good.any() and (~good).any():                 # ranking good (4-5) above bad (1-3)
        r = pc.argsort().argsort().astype(np.float64)
        out['auc_good'] = float((r[good].sum() - good.sum() * (good.sum() - 1) / 2) / (good.sum() * (~good).sum()))
    for thr in (3.5, 4.0, 4.5):                      # keep drafts predicted >= thr: how clean, how many
        keep = pc >= thr
        out['keep%.1f' % thr] = {'kept': int(keep.sum()), 'precision_good': float(good[keep].mean()) if keep.any() else None,
                                 'recall_good': float(keep[good].mean()) if good.any() else None,
                                 'bad_kept': int((keep & ~good).sum())}
    out['per_score'] = {int(s): {'n': int((y == s).sum()), 'mean_pred': float(pc[y == s].mean()),
                                 'mae': float(np.abs(pc[y == s] - s).mean())} for s in sorted(set(y.tolist()))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'data', 'rate', 'model'))
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--channels', default='rgb,draft,diff')
    ap.add_argument('--test-frac', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--backbone', default='lraspp', choices=['lraspp', 'student'])
    ap.add_argument('--task', default='reg', choices=['reg', 'cls'], help='cls: good (4-5) vs bad (1-3), BCE')
    a = ap.parse_args()
    import torch
    import rate
    channels = a.channels.split(',')
    os.makedirs(a.out, exist_ok=True)
    logf = open(os.path.join(a.out, 'log.txt'), 'a', encoding='utf-8')
    epf = open(os.path.join(a.out, 'epochs.jsonl'), 'w', encoding='utf-8')

    def log(*msg):
        s = ' '.join(str(m) for m in msg)
        print(s, flush=True)
        logf.write(time.strftime('%H:%M:%S ') + s + '\n')
        logf.flush()

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    items, r = rate.index(HERE, force=True), rate.ratings(HERE)
    fdir = os.path.join(HERE, 'data', 'rate', 'features')
    ids = sorted(i for i, v in r.items() if 1 <= v['score'] <= 5 and i in items)
    no_feat = [i for i in ids if not os.path.exists(os.path.join(fdir, i + '.npz'))]
    if 'diff' in channels and no_feat:
        log('no background maps for %d frames, left out: %s' % (len(no_feat), no_feat[:10]))
        ids = [i for i in ids if i not in set(no_feat)]
    scores = {i: r[i]['score'] for i in ids}
    from sklearn.model_selection import StratifiedGroupKFold     # the same split as rate_boost.py
    yy = np.array([scores[i] for i in ids])
    k_tr, k_te = next(StratifiedGroupKFold(5, shuffle=True, random_state=a.seed).split(ids, yy, [group_of(i) for i in ids]))
    tr, te = [ids[k] for k in k_tr], [ids[k] for k in k_te]
    hist = lambda xs: {s: sum(1 for i in xs if scores[i] == s) for s in range(1, 6)}
    log('=' * 100)
    log('run %s  task=%s channels=%s  epochs=%d batch=%d lr=%g seed=%d, split = rate_boost.py stratified by file' % (time.strftime('%Y-%m-%d %H:%M'), a.task, channels, a.epochs, a.batch, a.lr, a.seed))
    log('scored frames 1..5: %d (score 0 left out: %d)' % (len(ids), sum(1 for v in r.values() if v['score'] == 0)))
    log('train %d frames from %d files, scores %s' % (len(tr), len({group_of(i) for i in tr}), hist(tr)))
    log('test  %d frames from %d files, scores %s' % (len(te), len({group_of(i) for i in te}), hist(te)))
    json.dump({'train': tr, 'test': te}, open(os.path.join(a.out, 'split.json'), 'w'), indent=0)

    t0 = time.time()
    data = {i: load_item(items[i][0], items[i][1], os.path.join(fdir, i + '.npz') if 'diff' in channels else None,
                         i if 'diff' in channels else None) for i in ids}
    no_bg = [i for i in ids if not os.path.exists(os.path.join(BG_DIR, group_of(i) + '.jpg'))]
    log('prebuilt file backgrounds: %d of %d frames have one' % (len(ids) - len(no_bg), len(ids)))
    log('loaded %d frames in %.0f s' % (len(data), time.time() - t0))
    nd = len(DIFF_NAMES)
    L = np.stack([np.log1p(data[i][1].astype(np.float32)).reshape(nd, -1)[:, ::97] for i in tr], 1).reshape(nd, -1)
    dstat = np.stack([L.mean(1), L.std(1) + 1e-6]).astype(np.float32)
    log('difference maps (log1p) mean/std on train: %s' % {n: (round(float(m), 3), round(float(s), 3)) for n, m, s in zip(DIFF_NAMES, *dstat)})
    ytr = np.array([scores[i] for i in tr], np.float32)
    yte = np.array([scores[i] for i in te], np.float32)
    base_pred = float(ytr.mean())
    log('baseline "always the train mean %.2f": test MSE %.3f MAE %.3f' % (base_pred, np.mean((yte - base_pred) ** 2), np.mean(np.abs(yte - base_pred))))

    n_in = 3 * ('rgb' in channels) + 2 * ('draft' in channels) + len(DIFF_NAMES) * ('diff' in channels) \
        + 2 * ('diff' in channels and 'draft' in channels) + 2 * ('student' in channels)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(n_in, backbone=a.backbone).to(dev)
    log('model: %s backbone, all layers trained, + error maps, top-1%% pooling, %d inputs, %.2f M parameters, device %s'
        % ({'lraspp': 'LR-ASPP MobileNetV3-Large (COCO/VOC seg)', 'student': 'our student yolo26n-seg (%s)' % STUDENT}[a.backbone], n_in, sum(p.numel() for p in model.parameters()) / 1e6, dev))
    head = [p for n, p in model.named_parameters() if not n.startswith('body.')]
    body = [p for n, p in model.named_parameters() if n.startswith('body.')]
    opt = torch.optim.AdamW([{'params': body, 'lr': a.lr}, {'params': head, 'lr': a.lr * 3}], weight_decay=1e-4)
    steps = a.epochs * ((len(tr) + a.batch - 1) // a.batch)
    warm = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / steps))))
    scaler = torch.amp.GradScaler(enabled=dev == 'cuda')

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(8)                     # cv2 and numpy let go of the GIL: augmenting was 35 s an epoch

    def batch_tensor(batch_ids, aug):
        def one(args):
            i, seed = args
            u8, d = data[i]
            if aug:
                img, md, dd = augment(u8, d, np.random.default_rng(seed))
            else:
                img, md, dd = u8[:, :, :3].astype(np.float32), u8[:, :, 3:], d.astype(np.float32)
            return to_tensor(img, md, dd, dstat, channels)
        xs = list(pool.map(one, [(i, int(rng.integers(1 << 31))) for i in batch_ids]))
        return torch.from_numpy(np.stack(xs)).to(dev, non_blocking=True)

    def predict(batch_ids):
        model.eval()
        out = []
        with torch.no_grad(), torch.autocast(dev, enabled=dev == 'cuda'):
            for k in range(0, len(batch_ids), 32):
                x = batch_tensor(batch_ids[k:k + 32], False)
                out.append(as_score((model(x) + model(x.flip(3))) / 2).float().cpu().numpy())   # + mirror
        return np.concatenate(out)

    def as_score(out):
        """reg: the score itself; cls: 1 + 4 * P(good), so every metric and the keep>=4 line read the
        same (keep>=4 = P(good) >= 0.75)."""
        return out if a.task == 'reg' else 1.0 + 4.0 * torch.sigmoid(out - 3.0)

    best = {'mse': 1e9, 'auc': -1}
    for ep in range(1, a.epochs + 1):
        te0 = time.time()
        model.train()
        order = list(tr)
        random.Random(a.seed * 1000 + ep).shuffle(order)
        losses, preds, ys = [], [], []
        for k in range(0, len(order), a.batch):
            b = order[k:k + a.batch]
            if len(b) < 2:
                continue
            x = batch_tensor(b, True)
            y = torch.tensor([scores[i] for i in b], dtype=torch.float32, device=dev)
            with torch.autocast(dev, enabled=dev == 'cuda'):
                p = model(x)
            if a.task == 'reg':
                loss = torch.nn.functional.mse_loss(p.float(), y)
            else:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(p.float() - 3.0, (y >= 4).float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(float(loss.detach()) * len(b))
            preds.append(as_score(p.detach().float()).cpu().numpy())
            ys.append(y.cpu().numpy())
        t_train = time.time() - te0
        ptr = np.concatenate(preds)
        mtr = metrics(np.concatenate(ys), ptr)
        pte = predict(te)
        mte = metrics(yte, pte)
        rec = {'epoch': ep, 'lr': opt.param_groups[0]['lr'], 'train_loss': sum(losses) / len(ptr), 'train': mtr, 'test': mte,
               'grad_norm_last': float(gn), 'seconds': round(time.time() - te0, 1)}
        epf.write(json.dumps(rec) + '\n')
        epf.flush()
        mark = ''
        if (mte['mse'] < best['mse']) if a.task == 'reg' else (mte.get('auc_good', 0) > best['auc']):
            best = {'mse': mte['mse'], 'auc': mte.get('auc_good', 0), 'epoch': ep, 'test': mte}
            torch.save({'model': model.state_dict(), 'channels': channels, 'dstat': dstat, 'n_in': n_in, 'epoch': ep,
                        'size': (W, H), 'backbone': a.backbone}, os.path.join(a.out, 'best.pt'))
            mark = '  * best'
        ps = ' '.join('%d:%.2f' % (s, v['mean_pred']) for s, v in mte['per_score'].items())
        k4 = mte['keep4.0']
        log('ep %3d/%d lr %.2e | train loss %.3f MAE %.3f rho %.3f | test MSE %.3f RMSE %.3f MAE %.3f rho %.3f r %.3f '
            'AUC(4-5 vs 1-3) %.3f round-exact %.2f within1 %.2f | mean pred by score %s | keep>=4: %d kept, %s good, %d bad | %.1f s (train %.1f)%s'
            % (ep, a.epochs, rec['lr'], rec['train_loss'], mtr['mae'], mtr['spearman'], mte['mse'], mte['rmse'], mte['mae'],
               mte['spearman'], mte['pearson'], mte.get('auc_good', float('nan')), mte['exact_round'], mte['within_1'], ps,
               k4['kept'], '%.2f' % k4['precision_good'] if k4['precision_good'] is not None else '-', k4['bad_kept'],
               rec['seconds'], t_train, mark))
    torch.save({'model': model.state_dict(), 'channels': channels, 'dstat': dstat, 'n_in': n_in, 'epoch': a.epochs,
                'size': (W, H), 'backbone': a.backbone}, os.path.join(a.out, 'last.pt'))
    last = metrics(yte, pte)
    # speed of the final model, one frame and a batch
    model.eval()
    x = batch_tensor(te[:1], False)
    with torch.no_grad(), torch.autocast(dev, enabled=dev == 'cuda'):
        for _ in range(5):
            model(x)
        if dev == 'cuda':
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(50):
            model(x)
        if dev == 'cuda':
            torch.cuda.synchronize()
    ms = (time.time() - t) / 50 * 1000
    json.dump({'channels': channels, 'train': len(tr), 'test': len(te), 'baseline_mse': float(np.mean((yte - base_pred) ** 2)),
               'best_epoch': best['epoch'], 'best': best['test'], 'last': last, 'ms_per_frame': ms,
               'test_predictions': {i: [scores[i], round(float(p), 3)] for i, p in zip(te, pte)}},
              open(os.path.join(a.out, 'report.json'), 'w'), indent=1)
    log('done. last epoch: test MSE %.3f MAE %.3f rho %.3f; best epoch %d MSE %.3f (picked on test, optimistic); '
        'baseline MSE %.3f; %.1f ms a frame on %s' % (last['mse'], last['mae'], last['spearman'], best['epoch'], best['mse'],
                                                      float(np.mean((yte - base_pred) ** 2)), ms, dev))


if __name__ == '__main__':
    main()
