"""A plain UNet++ (Zhou et al. 2018) for "person / not person", as a yardstick next to the slot model (01.10): a
ResNet-18 encoder (ImageNet), the nested decoder, BCE + Dice, nothing clever. The input has every channel the slot
model has (10): the frame, the empty hall of its 15-minute file, the hall a few minutes before (for the single frames,
the same empty hall: nothing more recent is kept), the depth of the empty hall; the first convolution keeps ImageNet's
weights for the frame and starts the other seven small. Trained on N of SAM 3.1's single-frame
labels (sam31_stills, never 18.09 or 23.09) at 960 x 544; scored on the owner's test (/fix) the same way as the slot
model: its person mask cut into connected blobs, a blob found when it covers one of his people at IoU >= 0.5 (one to
one, the stride-4 grid). Two people walking shoulder to shoulder become one blob -- that is counted, not forgiven.

usage: unetpp.py --out runs/unetpp_a [--n 5000] [--epochs 20] [--until HH:MM]"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
IW, IH = 960, 544
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
SKIP = ('20260918', '20260923')


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class UNetPP(nn.Module):
    def __init__(self, pretrained=True, cin=10):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        conv = nn.Conv2d(cin, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            nn.init.kaiming_normal_(conv.weight)
            conv.weight.mul_(0.1)
            conv.weight[:, :3] = r.conv1.weight
        self.stem = nn.Sequential(conv, r.bn1, r.relu)                  # stride 2, 64
        self.enc = nn.ModuleList([nn.Sequential(r.maxpool, r.layer1), r.layer2, r.layer3, r.layer4])   # 4, 8, 16, 32
        ch, dc = [64, 64, 128, 256, 512], [32, 48, 64, 128, 256]
        self.nodes = nn.ModuleDict()
        for j in range(1, 5):
            for i in range(0, 5 - j):
                below = ch[i + 1] if j == 1 else dc[i + 1]
                self.nodes['%d_%d' % (i, j)] = block(ch[i] + (j - 1) * dc[i] + below, dc[i])
        self.head = nn.Conv2d(dc[0], 1, 1)

    def forward(self, x):
        f = [self.stem(x)]
        for e in self.enc:
            f.append(e(f[-1]))
        X = {(i, 0): f[i] for i in range(5)}
        for j in range(1, 5):
            for i in range(0, 5 - j):
                up = F.interpolate(X[(i + 1, j - 1)], size=X[(i, 0)].shape[-2:], mode='bilinear', align_corners=False)
                X[(i, j)] = self.nodes['%d_%d' % (i, j)](torch.cat([X[(i, k)] for k in range(j)] + [up], 1))
        return F.interpolate(self.head(X[(0, 4)]), size=x.shape[-2:], mode='bilinear', align_corners=False)


_DEPTH = {}


def depth(cam):
    if cam not in _DEPTH:
        d = np.load(ROOT / 'data' / 'scene' / ('depth_%s_da2_large.npy' % cam)).astype(np.float32)
        d = (d - d.min()) / max(1e-6, d.max() - d.min()) - 0.5
        _DEPTH[cam] = cv2.resize(d, (IW, IH), interpolation=cv2.INTER_LINEAR)
    return _DEPTH[cam]


def norm(bgr):
    return (bgr[:, :, ::-1].astype(np.float32) / 255.0 - MEAN) / STD


def stack(img, bg_long, bg_now, cam, jitter=None):
    """-> (10, IH, IW) float32: frame, empty hall, recent hall (ImageNet-normalised RGB each), depth."""
    rs = lambda a: cv2.resize(np.ascontiguousarray(a), (IW, IH), interpolation=cv2.INTER_AREA)
    f = rs(img)[:, :, ::-1].astype(np.float32) / 255.0
    if jitter:
        f = np.clip((f - 0.5) * jitter[0] + 0.5 + jitter[1], 0, 1)
    f = (f - MEAN) / STD
    return np.concatenate([f, norm(rs(bg_long)), norm(rs(bg_now)), depth(cam)[:, :, None]], 2).transpose(2, 0, 1)


class Data(torch.utils.data.Dataset):
    def __init__(self, items, train=True):
        self.items, self.train = items, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        img_p, lab_p, bg_p, cam = self.items[k]
        img = cv2.imread(img_p)
        bg = cv2.imread(bg_p)
        lab = cv2.imread(lab_p, cv2.IMREAD_UNCHANGED)
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        bg = cv2.resize(bg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
        m = (lab > 0).astype(np.uint8)
        x = stack(img, bg, bg, cam, (random.uniform(0.8, 1.2), random.uniform(-0.08, 0.08)) if self.train else None)
        m = cv2.resize(m, (IW, IH), interpolation=cv2.INTER_NEAREST)
        if self.train:                             # the same crop and flip for every channel (the camera stays put)
            s = random.uniform(0.75, 1.0)
            ch, cw = int(IH * s), int(IW * s)
            y, xx = random.randint(0, IH - ch), random.randint(0, IW - cw)
            x = np.stack([cv2.resize(c[y:y + ch, xx:xx + cw], (IW, IH), interpolation=cv2.INTER_LINEAR) for c in x])
            m = cv2.resize(m[y:y + ch, xx:xx + cw], (IW, IH), interpolation=cv2.INTER_NEAREST)
            if random.random() < 0.5:
                x, m = x[:, :, ::-1], m[:, ::-1]
        return torch.from_numpy(np.ascontiguousarray(x)), torch.from_numpy(np.ascontiguousarray(m)[None].astype(np.float32))


def items(n, seed=0):
    import rate
    d = ROOT / 'data' / 'sam31_stills' / 'drafts'
    bgs = ROOT / 'data' / 'seg_datasets' / 'backgrounds'
    out = []
    for i, (img, _) in rate.index(str(ROOT), force=True).items():
        parts = i.split('_')
        bg = bgs / ('%s_%s_%s.jpg' % (parts[0], parts[1], '_'.join(parts[2:4])))
        if not i.startswith(SKIP) and (d / ('%s.png' % i)).exists() and bg.exists():
            out.append((str(img), str(d / ('%s.png' % i)), str(bg), parts[1]))
    out.sort()
    random.Random(seed).shuffle(out)
    return out[:n]


def loss_fn(logit, m):
    bce = F.binary_cross_entropy_with_logits(logit, m)
    p = logit.sigmoid()
    dice = 1 - (2 * (p * m).sum((1, 2, 3)) + 1) / (p.sum((1, 2, 3)) + m.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def score(model, test, dev, thr=0.5):
    """The owner's test: blobs of the person mask as people (gold.match's way, on the stride-4 grid)."""
    import v2_data as VD
    from scipy.optimize import linear_sum_assignment
    model.eval()
    found = total = false = sf = st = 0
    ious, pix_i, pix_u = [], 0, 0
    for x in test.items:
        img = cv2.imdecode(x['jpg'], cv2.IMREAD_COLOR)
        a = torch.from_numpy(np.ascontiguousarray(stack(img, x['bg_long'], x['bg_long'], x['cam'])))[None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda'):
            p = model(a).float().sigmoid()[0, 0].cpu().numpy()
        g = cv2.resize(p, (VD.GW, VD.FH // 4), interpolation=cv2.INTER_AREA) >= thr
        nlab, comp = cv2.connectedComponents(g.astype(np.uint8), connectivity=8)
        P = []
        for v in range(1, nlab):
            b = comp == v
            if b.sum() >= 20:
                full = np.zeros((VD.GH, VD.GW), bool)
                full[:VD.FH // 4] = b
                P.append(full.reshape(-1))
        n = len(x['zone'])
        G = np.unpackbits(x['masks'], 1)[:, :VD.GH * VD.GW].astype(bool) if n else np.zeros((0, VD.GH * VD.GW), bool)
        U = G.any(0) if n else np.zeros(VD.GH * VD.GW, bool)
        Q = np.any(P, 0) if P else np.zeros(VD.GH * VD.GW, bool)
        pix_i += int((U & Q).sum()); pix_u += int((U | Q).sum())
        hit = {}
        if n and P:
            Gf, Pf = G.astype(np.float32), np.stack(P).astype(np.float32)
            inter = Gf @ Pf.T
            iou = inter / np.maximum(Gf.sum(1)[:, None] + Pf.sum(1)[None] - inter, 1)
            rr, cc = linear_sum_assignment(-iou)
            hit = {i: j for i, j in zip(rr, cc) if iou[i, j] >= 0.5}
            ious += [float(iou[i, j]) for i, j in hit.items()]
        total += n; found += len(hit); false += len(P) - len(hit)
        for i in range(n):
            if x['small'][i]:
                st += 1; sf += int(i in hit)
    R, Pr = found / max(1, total), found / max(1, found + false)
    model.train()
    return {'thr': thr, 'people': total, 'recall': round(R, 4), 'precision': round(Pr, 4), 'f1': round(2 * R * Pr / max(1e-9, R + Pr), 4),
            'false': false, 'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
            'recall_small': round(sf / max(1, st), 4), 'pixel_iou': round(pix_i / max(1, pix_u), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/unetpp_a')
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--until', default='')
    a = ap.parse_args()
    import v2_teacher_test as TT
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a), open(out / 'args.json', 'w'), indent=1)
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    model = UNetPP().to(dev).to(memory_format=torch.channels_last)
    data = Data(items(a.n))
    dl = torch.utils.data.DataLoader(data, batch_size=a.batch, shuffle=True, num_workers=4, drop_last=True, persistent_workers=True, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    total = a.epochs * len(dl)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=total, pct_start=0.05)
    test = TT.Test('fix')
    print('train', len(data), 'frames; test', len(test.items), 'frames', test.people, 'people', flush=True)
    best, step, t0 = -1.0, 0, time.time()
    for ep in range(1, a.epochs + 1):
        te, run = time.time(), []
        for x, m in dl:
            x, m = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last), m.to(dev, non_blocking=True)
            with torch.autocast('cuda', torch.bfloat16):
                logit = model(x)
            loss = loss_fn(logit.float(), m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step(); step += 1
            run.append(float(loss))
            if step % 50 == 0:
                json.dump({'step': step, 'steps': total, 'epoch': ep, 'loss': round(float(np.mean(run[-50:])), 4),
                           'updated': time.strftime('%H:%M:%S'), 'phase': 'training'}, open(out / 'status.json', 'w'))
        tests = [score(model, test, dev, t) for t in (0.4, 0.5, 0.6)]
        t = max(tests, key=lambda r: r['f1'])
        rec = {'epoch': ep, 'step': step, 'minutes': round((time.time() - te) / 60, 1), 'loss': round(float(np.mean(run)), 4),
               'time': time.strftime('%H:%M'), 'test': t, 'tests': tests}
        if t['f1'] > best:
            best = t['f1']; rec['best'] = True
            torch.save({'model': model.state_dict(), 'size': [IW, IH], 'thr': t['thr']}, out / 'best.pt')
        with open(out / 'epochs.jsonl', 'a') as f:
            f.write(json.dumps(rec) + '\n')
        print('EPOCH', json.dumps(rec), flush=True)
        if (out / 'STOP').exists() or (a.until and time.strftime('%H:%M') >= a.until):
            break
    torch.save({'model': model.state_dict(), 'size': [IW, IH]}, out / 'last.pt')
    print('done, best F1', best, 'in %.0f min' % ((time.time() - t0) / 60), flush=True)


if __name__ == '__main__':
    main()
