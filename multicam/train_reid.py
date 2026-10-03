"""Teach OSNet-AIN this shop from the harvested track pictures (reid_harvest.py).

Positives: pictures of one single-person piece of a track. Negatives: only pieces the same
camera saw at the same moment -- those are certainly different people. Two pieces apart in
time may well be one person broken by the tracker, so they are never pushed apart: a batch
is built from groups of simultaneous pieces and the triplet loss looks only inside a group.

To keep what the model knew before (MSMT17: thousands of people), every picture is also
pulled towards what the untouched model says about it. Without that anchor a few thousand
pseudo-labelled pieces could drag the embedding somewhere that only fits these two days.

usage: train_reid.py OUT.pt DAY [DAY ...]    (env RA_ITERS, RA_LR)"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'data' / 'weights' / 'osnet_ain_x1_0_msmt17.pt'
OVERLAP = 0.5        # s both pieces must be on screen together to count as two people
K = 4                # pictures per piece in a batch
GROUP = 4            # pieces per group at most
GROUPS = 6           # groups per batch
MARGIN = 0.3
ANCHOR = 1.0         # weight of staying close to the untouched model


def load_people(days, root=ROOT):
    people = []
    for day in days:
        folder = Path(root) / 'data/reid_harvest' / day
        m = json.load(open(folder / 'manifest.json', encoding='utf-8'))
        for p in m['people']:
            files = sorted((folder / p['folder']).glob('*.jpg'))
            if len(files) >= 2:
                people.append(dict(p, day=day, files=[str(f) for f in files]))
    return people


def rivals(people):
    """For every piece, the pieces the same camera showed at the same time."""
    out = [[] for _ in people]
    by = {}
    for i, p in enumerate(people):
        by.setdefault((p['day'], p['cam']), []).append(i)
    for idx in by.values():
        idx.sort(key=lambda i: people[i]['first'])
        for a_pos, a in enumerate(idx):
            for b in idx[a_pos + 1:]:
                if people[b]['first'] > people[a]['last'] - OVERLAP:
                    break
                if min(people[a]['last'], people[b]['last']) - people[b]['first'] >= OVERLAP:
                    out[a].append(b); out[b].append(a)
    return out


def batches(people, riv, rng):
    usable = [i for i, r in enumerate(riv) if r]
    while True:
        batch = []
        for _ in range(GROUPS):
            a = rng.choice(usable)
            others = rng.sample(riv[a], min(GROUP - 1, len(riv[a])))
            # members of one group must all be pairwise simultaneous
            group = [a]
            for o in others:
                if all(o in riv[g] for g in group):
                    group.append(o)
            batch.append(group)
        yield batch


def augment(img, rng):
    import cv2
    if rng.random() < 0.5:
        img = img[:, ::-1]
    h, w = img.shape[:2]
    if rng.random() < 0.5:                          # small shift and zoom
        pad = 10
        big = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
        x, y = rng.randint(0, 2 * pad), rng.randint(0, 2 * pad)
        img = big[y:y + h, x:x + w]
    img = np.ascontiguousarray(img).astype(np.float32)
    if rng.random() < 0.5:                          # light and colour of the shop change over the day
        img = img * rng.uniform(0.8, 1.2) + rng.uniform(-15, 15)
    if rng.random() < 0.5:                          # something in front of the person
        eh, ew = int(h * rng.uniform(0.1, 0.35)), int(w * rng.uniform(0.2, 0.6))
        y, x = rng.randint(0, h - eh), rng.randint(0, w - ew)
        img[y:y + eh, x:x + ew] = rng.uniform(0, 255)
    return np.clip(img, 0, 255)


def tensor(images, mean, std, device):
    import torch
    x = np.stack([im[:, :, ::-1] for im in images]).astype(np.float32) / 255.0      # BGR -> RGB
    x = torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).to(device)
    return (x - mean) / std


def features(model, x):
    v = model.global_avgpool(model.forward_features(x)).flatten(1)
    if model.fc is not None:
        v = model.fc(v)
    return v


def build(weights, device):
    from boxmot.reid.core.reid import ReID
    net = ReID(Path(weights), device=device, half=False)
    backend = net.model
    return backend.model, backend.mean_array.float().to(device), backend.std_array.float().to(device)


def train(out, days, iters=None, lr=None, root=ROOT, log=print, seed=0):
    import cv2
    import torch
    import torch.nn.functional as F
    iters = iters or int(os.environ.get('RA_ITERS', '3000'))
    lr = lr or float(os.environ.get('RA_LR', '3.5e-5'))
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    rng = random.Random(seed)
    people = load_people(days, root)
    riv = rivals(people)
    log('days %s: %d pieces, %d pictures, %d pieces with a simultaneous rival' % (
        days, len(people), sum(len(p['files']) for p in people), sum(1 for r in riv if r)))
    model, mean, std = build(BASE, device)
    frozen, _, _ = build(BASE, device)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    cache = {}

    def image(path):
        if path not in cache:
            if len(cache) > 60000:
                cache.pop(next(iter(cache)))
            im = cv2.imread(path)
            cache[path] = im if im.shape[:2] == (256, 128) else cv2.resize(im, (128, 256), interpolation=cv2.INTER_AREA)
        return cache[path]

    t0 = time.time()
    stream = batches(people, riv, rng)
    for it in range(1, iters + 1):
        groups = next(stream)
        imgs, labels, gids = [], [], []
        for g, group in enumerate(groups):
            for pid in group:
                files = people[pid]['files']
                for f in rng.sample(files, min(K, len(files))):
                    imgs.append(augment(image(f), rng)); labels.append(pid); gids.append(g)
        x = tensor(imgs, mean, std, device)
        labels = torch.tensor(labels, device=device); gids = torch.tensor(gids, device=device)
        with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
            v = features(model, x)
            with torch.no_grad():
                v0 = features(frozen, x)
        v = F.normalize(v.float(), dim=1); v0 = F.normalize(v0.float(), dim=1)
        d = 1 - v @ v.T
        same = labels[:, None] == labels[None, :]
        together = gids[:, None] == gids[None, :]
        pos = torch.where(same, d, torch.full_like(d, -1)).max(1).values
        neg = torch.where(together & ~same, d, torch.full_like(d, 9)).min(1).values
        has = (neg < 9) & (pos >= 0)
        trip = F.relu(pos - neg + MARGIN)[has].mean() if has.any() else d.sum() * 0
        anchor = (1 - (v * v0).sum(1)).mean()
        loss = trip + ANCHOR * anchor
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if it % 100 == 0 or it == 1:
            log('iter %d/%d  triplet %.4f  anchor %.4f  active %.2f  %.0f s' % (
                it, iters, trip.item(), anchor.item(), (F.relu(pos - neg + MARGIN)[has] > 0).float().mean().item() if has.any() else 0,
                time.time() - t0))
    model.eval()
    # model_name and num_classes let boxmot rebuild the same network whatever the file is called
    torch.save({'state_dict': model.state_dict(), 'model_name': 'osnet_ain_x1_0',
                'num_classes': int(model.classifier.out_features), 'days': days, 'iters': iters, 'lr': lr}, out)
    log('saved %s (%.0f s)' % (out, time.time() - t0))
    return out


if __name__ == '__main__':
    os.chdir(ROOT)
    train(sys.argv[1], sys.argv[2:], log=lambda m: print(time.strftime('%H:%M:%S'), m, flush=True))
