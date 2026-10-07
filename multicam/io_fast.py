"""Inside / outside / doorway for camera 1 (the door camera), a fast convolutional net (07.10.2026).

Input: the classifier's crop (inout.crop_box) at 96x160 with 7 channels -- RGB, the person's mask, the traced shop
floor (rooms.mask), and where each crop pixel lies in the whole frame (x, y in 0..1) -- plus the 10 geometry numbers
(inout_train.geometry). MobileNetV3-small, ImageNet weights, the stem widened to 7 channels (the extra ones start at
the mean of RGB / zero). Classes as inout_train: outside (2, 0), inside (1), doorway (3).

Every number is leave-one-day-out on camera 1's answers; next to it the boosting on the geometry alone, the same rows.
Then one net on every answer -> data/inout/io_fast.pt (+ .onnx) and its speed.

usage: io_fast.py [EPOCHS]  -> data/inout/io_fast_report.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CW, CH = 96, 160
CHANS = 7
FW, FH = 320, 180           # the whole frame, small, same 7 channels (07.10: compared with the crop and both)


def _resize(a, size):
    """cv2.resize of any channel count (OpenCV 5 takes at most 4 at once)."""
    import cv2
    return np.dstack([cv2.resize(np.ascontiguousarray(a[:, :, i:i + 4]), size, interpolation=cv2.INTER_AREA).reshape(size[1], size[0], -1)
                      for i in range(0, a.shape[2], 4)])


def dataset():
    import cv2
    import inout
    import inout_train as IT
    import rate
    import rooms
    S, L = inout.samples(str(ROOT)), inout.labels(str(ROOT))
    frames = rate.index(str(ROOT), force=True)
    floor = cv2.resize(rooms.mask('cam1'), (inout.W, inout.H), interpolation=cv2.INTER_NEAREST) > 0
    gx, gy = np.meshgrid(np.linspace(0, 255, inout.W), np.linspace(0, 255, inout.H))
    X, F, G, y, day, door, ids = [], [], [], [], [], [], []
    for sid, a in sorted(L.items()):
        s = S.get(sid)
        k = a['label'] if isinstance(a, dict) else a
        if s is None or s['cam'] != 'cam1' or k not in IT.CLASS or s['frame'] not in frames:
            continue
        img = cv2.imread(str(frames[s['frame']][0]))
        if img.shape[:2] != (inout.H, inout.W):
            img = cv2.resize(img, (inout.W, inout.H))
        lab = cv2.imread(str(frames[s['frame']][1]), cv2.IMREAD_UNCHANGED)
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        if lab.shape != (inout.H, inout.W):
            lab = cv2.resize(lab, (inout.W, inout.H), interpolation=cv2.INTER_NEAREST)
        full = np.dstack([img[:, :, ::-1], (lab == s['value']) * 255, floor * 255, gx, gy]).astype(np.float32)
        F.append(_resize(full, (FW, FH)).astype(np.uint8))
        X1, Y1, X2, Y2 = inout.crop_box(s['box'])
        sl = (slice(Y1, Y2), slice(X1, X2))
        stack = np.dstack([img[sl][:, :, ::-1], (lab[sl] == s['value']) * 255, floor[sl] * 255, gx[sl], gy[sl]]).astype(np.float32)
        X.append(_resize(stack, (CW, CH)).astype(np.uint8))
        G.append(IT.geometry(s)); y.append(IT.CLASS[k]); day.append(s['day']); door.append(bool(s.get('door'))); ids.append(sid)
    return np.stack(X), np.stack(F), np.array(G, np.float32), np.array(y), np.array(day), np.array(door), ids


def tiny(chans=CHANS, width=32):
    """07.10: three plain conv blocks (conv-BN-ReLU x2, max-pool), global pooling; ~0.1 M parameters."""
    import torch.nn as nn
    layers, c = [], chans
    for w in (width, width * 2, width * 4):
        layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                   nn.Conv2d(w, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d(2)]
        c = w
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
    return nn.Sequential(*layers)


def net(gmean, gstd, mode='crop'):
    import timm
    import torch
    import torch.nn as nn
    if mode.startswith('tiny_'):
        return tiny_net(gmean, gstd, mode[5:])
    body = timm.create_model('mobilenetv3_small_100', pretrained=True, num_classes=0)
    st = body.state_dict()
    b = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0, in_chans=CHANS)
    for k, w in st.items():
        if w.ndim == 4 and w.shape[1] == 3:
            extra = torch.zeros(w.shape[0], CHANS - 3, *w.shape[2:])
            extra[:, 0:1] = w.mean(1, keepdim=True)            # the person's mask starts like brightness
            st[k] = torch.cat([w, extra], 1)
            break
    b.load_state_dict(st)
    b2 = None
    if mode == 'both':
        b2 = timm.create_model('mobilenetv3_small_100', pretrained=False, num_classes=0, in_chans=CHANS)
        b2.load_state_dict(st)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode, self.body, self.fbody = mode, b, b2
            with torch.no_grad():
                dim = self.body.eval()(torch.zeros(1, CHANS, CH, CW)).shape[1] * (2 if mode == 'both' else 1)
            self.head = nn.Sequential(nn.Linear(dim + len(gmean), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 3))
            m = torch.tensor([0.485, 0.456, 0.406] + [0.5] * (CHANS - 3)) * 255
            s = torch.tensor([0.229, 0.224, 0.225] + [0.5] * (CHANS - 3)) * 255
            self.register_buffer('mean', m.view(1, CHANS, 1, 1)); self.register_buffer('std', s.view(1, CHANS, 1, 1))
            self.register_buffer('gmean', torch.tensor(gmean)); self.register_buffer('gstd', torch.tensor(gstd))

        def forward(self, x, f, g):                            # x: crop N x 7 x CH x CW, f: frame N x 7 x FH x FW, 0..255
            n = lambda t: (t - self.mean) / self.std
            if self.mode == 'crop':
                parts = [self.body(n(x))]
            elif self.mode == 'full':
                parts = [self.body(n(f))]
            else:
                parts = [self.body(n(x)), self.fbody(n(f))]
            return self.head(torch.cat(parts + [(g - self.gmean) / self.gstd], 1))
    return Net()


def lenet(chans=CHANS, grid=(5, 8)):
    """07.10, the owner's sketch: LeNet -- three 5x5 convs (8, 16, 32) each with 2x2 max-pool, the map flattened on a
    fixed grid (where in the frame matters, so no global average)."""
    import torch.nn as nn
    return nn.Sequential(nn.Conv2d(chans, 8, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(8, 16, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
                         nn.AdaptiveAvgPool2d(grid), nn.Flatten())


def tiny_net(gmean, gstd, mode):
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.mode = mode
            le = mode.startswith('le')
            core = mode[2:] if le else mode
            make = (lambda: lenet()) if le else (lambda: tiny())
            self.body = make() if core in ('crop', 'both') else None
            self.fbody = make() if core in ('full', 'both') else None
            one = 32 * 5 * 8 if le else 128
            dim = one * (2 if core == 'both' else 1)
            self.head = (nn.Sequential(nn.Linear(dim + len(gmean), 120), nn.ReLU(), nn.Dropout(0.3),
                                       nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 3)) if le else
                         nn.Sequential(nn.Linear(dim + len(gmean), 128), nn.ReLU(), nn.Dropout(0.3),
                                       nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3)))
            self.register_buffer('gmean', torch.tensor(gmean)); self.register_buffer('gstd', torch.tensor(gstd))

        def forward(self, x, f, g):
            parts = []
            if self.body is not None:
                parts.append(self.body(x / 127.5 - 1))
            if self.fbody is not None:
                parts.append(self.fbody(f / 127.5 - 1))
            return self.head(torch.cat(parts + [(g - self.gmean) / self.gstd], 1))
    return Net()


def train(X, F, G, y, epochs, dev, mode, seed=0):
    import torch
    import inout_train as IT
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = net(G.mean(0), G.std(0) + 1e-6, mode).to(dev)
    head = list(m.head.parameters())
    blr = 2e-3 if mode.startswith('tiny_') else 3e-4               # a net from scratch learns faster
    opt = torch.optim.AdamW([{'params': [p for k, p in m.named_parameters() if not k.startswith('head.')], 'lr': blr}, {'params': head, 'lr': 1e-3}], weight_decay=1e-3)
    bs = 32
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[blr, 1e-3], total_steps=epochs * int(np.ceil(len(y) / bs)), pct_start=0.2)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(IT.class_weights(y), dtype=torch.float32, device=dev), label_smoothing=0.05)
    Xt = torch.from_numpy(X.transpose(0, 3, 1, 2).copy()).to(dev)          # uint8 on the card, float per batch
    Ft = torch.from_numpy(F.transpose(0, 3, 1, 2).copy()).to(dev)
    Gt, yt = torch.from_numpy(G).to(dev), torch.from_numpy(y.astype(np.int64)).to(dev)
    m.train()
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(y))).to(dev)
        for k in range(0, len(y), bs):
            b = order[k:k + bs]
            n = len(b)
            gain = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)
            shift = torch.empty(n, 3, 1, 1, device=dev).uniform_(-12, 12)
            x, f = Xt[b].float(), Ft[b].float()
            x[:, :3] = (x[:, :3] * gain + shift).clamp(0, 255)
            f[:, :3] = (f[:, :3] * gain + shift).clamp(0, 255)
            loss = lossf(m(x, f, Gt[b] + torch.randn_like(Gt[b]) * 0.01), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


def predict(m, X, F, G):
    import torch
    dev = next(m.parameters()).device
    t = lambda A, k: torch.from_numpy(A[k:k + 256].transpose(0, 3, 1, 2).copy()).float().to(dev)
    with torch.no_grad():
        return np.concatenate([torch.softmax(m(t(X, k), t(F, k), torch.from_numpy(G[k:k + 256]).to(dev)), 1).cpu().numpy()
                               for k in range(0, len(X), 256)])


def report(name, P, y, door):
    pred = P.argmax(1)
    io = lambda m: int(((pred != y) & (y < 2) & (pred < 2) & m).sum())
    out = {'all': round(float((pred == y).mean()), 4), 'door': round(float((pred[door] == y[door]).mean()), 4),
           'door_wrong': int((pred[door] != y[door]).sum()), 'door_n': int(door.sum()), 'door_in_out': io(door),
           'in_out_all': io(np.ones(len(y), bool))}
    print('%-10s all %.3f | door %.3f (%d of %d wrong, inside<->outside %d)' % (
        name, out['all'], out['door'], out['door_wrong'], out['door_n'], out['door_in_out']), flush=True)
    return out


def main(epochs=25):
    import torch
    import inout_train as IT
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    t0 = time.time()
    X, F, G, y, day, door, ids = dataset()
    print('camera 1 answers', len(y), 'door', int(door.sum()), 'classes', np.bincount(y, minlength=3).tolist(),
          'days', sorted(set(day)), '%.0f s' % (time.time() - t0), flush=True)
    import os
    MODES = tuple(os.environ.get('RA_IO_MODES', 'crop,full,both').split(','))
    P = {k: np.zeros((len(y), 3)) for k in ('geom',) + MODES}
    for d in sorted(set(day)):
        tr, te = day != d, day == d
        if len(set(y[tr])) < 3:
            continue
        P['geom'][te] = IT.boost().fit(G[tr], y[tr]).predict_proba(G[te])
        for md in MODES:
            P[md][te] = predict(train(X[tr], F[tr], G[tr], y[tr], epochs, dev, md), X[te], F[te], G[te])
        print('day', d, int(te.sum()), 'done %.0f s' % (time.time() - t0), flush=True)
    for md in MODES:
        P[md + '+geom'] = (P['geom'] + P[md]) / 2
    rep = {k: report(k, v, y, door) for k, v in P.items()}
    best = max(MODES, key=lambda md: rep[md]['door'])
    rep['best'] = best
    np.savez(ROOT / 'data' / 'inout' / ('io_fast%s_oof.npz' % os.environ.get('RA_IO_TAG', '')), ids=np.array(ids), y=y, **{k.replace('+', '_'): v for k, v in P.items()})
    m = train(X, F, G, y, epochs, dev, best)
    out = ROOT / 'data' / 'inout'
    tag = os.environ.get('RA_IO_TAG', '')
    torch.save({'model': m.state_dict(), 'gmean': m.gmean.cpu().numpy(), 'gstd': m.gstd.cpu().numpy(), 'chans': CHANS,
                'size': [CH, CW], 'frame': [FH, FW], 'mode': best}, out / ('io_fast%s.pt' % tag))
    mc = m.cpu().eval()
    x1, f1, g1 = torch.zeros(16, CHANS, CH, CW), torch.zeros(16, CHANS, FH, FW), torch.zeros(16, G.shape[1])
    with torch.no_grad():
        for _ in range(3):
            mc(x1, f1, g1)
        t = time.time()
        for _ in range(20):
            mc(x1, f1, g1)
    rep['cpu_ms_per_16'] = round((time.time() - t) / 20 * 1000, 2)
    try:
        torch.onnx.export(mc, (x1, f1, g1), str(out / ('io_fast%s.onnx' % tag)), input_names=['x', 'f', 'g'], output_names=['logits'],
                          dynamic_axes={'x': {0: 'n'}, 'f': {0: 'n'}, 'g': {0: 'n'}}, opset_version=18)
    except Exception as e:
        rep['onnx_error'] = str(e)[:200]
    rep.update(n=len(y), epochs=epochs, minutes=round((time.time() - t0) / 60, 1))
    json.dump(rep, open(out / ('io_fast%s_report.json' % tag), 'w'), indent=1)
    print('best', best, 'cpu ms per 16 people', rep['cpu_ms_per_16'], 'saved', out / 'io_fast.pt', flush=True)


if __name__ == '__main__':
    main(*(int(a) for a in sys.argv[1:2]))
