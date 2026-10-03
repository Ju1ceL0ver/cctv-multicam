"""Inside the shop, behind the glass or in the doorway, from the crop alone: train and measure on the
owner's /inout answers.

Classes: 0 outside (answers 2, and 0 = people in the other shops of the gallery), 1 inside (1),
2 in the doorway (3). Reflections (4) and "not a person" (5) are not learned. Classes are weighted by
their inverse frequency; the box geometry is standardised on the training days. Every number is
leave-one-day-out: the day scored was never seen in training.

Models:
  geom     gradient boosting on where the box is (camera, box, feet, distance to the traced floor)
  feat     gradient boosting on frozen MobileNetV3-small features of the crop + the same geometry
  cnn      MobileNetV3-small fine-tuned on the crop, the geometry joined before the last layers
  ctx      two MobileNetV3-small branches: the crop, and the whole frame small (192x108) with a 4th channel,
           the person's mask; both joined with the geometry
  frame    the frame-and-mask branch alone, with the geometry
  a+b...   the mean of those models' probabilities, every pair and triple and all five
Boosting runs on the CPU in a process of its own (RA_BOOST_THREADS cores) while the net trains on the card;
the net's 1024 features are squeezed to 64 by PCA for it, and it stops when the held-out part stops improving.

usage: inout_train.py [--pretrained 0]  -> data/inout/report.json, model_cnn.pt, model_cnn.onnx
RA_DEVICE=cuda|cpu (default: cuda when there is one), RA_THREADS for the CPU."""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CW, CH = 96, 160            # crop fed to the net: the person with room around it
FW, FH = 192, 108           # the whole frame, small, with the person's mask as a 4th channel
NETS = ('cnn', 'ctx', 'frame')
CLASS = {2: 0, 0: 0, 1: 1, 3: 2}     # owner's key -> class
NAMES = ['outside', 'inside', 'doorway']
BACKBONE = 'mobilenetv3_small_100'
_BODY = {}                  # the pretrained backbone weights, fetched once per run


def dataset(root):
    """Crops, geometry, classes, days, near flags, ids."""
    import cv2
    import inout
    import rate
    S, L = inout.samples(root), inout.labels(root)
    frames = rate.index(root, force=True)
    X, Fr, G, y, day, near, ids = [], [], [], [], [], [], []
    for sid, a in sorted(L.items()):
        s = S.get(sid)
        if s is None or a['label'] not in CLASS or s['frame'] not in frames:
            continue
        img = cv2.imread(str(frames[s['frame']][0]))
        if img.shape[:2] != (inout.H, inout.W):
            img = cv2.resize(img, (inout.W, inout.H))
        X1, Y1, X2, Y2 = inout.crop_box(s['box'])
        X.append(cv2.resize(img[Y1:Y2, X1:X2], (CW, CH), interpolation=cv2.INTER_AREA)[:, :, ::-1])
        lab = cv2.imread(str(frames[s['frame']][1]), cv2.IMREAD_UNCHANGED)
        if lab.ndim == 3:
            lab = lab[:, :, 0]
        mask = cv2.resize((lab == s['value']).astype(np.uint8) * 255, (FW, FH), interpolation=cv2.INTER_AREA)
        Fr.append(np.dstack([cv2.resize(img, (FW, FH), interpolation=cv2.INTER_AREA)[:, :, ::-1], mask]))
        G.append(geometry(s))
        y.append(CLASS[a['label']])
        day.append(s['day']); near.append(bool(s['near'])); ids.append(sid)
    return (np.stack(X), np.stack(Fr), np.array(G, np.float32), np.array(y), np.array(day), np.array(near), ids)


def geometry(s):
    x1, y1, x2, y2 = s['box']
    W, H = 1280.0, 720.0
    return [1.0 if s['cam'] == 'cam2' else 0.0, x1 / W, y1 / H, x2 / W, y2 / H, (x2 - x1) / W, (y2 - y1) / H,
            s['foot'][0] / W, s['foot'][1] / H, s['floor_px'] / 100.0]


def class_weights(y):
    n = np.bincount(y, minlength=len(NAMES)).astype(float)
    w = len(y) / (len(NAMES) * np.maximum(n, 1))
    return w / w.mean()


def scores(P, y, near):
    """3-class accuracy and confusion; inside-vs-outside accuracy on those two; per subset."""
    from sklearn.metrics import roc_auc_score
    out = {}
    pred = P.argmax(1)
    for name, m in (('all', np.ones(len(y), bool)), ('near', near), ('far', ~near)):
        yy, pp, PP = y[m], pred[m], P[m]
        conf = [[int(((yy == a) & (pp == b)).sum()) for b in range(len(NAMES))] for a in range(len(NAMES))]
        io = yy < 2
        io_pred = PP[io][:, 1] > PP[io][:, 0]                         # inside or outside, doorway set aside
        out[name] = {'n': int(m.sum()), 'accuracy3': round(float((yy == pp).mean()), 4) if len(yy) else None,
                     'confusion_rows_truth': conf,
                     'recall': {NAMES[a]: round(conf[a][a] / max(1, sum(conf[a])), 4) for a in range(len(NAMES))},
                     'in_vs_out_wrong': int((io_pred != (yy[io] == 1)).sum()), 'in_vs_out_n': int(io.sum()),
                     'in_vs_out_auc': round(float(roc_auc_score(yy[io] == 1, PP[io][:, 1] - PP[io][:, 0])), 4)
                     if len(set((yy[io] == 1).tolist())) == 2 else None}
    return out


def _stem_rgb_to_rgbm(state):
    """Pretrained 3-channel weights for a 4-channel input: the mask channel starts as the mean of RGB."""
    import torch
    out = dict(state)
    for k, w in state.items():
        if w.ndim == 4 and w.shape[1] == 3:
            out[k] = torch.cat([w, w.mean(1, keepdim=True)], 1)
            break
    return out


def net(pretrained, geo_dim, gmean=None, gstd=None, kind='cnn'):
    """kind: cnn (crop), ctx (crop + frame with mask), frame (frame with mask); the geometry is always joined."""
    import timm
    import torch
    import torch.nn as nn

    def body(chans):
        if pretrained and 'state' not in _BODY:
            _BODY['state'] = timm.create_model(BACKBONE, pretrained=True, num_classes=0).state_dict()
        b = timm.create_model(BACKBONE, pretrained=False, num_classes=0, in_chans=chans)
        if pretrained:
            b.load_state_dict(_BODY['state'] if chans == 3 else _stem_rgb_to_rgbm(_BODY['state']))
        return b

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.kind = kind
            dim = 0
            with torch.no_grad():                   # MobileNetV3 pools past num_features (576 -> 1024)
                if kind in ('cnn', 'ctx'):
                    self.body = body(3)
                    dim += self.body.eval()(torch.zeros(1, 3, CH, CW)).shape[1]
                if kind in ('ctx', 'frame'):
                    self.fbody = body(4)
                    dim += self.fbody.eval()(torch.zeros(1, 4, FH, FW)).shape[1]
            self.head = nn.Sequential(nn.Linear(dim + geo_dim, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, len(NAMES)))
            cfg = (self.body if hasattr(self, 'body') else self.fbody).pretrained_cfg
            mean = torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406))) * 255
            std = torch.tensor(cfg.get('std', (0.229, 0.224, 0.225))) * 255
            self.register_buffer('mean', mean.view(1, 3, 1, 1))
            self.register_buffer('std', std.view(1, 3, 1, 1))
            self.register_buffer('fmean', torch.cat([mean, torch.tensor([127.5])]).view(1, 4, 1, 1))
            self.register_buffer('fstd', torch.cat([std, torch.tensor([127.5])]).view(1, 4, 1, 1))
            self.register_buffer('gmean', torch.tensor(np.zeros(geo_dim) if gmean is None else gmean, dtype=torch.float32))
            self.register_buffer('gstd', torch.tensor(np.ones(geo_dim) if gstd is None else gstd, dtype=torch.float32))

        def features(self, x):                       # the crop branch: N x 3 x CH x CW, 0..255 RGB
            return self.body((x - self.mean) / self.std)

        def forward(self, x, f, g):                  # f: N x 4 x FH x FW, RGB + mask, 0..255
            parts = []
            if self.kind in ('cnn', 'ctx'):
                parts.append(self.features(x))
            if self.kind in ('ctx', 'frame'):
                parts.append(self.fbody((f - self.fmean) / self.fstd))
            return self.head(torch.cat(parts + [(g - self.gmean) / self.gstd], 1))

        def backbones(self):
            return [p for n, p in self.named_parameters() if not n.startswith('head.')]
    return Net()


def device():
    import torch
    want = os.environ.get('RA_DEVICE')
    return want or ('cuda' if torch.cuda.is_available() else 'cpu')


def to_tensor(X):
    import torch
    return torch.from_numpy(np.ascontiguousarray(X.transpose(0, 3, 1, 2))).float()


def train_net(kind, X, Fr, G, y, pretrained, epochs=20, seed=0, dev='cpu'):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    gmean, gstd = G.mean(0), G.std(0) + 1e-6
    m = net(pretrained, G.shape[1], gmean, gstd, kind).to(dev)
    opt = torch.optim.AdamW([{'params': m.backbones(), 'lr': 3e-4}, {'params': m.head.parameters(), 'lr': 1e-3}],
                            weight_decay=1e-3)
    bs = 32
    steps = epochs * int(np.ceil(len(y) / bs))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=steps, pct_start=0.2)
    lossf = torch.nn.CrossEntropyLoss(weight=torch.tensor(class_weights(y), dtype=torch.float32, device=dev),
                                      label_smoothing=0.05)
    Xt, Ft = to_tensor(X).to(dev), to_tensor(Fr).to(dev)
    Gt, yt = torch.from_numpy(G).to(dev), torch.from_numpy(y.astype(np.int64)).to(dev)
    m.train()
    for _ in range(epochs):
        order = torch.from_numpy(rng.permutation(len(y))).to(dev)
        for k in range(0, len(y), bs):
            b = order[k:k + bs]
            n = len(b)
            gain = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.8, 1.2)   # light and colour; no flips: the glass side matters
            shift = torch.empty(n, 3, 1, 1, device=dev).uniform_(-12, 12)
            x = (Xt[b] * gain + shift).clamp(0, 255)
            f = torch.cat([(Ft[b][:, :3] * gain + shift).clamp(0, 255), Ft[b][:, 3:]], 1)
            g = Gt[b] + torch.randn_like(Gt[b]) * 0.01                        # a box is never drawn to the pixel
            loss = lossf(m(x, f, g), yt[b])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return m.eval()


def predict_net(m, X, Fr, G):
    import torch
    dev = next(m.parameters()).device
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 256):
            out.append(torch.softmax(m(to_tensor(X[k:k + 256]).to(dev), to_tensor(Fr[k:k + 256]).to(dev),
                                       torch.from_numpy(G[k:k + 256]).to(dev)), 1).cpu().numpy())
    return np.concatenate(out)


def frozen_features(X, pretrained, dev):
    import torch
    m = net(pretrained, 0).to(dev).eval()
    with torch.no_grad():
        return np.concatenate([m.features(to_tensor(X[k:k + 256]).to(dev)).cpu().numpy() for k in range(0, len(X), 256)])


def boost():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight='balanced', early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=20, random_state=0)


PCA_DIM = 64


def fit_boost(A, y, B, pca_cols=0):
    """Probabilities for B. The first pca_cols columns (the net's features) are squeezed to PCA_DIM,
    the PCA fitted on the training rows only."""
    if pca_cols:
        from sklearn.decomposition import PCA
        p = PCA(PCA_DIM, random_state=0).fit(A[:, :pca_cols])
        A = np.hstack([p.transform(A[:, :pca_cols]), A[:, pca_cols:]])
        B = np.hstack([p.transform(B[:, :pca_cols]), B[:, pca_cols:]])
    m = boost().fit(A, y)
    return m.predict_proba(B), m.n_iter_


def boost_all(G, FG, y, day, pca_cols, threads):
    """Both boostings for every left-out day, in a process of their own with a fixed share of the cores."""
    from threadpoolctl import threadpool_limits
    t0 = time.time()
    out = {k: np.zeros((len(y), len(NAMES))) for k in ('geom', 'feat')}
    with threadpool_limits(limits=threads):
        for d in sorted(set(day)):
            tr, te = day != d, day == d
            for name, A, pc in (('geom', G, 0), ('feat', FG, pca_cols)):
                t1 = time.time()
                out[name][te], iters = fit_boost(A[tr], y[tr], A[te], pc)
                print('boost %s %s: %.1f s, %d iterations' % (d, name, time.time() - t1, iters), flush=True)
    print('boosting done, %.0f s' % (time.time() - t0), flush=True)
    return out


def combos(oof, bases):
    """The mean of every pair and triple of the base models, and of all of them."""
    from itertools import combinations
    out = {}
    for r in (2, 3, len(bases)):
        for c in combinations(bases, r):
            out['+'.join(c)] = sum(oof[b] for b in c) / len(c)
    return out


def speeds(final, X, Fr, G, out_dir):
    import torch
    speed = {}
    dev = next(final.parameters()).device
    for n in (1, 16):
        xb, fb, gb = to_tensor(X[:n]).to(dev), to_tensor(Fr[:n]).to(dev), torch.from_numpy(G[:n]).to(dev)
        with torch.no_grad():
            for _ in range(5):
                final(xb, fb, gb)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
            t = time.time()
            for _ in range(50):
                final(xb, fb, gb)
            if dev.type == 'cuda':
                torch.cuda.synchronize()
        speed['%s_torch_%d_ms' % (dev.type, n)] = round((time.time() - t) / 50 * 1000, 2)
    try:
        import onnxruntime as ort
        cpu = final.to('cpu')
        onnx_path = out_dir / ('model_%s.onnx' % final.kind)
        x16, f16, g16 = to_tensor(X[:16]), to_tensor(Fr[:16]), torch.from_numpy(G[:16])
        torch.onnx.export(cpu, (x16, f16, g16), str(onnx_path), input_names=['crop', 'frame', 'geom'], output_names=['logits'],
                          dynamic_axes={'crop': {0: 'n'}, 'frame': {0: 'n'}, 'geom': {0: 'n'}, 'logits': {0: 'n'}},
                          opset_version=17)
        so = ort.SessionOptions(); so.intra_op_num_threads = 4
        sess = ort.InferenceSession(str(onnx_path), so, providers=['CPUExecutionProvider'])
        for n in (1, 16):
            have = {'crop': x16[:n].numpy(), 'frame': f16[:n].numpy(), 'geom': g16[:n].numpy()}
            feed = {i.name: have[i.name] for i in sess.get_inputs()}     # an unused input is dropped from the graph
            for _ in range(3):
                sess.run(None, feed)
            t = time.time()
            for _ in range(50):
                sess.run(None, feed)
            speed['cpu_onnx_%d_ms' % n] = round((time.time() - t) / 50 * 1000, 2)
    except Exception as exc:
        speed['onnx_error'] = str(exc)[-300:]
    return speed


def main():
    import torch
    for stream in (sys.stdout, sys.stderr):           # torch.onnx prints emoji; a Windows console codepage cannot
        try:
            stream.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    torch.set_num_threads(int(os.environ.get('RA_THREADS', 8)))
    pretrained = '--pretrained' not in sys.argv or sys.argv[sys.argv.index('--pretrained') + 1] != '0'
    dev = device()
    root = str(ROOT)
    out_dir = Path(root) / 'data' / 'inout'
    t0 = time.time()
    X, Fr, G, y, day, near, ids = dataset(root)
    print('people %d: %s on %s; %.0f s' % (len(y), dict(zip(NAMES, np.bincount(y, minlength=3).tolist())), dev,
                                          time.time() - t0), flush=True)
    F = frozen_features(X, pretrained, dev)
    FG = np.hstack([F, G])
    oof = {k: np.zeros((len(y), len(NAMES))) for k in NETS}
    days = sorted(set(day))
    from concurrent.futures import ProcessPoolExecutor
    cpu = ProcessPoolExecutor(1).submit(boost_all, G, FG, y, day, F.shape[1],
                                        int(os.environ.get('RA_BOOST_THREADS', 16)))
    for kind in NETS:
        for d in days:
            tr, te = day != d, day == d
            oof[kind][te] = predict_net(train_net(kind, X[tr], Fr[tr], G[tr], y[tr], pretrained, dev=dev), X[te], Fr[te], G[te])
        print('net %s done, %.0f s' % (kind, time.time() - t0), flush=True)
    oof.update(cpu.result())
    bases = ['geom', 'feat'] + list(NETS)
    oof.update(combos(oof, bases))
    rule = np.zeros((len(y), len(NAMES))); rule[np.arange(len(y)), (G[:, -1] > 0).astype(int)] = 1
    report = {'people': len(y), 'classes': dict(zip(NAMES, np.bincount(y, minlength=3).tolist())),
              'days': sorted(set(day.tolist())), 'device': dev, 'floor_rule': scores(rule, y, near)}
    for k, P in oof.items():
        report[k] = scores(P, y, near)
    ranking = sorted(oof, key=lambda k: -report[k]['all']['accuracy3'])
    report['ranking'] = [(k, report[k]['all']['accuracy3']) for k in ranking]
    best = ranking[0]
    report['best'] = best
    wrong = np.nonzero(oof[best].argmax(1) != y)[0]
    report['wrong_best'] = [{'id': ids[i], 'truth': NAMES[y[i]], 'said': NAMES[int(oof[best][i].argmax())],
                             'p': [round(float(v), 3) for v in oof[best][i]]} for i in wrong]
    best_net = max(NETS, key=lambda k: report[k]['all']['accuracy3'])
    final = train_net(best_net, X, Fr, G, y, pretrained, dev=dev)
    torch.save(final.state_dict(), out_dir / ('model_%s.pt' % best_net))
    report['final_net'] = best_net
    report['speed'] = speeds(final, X, Fr, G, out_dir)
    report['params_m'] = round(sum(p.numel() for p in final.parameters()) / 1e6, 2)
    report['seconds'] = round(time.time() - t0)
    json.dump(report, open(out_dir / 'report.json', 'w'), indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ('ranking', 'best', 'final_net', 'speed', 'params_m', 'seconds')},
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
