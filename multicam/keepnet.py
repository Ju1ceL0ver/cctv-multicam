"""The small network "keep / delete" for the people of SAM's drafts, trained on the owner's decisions.

Decisions: (1) /fix -- every person of a window draft (data/fix/<frame>_init.png) and what the owner did with it: his own mask
has (nearly) nothing there = delete, else keep; (2) /keep -- one person of SAM 3.1's single frames, keep / delete / unclear.
Rule applied first, not learned: a person under MIN_FULL px (of the 2176 x 1224 frame) is deleted.

Input: the person with room around it (RGB) + its mask (4th channel) at 128 x 128 (aspect kept), and numbers: camera, where
the foot is, size, shape. Output: the chance that the owner deletes it.

Test: his /fix test frames (23.09, never trained on) and the people of 23.09 in /keep. His criterion for the threshold: not one
person he keeps is thrown out -- so the threshold is the highest score of a kept person in the test (plus a hair), and the
report is how many of his deletions it still catches. A day-by-day cross-validation on the rest says how stable that is.

usage: keepnet.py [EPOCHS]    -> data/keep/keepnet.pt, data/keep/keepnet_report.json"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MIN_FULL = 300
S = 128
TEST_DAY = '20260923'
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


def person_crop(img, lab, value, box):
    """4 x S x S float tensor: RGB of the person with room around, its mask. img/lab at the same size."""
    H, W = lab.shape
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    mx, my = max(8, int(0.6 * w)), max(8, int(0.35 * h))
    X1, Y1, X2, Y2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
    c = img[Y1:Y2, X1:X2]
    m = (lab[Y1:Y2, X1:X2] == value).astype(np.uint8) * 255
    f = S / max(c.shape[:2])
    c = cv2.resize(c, (max(1, int(c.shape[1] * f)), max(1, int(c.shape[0] * f))), interpolation=cv2.INTER_AREA)
    m = cv2.resize(m, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((S, S, 4), np.uint8)
    out[:c.shape[0], :c.shape[1], :3] = c[:, :, ::-1]
    out[:c.shape[0], :c.shape[1], 3] = m
    return out


def numbers(cam, foot, area_full, box, W, H):
    x1, y1, x2, y2 = box
    return [1.0 if cam == 'cam2' else 0.0, foot[0] / W, foot[1] / H, np.log10(max(area_full, 1)) - 3.7,
            (x2 - x1) / max(1, (y2 - y1)), (x2 - x1) / W, (y2 - y1) / H]


DEL_G, IGN_G = (2, 4, 8, 9), (0, 3, 5, 6, 7)


def overrides():
    """The owner's second look at the people he kept and the network doubted: {original sample id: new label}."""
    R = ROOT / 'data' / 'keep'
    L = json.load(open(R / 'labels.json'))
    Sm = json.load(open(R / 'samples.json'))
    return {Sm[sid]['orig']: v['label'] for sid, v in L.items() if sid in Sm and Sm[sid].get('review')}


def load_fix(over=None):
    over = over or {}
    R = ROOT / 'data' / 'fix'
    st = json.load(open(R / 'state.json'))
    items = {x['id']: x for x in json.load(open(R / 'manifest.json'))['items']}
    rows = []
    for k, v in st.items():
        if v.get('verdict') != 'take' or k not in items:
            continue
        it = items[k]
        img = cv2.imread(str(R / ('%s.jpg' % k)))
        ini = cv2.imread(str(R / ('%s_init.png' % k)), cv2.IMREAD_UNCHANGED)
        mk = cv2.imread(str(R / ('%s_mask.png' % k)), cv2.IMREAD_UNCHANGED)
        if img is None or ini is None:
            continue
        ini = ini[..., 0] if ini.ndim == 3 else ini
        mk = ini if mk is None else (mk[..., 0] if mk.ndim == 3 else mk)      # taken as it was: no mask file, nothing deleted
        H, W = ini.shape
        for i in np.unique(ini):
            if not i:
                continue
            ys, xs = np.nonzero(ini == i)
            a = len(ys)
            if a < MIN_FULL:
                continue
            box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
            low = ys >= ys.max() - 4
            foot = [int(xs[low].mean()), int(ys.max())]
            gone = int((mk == i).sum()) < 0.2 * a
            pid = '%s_p%d' % (k, i)
            if pid in over:
                if over[pid] in IGN_G:
                    continue
                gone = over[pid] in DEL_G
            rows.append({'src': 'fix', 'id': '%s_p%d' % (k, i), 'day': (it.get('tag') or k)[:8], 'cam': it['cam'], 'y': 1 if gone else 0,
                         'test': it.get('set') == 'test', 'x': person_crop(img, ini, int(i), box), 'n': numbers(it['cam'], foot, a, box, W, H)})
    return rows


def load_keep(over=None):
    over = over or {}
    R = ROOT / 'data' / 'keep'
    L = json.load(open(R / 'labels.json'))
    Sm = json.load(open(R / 'samples.json'))
    import rate
    idx = rate.index(str(ROOT))
    rows, cache = [], {}
    for sid, v in L.items():
        if Sm.get(sid, {}).get('review'):         # the second look: it overrides the first answer of its original
            continue
        lbl = over.get(sid, v['label'])
        if lbl in IGN_G:                         # ignore group: not a keep / delete question
            continue
        s = Sm[sid]
        if s['area_full'] < MIN_FULL:
            continue
        fr = s['frame']
        if fr not in cache:
            cache.clear()
            img = cv2.imread(str(idx[fr][0]))
            lab = cv2.imread(str(ROOT / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % fr)), cv2.IMREAD_UNCHANGED)
            lab = lab[..., 0] if lab.ndim == 3 else lab
            img = cv2.resize(img, (1280, 720)) if img.shape[:2] != (720, 1280) else img
            lab = cv2.resize(lab, (1280, 720), interpolation=cv2.INTER_NEAREST) if lab.shape != (720, 1280) else lab
            cache[fr] = (img, lab)
        img, lab = cache[fr]
        rows.append({'src': 'keep', 'id': sid, 'day': s['day'], 'cam': s['cam'], 'y': 1 if lbl in DEL_G else 0, 'cause': lbl, 'test': s['day'] == TEST_DAY,
                     'x': person_crop(img, lab, s['value'], s['box']), 'n': numbers(s['cam'], s['foot'], s['area_full'], s['box'], 1280, 720)})
    return rows


def load_keep2(unanswered=False):
    """The re-labelling from scratch (/keep2): people of both kinds (SAM single frames and /fix window frames), his answers only."""
    R = ROOT / 'data' / 'keep2'
    L = json.load(open(R / 'labels.json'))
    Sm = json.load(open(R / 'samples.json'))
    import rate
    idx = rate.index(str(ROOT))
    F = ROOT / 'data' / 'fix'
    rows, cache = [], {}
    pairs = [(i, {'label': 1}) for i in Sm if i not in L] if unanswered else list(L.items())     # unanswered: labels are dummies, only for scoring
    for sid, v in sorted(pairs, key=lambda kv: Sm.get(kv[0], {}).get('frame', '')):
        s = Sm.get(sid)
        lbl = v['label']
        if s is None or lbl in IGN_G or s['area_full'] < MIN_FULL:
            continue
        fr = s['frame']
        if (s.get('src'), fr) not in cache:
            cache.clear()
            if s.get('src') == 'fix':
                img, lab = cv2.imread(str(F / ('%s.jpg' % fr))), cv2.imread(str(F / ('%s_init.png' % fr)), cv2.IMREAD_UNCHANGED)
            else:
                img, lab = cv2.imread(str(idx[fr][0])), cv2.imread(str(ROOT / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % fr)), cv2.IMREAD_UNCHANGED)
            lab = lab[..., 0] if lab.ndim == 3 else lab
            img = cv2.resize(img, (1280, 720)) if img.shape[:2] != (720, 1280) else img
            lab = cv2.resize(lab, (1280, 720), interpolation=cv2.INTER_NEAREST) if lab.shape != (720, 1280) else lab
            cache[(s.get('src'), fr)] = (img, lab)
        img, lab = cache[(s.get('src'), fr)]
        Hh, Ww = lab.shape
        rows.append({'src': 'fix' if s.get('src') == 'fix' else 'keep', 'id': sid, 'day': s['day'], 'cam': s['cam'], 'y': 1 if lbl in DEL_G else 0, 'cause': lbl,
                     'test': s['day'] == TEST_DAY, 'x': person_crop(img, lab, s['value'], s['box']), 'n': numbers(s['cam'], s['foot'], s['area_full'], s['box'], Ww, Hh)})
    return rows


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        import timm
        self.cnn = timm.create_model('mobilenetv3_small_100', pretrained=True, in_chans=4, num_classes=0)
        self.cnn.eval()
        with torch.no_grad():
            d = self.cnn(torch.zeros(1, 4, S, S)).shape[1]
        self.cnn.train()
        self.num = nn.Sequential(nn.Linear(7, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(d + 32, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 1))
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406, 0.0]).view(1, 4, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225, 1.0]).view(1, 4, 1, 1))

    def forward(self, x, n):
        x = (x - self.mean) / self.std
        return self.head(torch.cat([self.cnn(x), self.num(n)], 1)).squeeze(1)


def tensors(rows):
    x = torch.from_numpy(np.stack([r['x'] for r in rows])).permute(0, 3, 1, 2).float() / 255
    return x, torch.tensor([r['n'] for r in rows], dtype=torch.float32), torch.tensor([r['y'] for r in rows], dtype=torch.float32)


def augment(x):
    b = x.shape[0]
    g = 1 + 0.25 * (torch.rand(b, 1, 1, 1, device=x.device) - 0.5) * 2
    x = x.clone()
    x[:, :3] = (x[:, :3] * g).clamp(0, 1)
    return x


def fit(train, epochs=12, seed=0):
    torch.manual_seed(seed)
    net = Net().to(dev)
    x, n, y = tensors(train)
    pos = float(y.mean())
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / max(pos, 1e-3) ** 0.5, device=dev))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * ((len(train) + 63) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps)
    net.train()
    for e in range(epochs):
        perm = torch.randperm(len(train))
        for i in range(0, len(train), 64):
            ix = perm[i:i + 64]
            xb, nb, yb = x[ix].to(dev), n[ix].to(dev), y[ix].to(dev)
            loss = lossf(net(augment(xb), nb), yb)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def score(net, rows):
    net.eval()
    x, n, _ = tensors(rows)
    out = []
    for i in range(0, len(rows), 128):
        out.append(torch.sigmoid(net(x[i:i + 128].to(dev), n[i:i + 128].to(dev))).cpu())
    return torch.cat(out).numpy()


def zero_fp(p, y):
    """Threshold = just above the highest score of a person the owner KEEPS; what that catches of his deletions."""
    keep, dele = p[y == 0], p[y == 1]
    t = float(keep.max()) + 1e-6 if len(keep) else 0.5
    caught = int((dele >= t).sum())
    one = np.sort(keep)[-2] + 1e-6 if len(keep) > 1 else t
    return t, caught, len(dele), float(one), int((dele >= one).sum())


def auc(p, y):
    o = np.argsort(p); r = np.empty(len(p)); r[o] = np.arange(len(p))
    pos = y == 1
    return float((r[pos].sum() - pos.sum() * (pos.sum() - 1) / 2) / max(1, pos.sum() * (~pos).sum()))


if __name__ == '__main__':
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    mode = sys.argv[2] if len(sys.argv) > 2 else 'all'          # 'keep': train only on the fresh /keep answers (test stays the same)
    tag = '' if mode == 'all' else '_' + mode
    t0 = time.time()
    over = overrides()
    print('second looks: %d (new delete/ignore: %d)' % (len(over), sum(v != 1 for v in over.values())), flush=True)
    rows = load_keep2() if mode == 'keep2' else load_fix(over) + load_keep(over)
    print('people: %d (fix %d, keep %d), deleted %d (%.1f %%), %.0f s' % (len(rows), sum(r['src'] == 'fix' for r in rows), sum(r['src'] == 'keep' for r in rows),
                                                                       sum(r['y'] for r in rows), 100 * np.mean([r['y'] for r in rows]), time.time() - t0), flush=True)
    train = [r for r in rows if not r['test'] and (mode in ('all', 'keep2') or r['src'] == mode)]
    test = [r for r in rows if r['test']]
    yt = np.array([r['y'] for r in test])
    print('train %d (deleted %d), test %d (kept %d, deleted %d)' % (len(train), sum(r['y'] for r in train), len(test), (yt == 0).sum(), (yt == 1).sum()), flush=True)
    report = {'people': len(rows), 'train': len(train), 'test': len(test)}
    # day-by-day cross-validation on the rest: stability of the zero-false-removal threshold
    days = sorted({r['day'] for r in train})
    folds = [days[i::4] for i in range(4)]
    cv = []
    oof = {}
    for fold in folds:
        tr = [r for r in train if r['day'] not in fold]
        te = [r for r in train if r['day'] in fold]
        net = fit(tr, epochs)
        p = score(net, te); y = np.array([r['y'] for r in te])
        oof.update({r['id']: float(v) for r, v in zip(te, p)})
        t, c, nd, t1, c1 = zero_fp(p, y)
        cv.append({'days': fold, 'n': len(te), 'deleted': int(nd), 'auc': round(auc(p, y), 3), 'threshold_zero_fp': round(t, 3), 'caught': c, 'caught_if_one_kept_lost': c1})
        print('  CV days %s: n %d, deleted %d, AUC %.3f, zero-FP threshold %.3f catches %d of %d (%d if one kept person may be lost)' % (fold, len(te), nd, auc(p, y), t, c, nd, c1), flush=True)
    report['cv'] = cv
    net = fit(train, epochs)
    p = score(net, test)
    oof.update({r['id']: float(v) for r, v in zip(test, p)})
    t, c, nd, t1, c1 = zero_fp(p, yt)
    report['test'] = {'auc': round(auc(p, yt), 3), 'threshold_zero_fp': round(t, 4), 'caught': c, 'deleted': int(nd), 'threshold_one_lost': round(t1, 4), 'caught_one_lost': c1,
                      'by_source': {s: {'n': sum(r['src'] == s for r in test), 'deleted': int(sum(r['y'] for r in test if r['src'] == s))} for s in ('fix', 'keep')}}
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
        report['test']['at_%.1f' % thr] = {'removes_deleted': int(((p >= thr) & (yt == 1)).sum()), 'removes_kept': int(((p >= thr) & (yt == 0)).sum())}
    print('TEST (23.09): AUC %.3f; threshold %.4f (zero kept removed) catches %d of his %d deletions; allowing 1 lost kept (t=%.4f): %d' % (auc(p, yt), t, c, nd, t1, c1), flush=True)
    for thr in (0.1, 0.2, 0.3, 0.5, 0.7):
        print('   score >= %.1f: removes %d of his %d deletions and %d of %d kept' % (thr, ((p >= thr) & (yt == 1)).sum(), nd, ((p >= thr) & (yt == 0)).sum(), (yt == 0).sum()), flush=True)
    # the final network: on everything, threshold from the test (the highest kept score) -- stored with the weights
    final = fit([r for r in rows if mode in ('all', 'keep2') or r['src'] == mode], epochs)
    torch.save({'model': final.state_dict(), 'threshold': float(t), 'min_full': MIN_FULL, 'epochs': epochs}, ROOT / 'data' / ('keep2' if mode == 'keep2' else 'keep') / ('keepnet%s.pt' % tag))
    report['saved'] = 'data/keep/keepnet%s.pt' % tag
    json.dump({r['id']: {'score': round(oof[r['id']], 4), 'y': r['y'], 'src': r['src'], 'day': r['day'], 'cam': r['cam']} for r in rows if r['id'] in oof},
              open(ROOT / 'data' / 'keep' / ('keepnet_oof%s.json' % tag), 'w'))
    json.dump(report, open(ROOT / 'data' / 'keep' / ('keepnet_report%s.json' % tag), 'w'), indent=1)
    print('done %.0f s' % (time.time() - t0), flush=True)
