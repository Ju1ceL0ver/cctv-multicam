"""Inside / outside / doorway, camera 1 -- a starting point for your own model.

Data: io_cam1.npz (next to this file), 1628 people answered by you on /inout, camera 1 only.
  X    N x 160 x 96 x 7  uint8   the crop around the person
  F    N x 180 x 320 x 7 uint8   the whole frame, small
       channels of both: 0-2 RGB, 3 the person's mask (0/255), 4 the shop floor (0/255),
       5 x and 6 y of each pixel in the whole frame (0..255)
  G    N x 10 float32            camera (0 here), box x1 y1 x2 y2, width, height, foot x y (all /1280, /720),
                                 signed distance of the feet to the floor's edge /100 px (+ on the floor)
  y    N                          0 outside (your 2 and 0), 1 inside (1), 2 doorway (3)
  day  N                          the day, for leave-one-day-out
  door N                          True = the door batch (the people that matter)
  ids  N                          the /inout sample id

The score: every day is predicted by a model trained on the other days (the same as on the machine).
Numbers to beat (machine, 07.10): geometry boosting -- door 0.877, all 0.927;
MobileNetV3 on the frame + boosting -- door 0.886, all 0.934.

usage: python3 train.py            # boosting baseline + the net below
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.ensemble import HistGradientBoostingClassifier

D = np.load(__file__.rsplit('/', 1)[0] + '/io_cam1.npz')
X, Fr, G, y, day, door = D['X'], D['F'], D['G'], D['y'], D['day'], D['door']
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'


# Resize once on CPU; all seven channels keep the same spatial alignment.
# 88 x 160 -> 44 x 80 -> 22 x 40 -> 11 x 20.
Fr = Fn.interpolate(torch.from_numpy(Fr.transpose(0, 3, 1, 2).copy()).float(),
                    size=(88, 160), mode='bilinear', align_corners=False).contiguous()
G = G[:, 1:]  # camera is constant: remove it entirely


class Net(nn.Module):
    """Three small conv/pool blocks, then three fully connected layers."""
    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(
            nn.Conv2d(7, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.fc1 = nn.Linear(256 * 2 * 2 + G.shape[1], 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 3)

    def forward(self, f, g):
        x = self.blocks(f / 127.5 - 1)
        # Resize to a divisible grid for adaptive pooling on MPS.
        x = Fn.interpolate(x, size=(10, 20), mode="bilinear", align_corners=False)
        x = Fn.adaptive_avg_pool2d(x, (2, 2)).flatten(1)
        x = torch.cat([x, g], 1)
        return self.fc3(Fn.relu(self.fc2(Fn.relu(self.fc1(x)))))


def augment(f, g, gm, gs, flip_prob=0.5):
    """Photometric changes affect RGB only; flip all spatial channels together."""
    f, g = f.clone(), g.clone()
    n = len(f)
    gain = torch.empty(n, 1, 1, 1, device=f.device).uniform_(0.85, 1.15)
    contrast = torch.empty(n, 1, 1, 1, device=f.device).uniform_(0.85, 1.15)
    rgb = f[:, :3]
    mean = rgb.mean((2, 3), keepdim=True)
    rgb = (rgb - mean) * contrast + mean
    rgb = rgb * gain + torch.randn_like(rgb) * 2.0
    f[:, :3] = rgb.clamp(0, 255)
    flip = torch.rand(n, device=f.device) < flip_prob
    f[flip] = f[flip].flip(-1)
    f[flip, 5] = 255 - f[flip, 5]  # horizontal coordinate channel
    raw = g * gs + gm
    # Nine numbers: x1,y1,x2,y2,width,height,foot_x,foot_y,floor_distance.
    left, right = raw[flip, 0].clone(), raw[flip, 2].clone()
    raw[flip, 0], raw[flip, 2] = 1 - right, 1 - left
    raw[flip, 6] = 1 - raw[flip, 6]
    return f, (raw - gm) / gs


def fit_net(tr, epochs=30, lr=1e-3, bs=64, flip_prob=0.5, balanced=True):
    torch.manual_seed(0)
    m = Net().to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
    w = torch.tensor(len(tr) / (3 * np.maximum(np.bincount(y[tr], minlength=3), 1)), dtype=torch.float32, device=DEV)
    gm, gs = G[tr].mean(0), np.maximum(G[tr].std(0), 0.01)
    # Keep the training tensors on MPS: no transfer for each minibatch.
    f = Fr[tr].to(DEV)
    g = torch.from_numpy((G[tr] - gm) / gs).float().to(DEV)
    t = torch.from_numpy(y[tr]).long().to(DEV)
    gm_t = torch.tensor(gm, device=DEV)
    gs_t = torch.tensor(gs, device=DEV)
    m.train()
    for epoch in range(epochs):
        losses, correct = [], 0
        for b in torch.randperm(len(tr)).split(bs):
            fb, gb = augment(f[b], g[b], gm_t, gs_t, flip_prob)
            logits = m(fb, gb)
            loss = Fn.cross_entropy(logits, t[b], weight=w if balanced else None)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            losses.append(loss.detach())
            correct += (logits.detach().argmax(1) == t[b]).sum()
        print(f'epoch {epoch+1:02d} loss={torch.stack(losses).mean().item():.4f} train_acc={correct.item()/len(tr):.3f}', flush=True)
    return m.eval(), gm, gs


def predict_net(model, te, bs=64):
    m, gm, gs = model
    result = []
    with torch.inference_mode():
        for indices in np.array_split(te, max(1, int(np.ceil(len(te) / bs)))):
            f = Fr[indices].to(DEV)
            g = torch.from_numpy((G[indices] - gm) / gs).float().to(DEV)
            result.append(torch.softmax(m(f, g), 1).cpu().numpy())
    return np.concatenate(result)


def boost():
    return HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                          class_weight='balanced', early_stopping=True, validation_fraction=0.15,
                                          n_iter_no_change=20, random_state=0)


def score(name, P):
    pred = P.argmax(1)
    io = int(((pred != y) & (y < 2) & (pred < 2) & door).sum())
    print('%-12s door %.3f (%d of %d wrong, inside<->outside %d) | all %.3f' % (
        name, (pred[door] == y[door]).mean(), (pred[door] != y[door]).sum(), door.sum(), io, (pred == y).mean()))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--day', help='Run only this held-out day')
    parser.add_argument("--mask-features", action="store_true", help="Use geometry plus mask/floor overlap features (run mask_baseline.py first)")
    parser.add_argument("--no-flip", action="store_true")
    parser.add_argument("--unweighted", action="store_true")
    args = parser.parse_args()
    if args.mask_features:
        from pathlib import Path
        G = np.load(Path(__file__).parent / "diagnostics" / "mask_features.npy").astype(np.float32)
    print(len(y), 'people,', door.sum(), 'at the door, classes', np.bincount(y).tolist(), 'device', DEV)
    Pb, Pn = np.zeros((len(y), 3)), np.zeros((len(y), 3))
    selected = np.unique(day) if args.day is None else [args.day]
    for d in selected:
        tr, te = np.where(day != d)[0], np.where(day == d)[0]
        Pb[te] = boost().fit(G[tr], y[tr]).predict_proba(G[te])
        Pn[te] = predict_net(fit_net(tr, epochs=args.epochs, flip_prob=0 if args.no_flip else 0.5, balanced=not args.unweighted), te)
        print('day', d, len(te))
    if args.day is not None:
        mask = day == args.day
        y, door, Pb, Pn = y[mask], door[mask], Pb[mask], Pn[mask]
    score('boosting', Pb)
    score('net', Pn)
    score('net+boost', (Pb + Pn) / 2)
