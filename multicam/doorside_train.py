"""Frames of the owner's /doorside TRAIN stretches for the inside/outside model, and a refit (07.10.2026).

collect DAY...: every 3rd tick (0.24 s) of every train stretch marked 'done', every SAM track with a known side at that
  moment (the last mark of its person at or before it; merged pieces are one person; not-a-person left out) -> the
  binary model's own features of that crop (BinaryDoorClassifier's input path) -> data/door_side/train_frames.npz
fit: the 1628 /inout examples (labels_2class.json) + these frames -> inout_lab/binary_door_v2.pkl, and the
  /doorside TEST stretches scored with both models (per frame, every 3rd tick, the same way)."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'inout_lab'))
OUT = ROOT / 'data' / 'door_side'
SOFT = 1.5                   # s around the owner's side change where the frame's side is a matter of taste


def grabber():
    from binary_door import BinaryDoorClassifier

    class Grab(BinaryDoorClassifier):
        def _predict_inputs(self, full, crop_mask, geometry, signed=None, crop_rgb=None, crop_floor=None):
            return self._features(full, crop_mask, geometry, signed)
    return Grab()


def frames(day, role):
    """[(features, inside 0/1, tag, person, t)] of the day's done stretches of that role."""
    import cv2
    import door_side as S
    import sam31_reid as R
    g = grabber()
    st = S.load(day)
    out = []
    for tag, s in st.items():
        if not s.get('done') or s.get('role', 'test') != role:
            continue
        s = S.migrate(tag, s)
        merge, nop = s.get('merge', {}), set(s.get('noperson', []))
        marks = {}
        for p, xs in s['sides'].items():
            marks.setdefault(S.root_of(merge, p), []).extend(xs)
        marks = {k: sorted(v) for k, v in marks.items()}
        base = S.DOOR / tag / 'cam1'
        info = json.load(open(base / 'info.json'))
        M = R.Masks(base / 'chunks.npz')
        owned, _ = R.link_seams(M, {int(a): int(sh) for a, e, sh in info['sessions']})
        by_k = {}
        for piece, rs in owned.items():
            for r in rs:
                k = int(M.rows[r, 1])
                if k % 3:
                    continue
                r_ = S.root_of(merge, S.eff(s.get('cuts', {}), piece, k / S.FPS))
                if r_ in nop or r_ not in marks:
                    continue
                by_k.setdefault(k, []).append((r, r_))
        cap = cv2.VideoCapture(str(base / 'video.mp4'))
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        for k in sorted(by_k):
            t = k / S.FPS
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = cap.read()
            if not ok:
                continue
            rgb = f[:, :, ::-1].copy()
            for r, p in by_k[k]:
                xs = [m for m in marks[p] if m[0] <= t + 1e-6] or marks[p][:1]   # before the first mark: its side
                ch = [(m[0], m[1]) for i_, m in enumerate(marks[p]) if i_ and m[1] != marks[p][i_ - 1][1]]
                near = min(ch, key=lambda c: abs(c[0] - t)) if ch else None
                dt = (t - near[0]) if near else 99.0
                hard = int(xs[-1][1] == 'in')
                if near and abs(dt) <= SOFT:
                    q = 1 / (1 + np.exp(-dt / (SOFT / 3)))           # 0 -> 1 across the change
                    pin = q if near[1] == 'in' else 1 - q
                else:
                    pin = float(hard)
                x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
                m = np.zeros((H, W), np.uint8)
                c = M.crop(r)
                m[y1:y1 + c.shape[0], x1:x1 + c.shape[1]] = c[:H - y1, :W - x1]
                try:
                    out.append((g.predict_rgb(rgb, m), hard, tag, p, round(t, 2), round(float(dt), 2), round(float(pin), 4)))
                except ValueError:
                    pass
        cap.release()
        print(day, role, tag, len(out), flush=True)
    return out


def collect(days=None):
    days = days or ['20260917']
    rows = [x for d in days for x in frames(d, 'train')]
    np.savez(OUT / 'train_frames.npz', X=np.stack([r[0] for r in rows]), y=np.array([r[1] for r in rows]),
             tag=np.array([r[2] for r in rows]), person=np.array([r[3] for r in rows]), t=np.array([r[4] for r in rows]),
             dt=np.array([r[5] for r in rows]), pin=np.array([r[6] for r in rows]))
    print('train frames', len(rows), 'inside share', round(float(np.mean([r[1] for r in rows])), 3), flush=True)


def fit(days=None):
    from sklearn.ensemble import ExtraTreesClassifier
    from binary_door import binary_features
    z = np.load(ROOT / 'data' / 'inout' / 'io_cam1.npz')
    F, X0, G = z['F'], z['X'], z['G']
    lab = json.load(open(ROOT / 'inout_lab' / 'labels_2class.json'))
    Xo = np.stack([binary_features(F[i], X0[i][:, :, 3], G[i, 1:]) for i in range(len(G))])
    yo = np.array([lab[str(i)] == 'inside' for i in range(len(Xo))]).astype(int)
    t = np.load(OUT / 'train_frames.npz')
    X, y = np.concatenate([Xo, t['X']]), np.concatenate([yo, t['y']])
    # the new frames come in runs of one person; weight them so all of them count like ~1/4 of the old set
    w = np.concatenate([np.ones(len(yo)), np.full(len(t['y']), 0.25 * len(yo) / max(1, len(t['y'])))])
    m = ExtraTreesClassifier(300, min_samples_leaf=2, max_features=1., random_state=0, n_jobs=8).fit(X, y, sample_weight=w)
    m.n_jobs = 1
    pickle.dump(m, open(ROOT / 'inout_lab' / 'binary_door_v2.pkl', 'wb'))
    old = pickle.load(open(ROOT / 'inout_lab' / 'binary_door.pkl', 'rb'))
    test = [x for d in days for x in frames(d, 'test')]
    if test:
        Xt, yt = np.stack([r[0] for r in test]), np.array([r[1] for r in test])
        for name, mm in (('old', old), ('v2', m)):
            p = mm.predict_proba(Xt)[:, 1]
            print(name, 'test frames', len(yt), 'accuracy %.4f' % float(((p >= .5) == yt).mean()),
                  'sure (>=0.9) share %.3f accuracy %.4f' % (float((np.maximum(p, 1 - p) >= .9).mean()),
                                                            float(((p >= .5) == yt)[np.maximum(p, 1 - p) >= .9].mean())), flush=True)


def fit_soft(days=None):
    """Soft targets within +-SOFT s of a change (each such frame twice: inside with weight p, outside with 1 - p);
    the test frames scored by old / v2 / v3, all frames and without the +-SOFT band."""
    from sklearn.ensemble import ExtraTreesClassifier
    from binary_door import binary_features
    z = np.load(ROOT / 'data' / 'inout' / 'io_cam1.npz')
    F, X0, G = z['F'], z['X'], z['G']
    lab = json.load(open(ROOT / 'inout_lab' / 'labels_2class.json'))
    Xo = np.stack([binary_features(F[i], X0[i][:, :, 3], G[i, 1:]) for i in range(len(G))])
    yo = np.array([lab[str(i)] == 'inside' for i in range(len(Xo))]).astype(int)
    t = np.load(OUT / 'train_frames.npz')
    Xn, pn = t['X'], t['pin']
    wn = 0.25 * len(yo) / max(1, len(pn))
    X = np.concatenate([Xo, Xn, Xn])
    y = np.concatenate([yo, np.ones(len(pn), int), np.zeros(len(pn), int)])
    w = np.concatenate([np.ones(len(yo)), wn * pn, wn * (1 - pn)])
    keep = w > 1e-6
    m = ExtraTreesClassifier(300, min_samples_leaf=2, max_features=1., random_state=0, n_jobs=8).fit(X[keep], y[keep], sample_weight=w[keep])
    m.n_jobs = 1
    pickle.dump(m, open(ROOT / 'inout_lab' / 'binary_door_v3.pkl', 'wb'))
    print('soft frames within +-%.1f s: %d of %d' % (SOFT, int(((pn > 0) & (pn < 1)).sum()), len(pn)), flush=True)
    test = [x for d in (days or ['20260918', '20260919']) for x in frames(d, 'test')]
    Xt, yt, dt = np.stack([r[0] for r in test]), np.array([r[1] for r in test]), np.array([r[5] for r in test])
    far = np.abs(dt) > SOFT
    for name in ('binary_door', 'binary_door_v2', 'binary_door_v3'):
        mm = pickle.load(open(ROOT / 'inout_lab' / (name + '.pkl'), 'rb'))
        p = mm.predict_proba(Xt)[:, 1]
        ok = (p >= .5) == yt
        sure = np.maximum(p, 1 - p) >= .9
        print('%-15s all %.4f | away from changes (%d) %.4f | sure %.3f acc %.4f | near changes (%d) %.4f' % (
            name, ok.mean(), far.sum(), ok[far].mean(), sure.mean(), ok[sure].mean(), (~far).sum(), ok[~far].mean() if (~far).any() else float('nan')), flush=True)


if __name__ == '__main__':
    {'collect': collect, 'fit': fit, 'fit_soft': fit_soft}[sys.argv[1]](sys.argv[2:] or None)
