"""How well does an appearance descriptor tell people apart, measured on ground truth.

AUC = P(a same-person pair is closer than a different-person pair). Detection level
(within camera >=3 s apart; across cameras) and piece level (mean descriptor of a
tracker piece vs another piece), which is the level association works at."""
import sys, os, json, numpy as np
from collections import Counter
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from clipdata import load

rng = np.random.RandomState(0)


def auc(pos, neg):
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    allv = np.concatenate([pos, neg]); ranks = allv.argsort().argsort()
    r = ranks[:len(pos)].sum()
    # distances: lower is "same" -> AUC of pos being smaller
    return 1 - (r - len(pos) * (len(pos) - 1) / 2) / (len(pos) * len(neg))


def lab_dist(a, b):
    f = [0, 1, 2, 6, 7, 8, 12, 13, 14]; w = np.array([0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1])
    return np.linalg.norm((a[..., f] - b[..., f]) * w, axis=-1) / 30.0


def cos_dist(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9); b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return 1 - (a * b).sum(-1)


def main(clip):
    dets, feats, meta, embs = load(clip)
    gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')))
    descs = {'lab': (feats, lab_dist)}
    for m in ('osnet_x0_25_msmt17', 'osnet_ain_x1_0_msmt17', 'osnet_ain_x1_0_msmt17_masked', 'osnet_x0_25_msmt17_masked'):
        p = os.path.join('data', 'raw_clips', clip, 'emb_%s.npz' % m)
        if os.path.exists(p):
            z = dict(np.load(p)); descs[m] = ({c: z[c] for c in ('cam1', 'cam2')}, cos_dist)
    L = {c: {int(k): v for k, v in gt[c].items()} for c in ('cam1', 'cam2')}
    idx = {c: np.array(sorted(L[c])) for c in L}
    lab = {c: np.array([L[c][i] for i in idx[c]]) for c in L}
    t = {c: dets[c][idx[c], 0] for c in L}
    # detection-level pairs
    pairs = {'within': ([], []), 'cross': ([], [])}
    for kind in pairs:
        pos, neg = [], []
        for _ in range(200000):
            if kind == 'within':
                c1 = c2 = ('cam1', 'cam2')[rng.randint(2)]
            else:
                c1, c2 = 'cam1', 'cam2'
            i, j = rng.randint(len(idx[c1])), rng.randint(len(idx[c2]))
            if kind == 'within' and abs(t[c1][i] - t[c2][j]) < 3:
                continue
            (pos if lab[c1][i] == lab[c2][j] else neg).append((c1, i, c2, j))
            if len(pos) > 4000 and len(neg) > 4000:
                break
        pairs[kind] = (pos[:4000], neg[:4000])
    print('%s: labelled dets cam1 %d cam2 %d, people %s' % (clip, len(idx['cam1']), len(idx['cam2']),
          dict(Counter(list(lab['cam1']) + list(lab['cam2'])).most_common(6))))
    # piece level
    pieces_path = os.path.join('data', 'raw_clips', clip, 'pieces_yolo26x-seg.json')
    pieces = json.load(open(pieces_path)) if os.path.exists(pieces_path) else []
    for name, (D, fn) in descs.items():
        res = []
        for kind, (pos, neg) in pairs.items():
            dp = [fn(D[c1][idx[c1][i]], D[c2][idx[c2][j]]) for c1, i, c2, j in pos]
            dn = [fn(D[c1][idx[c1][i]], D[c2][idx[c2][j]]) for c1, i, c2, j in neg]
            res.append('%s AUC %.3f (same med %.2f, diff med %.2f)' % (kind, auc(dp, dn), np.median(dp), np.median(dn)))
        pr = []
        for p in pieces:
            c = p['cam']; ds = [d for d in p['dets'] if d in L[c]]
            if len(ds) < 10:
                continue
            g = Counter(L[c][d] for d in ds).most_common(1)[0][0]
            v = D[c][ds]
            if name != 'lab':
                v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
            pr.append((c, g, v.mean(0)))
        for kind in ('within', 'cross'):
            pos, neg = [], []
            for a in range(len(pr)):
                for b in range(a + 1, len(pr)):
                    if (pr[a][0] == pr[b][0]) != (kind == 'within'):
                        continue
                    d = fn(pr[a][2], pr[b][2])
                    (pos if pr[a][1] == pr[b][1] else neg).append(d)
            res.append('piece-%s AUC %.3f (n %d/%d)' % (kind, auc(pos, neg), len(pos), len(neg)))
        print('  %-24s %s' % (name, ' | '.join(res)))


if __name__ == '__main__':
    for clip in sys.argv[1:]:
        main(clip)
