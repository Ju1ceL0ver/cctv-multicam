"""Which ReID model separates these particular people best.

The question is not which model wins on Market-1501 but which one, on our own
labelled clip, keeps the same person together and different people apart -- across
the two cameras and across time, which is where the sprint goal lives."""
import os, sys, json, glob, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, ROOT)
from clipdata import load

CLIP = sys.argv[1] if len(sys.argv) > 1 else 'c103700'
D = os.path.join(ROOT, 'data', 'raw_clips', CLIP)


def gt_ids():
    g = json.load(open(os.path.join(D, 'gt_identity_yolo26x-seg.json'), encoding='utf-8'))
    return g


def pairs(ids, cams, ts, cross_only, min_gap):
    same, diff = [], []
    n = len(ids)
    rng = np.random.default_rng(0)
    idx = np.arange(n)
    for i in idx:
        j = idx[i + 1:]
        if cross_only:
            j = j[cams[j] != cams[i]]
        j = j[np.abs(ts[j] - ts[i]) >= min_gap] if min_gap else j
        if len(j) > 400:
            j = rng.choice(j, 400, replace=False)
        s = ids[j] == ids[i]
        same.append(j[s]); diff.append(j[~s])
    return same, diff


def score(E, ids, cams, ts, cross_only, min_gap):
    E = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-6)
    same_d, diff_d = [], []
    rng = np.random.default_rng(0)
    n = len(ids)
    for i in range(n):
        j = np.arange(i + 1, n)
        if cross_only:
            j = j[cams[j] != cams[i]]
        if min_gap:
            j = j[np.abs(ts[j] - ts[i]) >= min_gap]
        if len(j) == 0:
            continue
        if len(j) > 300:
            j = rng.choice(j, 300, replace=False)
        d = 1.0 - E[j] @ E[i]
        s = ids[j] == ids[i]
        same_d.append(d[s]); diff_d.append(d[~s])
    same_d = np.concatenate(same_d) if same_d else np.zeros(0)
    diff_d = np.concatenate(diff_d) if diff_d else np.zeros(0)
    if len(same_d) < 20 or len(diff_d) < 20:
        return None
    thr = np.quantile(diff_d, 0.01)           # gate that wrongly joins 1 % of stranger pairs
    tpr = float((same_d <= thr).mean())        # share of true pairs that survive it
    lab = np.r_[np.ones(len(same_d)), np.zeros(len(diff_d))]
    sc = -np.r_[same_d, diff_d]
    o = np.argsort(sc)
    r = np.empty(len(sc)); r[o] = np.arange(len(sc))
    auc = float((r[lab == 1].sum() - len(same_d) * (len(same_d) - 1) / 2) / (len(same_d) * len(diff_d)))
    return dict(tpr_at_1pct=tpr, auc=auc, thr=float(thr), same=len(same_d), diff=len(diff_d))


def main():
    dets, feats, meta, _ = load(CLIP, apply_sync=True)
    g = gt_ids()
    ids, cams, ts, rows = [], [], [], {c: [] for c in ('cam1', 'cam2')}
    for c in ('cam1', 'cam2'):
        for k, v in sorted(g[c].items(), key=lambda kv: int(kv[0])):
            i = int(k)
            if v in (None, '', 'unknown') or i >= len(dets[c]):
                continue
            ids.append(str(v)); cams.append(0 if c == 'cam1' else 1); ts.append(float(dets[c][i, 0])); rows[c].append(i)
    ids = np.asarray(ids); cams = np.asarray(cams); ts = np.asarray(ts)
    print('labelled detections: %d, identities: %d' % (len(ids), len(set(ids))))
    out = {}
    for p in sorted(glob.glob(os.path.join(D, 'emb_*.npz'))):
        name = os.path.basename(p)[4:-4]
        if name.endswith('_masked'):
            continue
        z = np.load(p)
        try:
            E = np.concatenate([z['cam1'][rows['cam1']], z['cam2'][rows['cam2']]])
        except Exception as e:
            print(name, 'skipped', e); continue
        if E.shape[1] == 0:
            continue
        r_all = score(E, ids, cams, ts, False, 0.0)
        r_x = score(E, ids, cams, ts, True, 0.0)
        r_far = score(E, ids, cams, ts, False, 10.0)
        out[name] = dict(dim=int(E.shape[1]), all=r_all, cross=r_x, apart10s=r_far)
        f = lambda r: ('tpr@1%%far %.3f auc %.4f' % (r['tpr_at_1pct'], r['auc'])) if r else 'n/a'
        print('%-34s d=%4d  all: %s | cross-cam: %s | >10s apart: %s'
              % (name, E.shape[1], f(r_all), f(r_x), f(r_far)), flush=True)
    json.dump(out, open(os.path.join(ROOT, 'data', 'reid_compare.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
