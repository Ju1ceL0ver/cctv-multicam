"""How far SAM 3.1's masks move when the frame is given at another size (or chunk, or step): per frame, the
masks of run A matched to run B's by IoU (Hungarian), both at the full frame.

usage: sam31_cmp.py TAG_A TAG_B   -> prints and writes data/logs/sam31/cmp_A_B.json"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def per_frame(tag):
    import sam31_reid as R
    base = ROOT / 'data' / 'logs' / 'sam31' / tag
    info = json.load(open(base / 'info.json'))
    M = R.Masks(base / 'chunks.npz')
    owned, _ = R.link_seams(M, info['overlap'])
    out = {}
    for p, rs in owned.items():
        for r in rs:
            out.setdefault(int(M.rows[r, 1]) * info.get('step', 2), []).append(r)    # key: frame of the 25 fps recording
    return M, out, info


def iou(Ma, a, Mb, b):
    A, B = Ma.rows[a, 4:8], Mb.rows[b, 4:8]
    X1, Y1, X2, Y2 = max(A[0], B[0]), max(A[1], B[1]), min(A[2], B[2]), min(A[3], B[3])
    if X2 <= X1 or Y2 <= Y1:
        return 0.0
    ma, mb = Ma.crop(a), Mb.crop(b)
    sa = ma[int(Y1 - A[1]):int(Y2 - A[1]), int(X1 - A[0]):int(X2 - A[0])]
    sb = mb[int(Y1 - B[1]):int(Y2 - B[1]), int(X1 - B[0]):int(X2 - B[0])]
    inter = float((sa & sb).sum())
    return inter / max(1.0, ma.sum() + mb.sum() - inter)


def main():
    ta, tb = sys.argv[1], sys.argv[2]
    Ma, fa, ia = per_frame(ta)
    Mb, fb, ib = per_frame(tb)
    ious, only_a, only_b, frames = [], 0, 0, 0
    for f in sorted(set(fa) & set(fb)):
        ra, rb = fa[f], fb[f]
        S = np.array([[iou(Ma, a, Mb, b) for b in rb] for a in ra])
        r, c = linear_sum_assignment(-S)
        good = [(i, j) for i, j in zip(r, c) if S[i, j] >= 0.5]
        ious += [S[i, j] for i, j in good]
        only_a += len(ra) - len(good); only_b += len(rb) - len(good); frames += 1
    rep = {'a': ta, 'b': tb, 'frames': frames, 'matched': len(ious),
           'iou_median': round(float(np.median(ious)), 3) if ious else None,
           'iou_p10': round(float(np.percentile(ious, 10)), 3) if ious else None,
           'only_a': only_a, 'only_b': only_b,
           'speed_a': ia.get('s_per_frame'), 'speed_b': ib.get('s_per_frame'),
           'gpu_a': ia.get('gpu_peak_gb'), 'gpu_b': ib.get('gpu_peak_gb')}
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'sam31' / ('cmp_%s_%s.json' % (ta, tb)), 'w'), indent=1)
    print(rep)


if __name__ == '__main__':
    main()
