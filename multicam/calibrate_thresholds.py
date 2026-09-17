"""Appearance thresholds derived from ground truth, not guessed.

For every pair of tracker pieces with known identities, the appearance distance is
computed the way association computes it (nearest pair of ReID prototypes). The gate
is then set where the false-accept rate over DIFFERENT people is a chosen small value,
and the resulting true-accept rate over SAME-person pairs is reported, separately for
pairs on one camera and across cameras."""
import sys, os, json, numpy as np
from collections import Counter
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build, appearance, time_overlap
from clipdata import load

FPR = float(os.environ.get('RA_FPR', '0.02'))


def pieces_of(clip):
    dets, feats, meta, embs = load(clip)
    gt = json.load(open(os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')))
    cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}
    per_cam, _ = build(cams, dets, feats, embs, meta.get('clean'))
    out = []
    for it in per_cam['cam1'] + per_cam['cam2']:
        c = Counter(gt[it['cam']].get(str(int(i)), '?') for i in it['det']); c.pop('?', None)
        if not c:
            continue
        lab, n = c.most_common(1)[0]
        if n < 0.8 * sum(c.values()):
            continue
        out.append((lab, it))
    print('%s: %d labelled pieces, embeddings=%s' % (clip, len(out), meta.get('embeddings')))
    return out


def main(clips):
    same_w, diff_w, same_c, diff_c = [], [], [], []
    same_gap, diff_gap = [], []
    for clip in clips:
        pieces = pieces_of(clip)
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                (la, a), (lb, b) = pieces[i], pieces[j]
                d = appearance(a, b)
                cross = a['cam'] != b['cam']
                if la == lb:
                    (same_c if cross else same_w).append(d)
                else:
                    (diff_c if cross else diff_w).append(d)
                if not cross:
                    gap = max(b['t'][0] - a['t'][-1], a['t'][0] - b['t'][-1])
                    if gap > 0:
                        (same_gap if la == lb else diff_gap).append((gap, d))
    out = {}
    for name, same, diff in (('within', same_w, diff_w), ('cross', same_c, diff_c)):
        same, diff = np.array(same), np.array(diff)
        thr = float(np.quantile(diff, FPR))
        out[name] = {'threshold': round(thr, 3), 'kept_same': round(float((same <= thr).mean()), 3),
                     'same_median': round(float(np.median(same)), 3), 'diff_median': round(float(np.median(diff)), 3),
                     'same_p90': round(float(np.quantile(same, 0.9)), 3), 'n_same': int(len(same)), 'n_diff': int(len(diff))}
        print('%-7s same median %.2f p90 %.2f | diff median %.2f | gate at %d%% false accepts = %.2f -> keeps %.0f%% of same-person pairs'
              % (name, out[name]['same_median'], out[name]['same_p90'], out[name]['diff_median'], FPR * 100,
                 thr, 100 * out[name]['kept_same']))
    json.dump(dict(out, fpr=FPR, clips=list(clips)), open('data/appearance_thresholds.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1:] or ['c103700', 'c155236'])
