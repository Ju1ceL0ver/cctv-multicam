"""Per-camera tracking in image space (ByteTrack-style, with clothing colour).

Floor positions are too jumpy for frame-to-frame association inside one camera:
when the feet disappear behind a stand the floor estimate switches from foot to
head and can move half a metre in one frame. Box geometry does not jump, so
tracks are formed in pixels and only then placed on the floor."""
import numpy as np
from scipy.optimize import linear_sum_assignment

APP_IDX = [0, 1, 2, 6, 7, 8, 12, 13, 14]
APP_W = np.array([0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1])


def iou_matrix(A, B):
    x1 = np.maximum(A[:, None, 0], B[None, :, 0]); y1 = np.maximum(A[:, None, 1], B[None, :, 1])
    x2 = np.minimum(A[:, None, 2], B[None, :, 2]); y2 = np.minimum(A[:, None, 3], B[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (A[:, 2] - A[:, 0]) * (A[:, 3] - A[:, 1]); ab = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def _os_env(k, v):
    import os
    return os.environ.get(k, v)


def app_matrix(TA, DA):
    """Appearance distance between tracks and detections.

    With ReID embeddings (unit vectors) this is cosine distance scaled so that the
    thresholds keep their meaning (~1.0 = clearly different); the Lab-colour
    fallback keeps the old formula."""
    if TA.shape[1] > 32:
        return 1 - TA @ DA.T
    d = (TA[:, None, APP_IDX] - DA[None, :, APP_IDX]) * APP_W
    return np.linalg.norm(d, axis=2) / 30.0


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-9)


GATES = {'high': 1.2, 'low': 0.9, 'split': 1.1}     # overwritten from data/appearance_thresholds.json
DUP_IOU = float(_os_env('RA_DUPIOU', '0.85'))      # two live tracks on the same box are one person.
#   Measured on the four labelled clips: pieces 107 -> 73 and 94 -> 76, one-id on the
#   evening clip 0.866 -> 0.904, nothing else moved. At 0.7 two people standing close
#   start merging (c155236: 0.994 -> 0.912, wrong 0 % -> 8 %), so this stays strict.


def load_gates(path='data/appearance_thresholds.json'):
    import json, os
    if os.path.exists(path):
        t = json.load(open(path))
        if 'tracker' in t:
            GATES.update(t['tracker'])
    return GATES


def track_camera(d, feat, fps=25.0, high=0.5, max_age_s=None, min_len=8, emb=None):
    """d: (N, 10) t, x1, y1, x2, y2, score, fx, fy, hx, hy. Returns list of index arrays."""
    import os as _os
    if max_age_s is None:
        max_age_s = float(_os.environ.get('RA_MAXAGE', '1.5'))
    reacq_on = _os.environ.get('RA_REACQ', '0') == '1'   # measured: re-acquisition adds errors faster than it saves pieces
    if emb is not None and len(emb):
        feat = unit(emb.astype(np.float32))
    frames = np.round(d[:, 0] * fps).astype(int)
    order = np.argsort(frames, kind='stable')
    tracks = []            # dict(idx=[...], box, v, app, last)
    active = []
    for f in np.unique(frames):
        idx = order[frames[order] == f]
        idx = idx[(d[idx, 3] - d[idx, 1]) >= 12]              # degenerate boxes out
        boxes, scores = d[idx, 1:5], d[idx, 5]
        active = [a for a in active if f - tracks[a]['last'] <= max_age_s * fps]
        unmatched_d = list(range(len(idx)))
        unmatched_t = list(active)
        for stage in ('high', 'low'):
            cand = [j for j in unmatched_d if (scores[j] >= high) == (stage == 'high')]
            if not cand or not unmatched_t:
                continue
            P = np.array([tracks[a]['box'] + tracks[a]['v'] * (f - tracks[a]['last']) for a in unmatched_t])
            D = boxes[cand]
            iou = iou_matrix(P, D)
            app = app_matrix(np.array([tracks[a]['app'] for a in unmatched_t]), feat[idx[cand]])
            ph = P[:, 3] - P[:, 1]
            pc = np.c_[(P[:, 0] + P[:, 2]) / 2, (P[:, 1] + P[:, 3]) / 2]
            dc = np.c_[(D[:, 0] + D[:, 2]) / 2, (D[:, 1] + D[:, 3]) / 2]
            cdist = np.linalg.norm(pc[:, None] - dc[None], axis=2) / np.maximum(ph[:, None], 1)
            # crowding: when predictions of different tracks overlap, colour must decide
            crowd = (iou_matrix(P, P) > 0.15).sum(1) > 1
            w_app = np.where(crowd, 0.8, 0.35)[:, None]
            cost = (1 - iou) + w_app * app + 0.3 * cdist
            gates = load_gates() if emb is not None else {'high': 1.2, 'low': 0.9}
            g = gates['high'] if stage == 'high' else gates['low']
            # A shopper who walked behind a display stand comes back out somewhere
            # else: after a gap, appearance alone may re-acquire the track inside a
            # wide window, which is what keeps one person from becoming ten pieces.
            lost = np.array([f - tracks[a]['last'] for a in unmatched_t]) > 0.3 * fps
            reacquire = (lost[:, None] & (app < 0.7 * g) & (cdist < 2.0)) if reacq_on else np.zeros_like(iou, bool)
            ok = (((iou > 0.15) | (cdist < 0.35)) & (app < g)) | reacquire
            cost = np.where(ok, cost, 1e6)
            r, c = linear_sum_assignment(cost)
            done_t, done_d = set(), set()
            for i, j in zip(r, c):
                if cost[i, j] >= 1e6:
                    continue
                a = unmatched_t[i]; jj = cand[j]
                tr = tracks[a]
                gap = max(1, f - tr['last'])
                nb = boxes[jj]
                tr['v'] = 0.7 * tr['v'] + 0.3 * np.clip((nb - tr['box']) / gap, -40, 40)
                tr['box'] = nb; tr['last'] = f; tr['idx'].append(int(idx[jj]))
                tr['app'] = 0.9 * tr['app'] + 0.1 * feat[idx[jj]]
                if feat.shape[1] > 32:
                    tr['app'] = tr['app'] / max(np.linalg.norm(tr['app']), 1e-9)
                done_t.add(a); done_d.add(jj)
            unmatched_t = [a for a in unmatched_t if a not in done_t]
            unmatched_d = [j for j in unmatched_d if j not in done_d]
        for j in unmatched_d:
            if scores[j] < high:
                continue
            tracks.append({'idx': [int(idx[j])], 'box': boxes[j].copy(), 'v': np.zeros(4), 'app': feat[idx[j]].copy(), 'last': f})
            active.append(len(tracks) - 1)

        # Two tracks on one person take turns claiming their detections, frame after
        # frame -- measured on the labelled clips, that is where 98 % of the breaks come
        # from (one shopper alternated between two pieces 823 times). Whenever two live
        # tracks sit on the same box, the younger one is folded into the older.
        if DUP_IOU > 0 and len(active) > 1:
            B = np.array([tracks[a]['box'] for a in active])
            ov = iou_matrix(B, B)
            drop = set()
            for i in range(len(active)):
                for j in range(i + 1, len(active)):
                    if ov[i, j] <= DUP_IOU:
                        continue
                    a, b = active[i], active[j]
                    if a in drop or b in drop:
                        continue
                    keep, gone = (a, b) if len(tracks[a]['idx']) >= len(tracks[b]['idx']) else (b, a)
                    tracks[keep]['idx'].extend(tracks[gone]['idx'])
                    tracks[keep]['last'] = max(tracks[keep]['last'], tracks[gone]['last'])
                    tracks[gone]['idx'] = []
                    drop.add(gone)
            if drop:
                active = [a for a in active if a not in drop]
    out = []
    for tr in tracks:
        if len(tr['idx']) < min_len:
            continue
        k = np.array(sorted(set(tr['idx'])))
        out.append(k[np.argsort(d[k, 0], kind='stable')])
    return out


SPLIT_THR = 1.1


def split_on_appearance_change(d, feat, track, fps=25.0, win_s=1.0, thr=1.1, emb=None):
    """Cut a track where the clothing colour changes abruptly and stays changed.

    Two people swapping ids inside one track is the tracker's worst failure;
    their colour descriptors differ persistently, while pose/lighting changes of
    one person are gradual. Compare the median descriptor of the second before
    and after each candidate point."""
    if len(track) < 3 * int(win_s * fps):
        return [track]
    t = d[track, 0]
    if emb is not None and len(emb):
        F = unit(emb[track].astype(np.float32)) * 30.0            # /30 in the score restores cosine units
        thr = load_gates().get('split', 0.45)
    else:
        F = feat[track][:, APP_IDX] * APP_W
    w = int(win_s * fps)
    score = np.zeros(len(track))
    for i in range(w, len(track) - w):
        a = np.median(F[i - w:i], 0); b = np.median(F[i:i + w], 0)
        score[i] = np.linalg.norm(a - b) / 30.0
    cuts = []
    i = w
    while i < len(track) - w:
        if score[i] > thr:
            j = i + int(np.argmax(score[i:min(len(track) - w, i + w)]))
            cuts.append(j)
            i = j + w
        else:
            i += 1
    if not cuts:
        return [track]
    pieces, prev = [], 0
    for c in cuts:
        pieces.append(track[prev:c]); prev = c
    pieces.append(track[prev:])
    return [p for p in pieces if len(p) >= 8]
