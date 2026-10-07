"""SAM 3.1 with the distilled image encoder against SAM 3.1 itself, on 23.09 (never seen by the student) (05.10.2026).

The same stretches, the same sessions, the same detector, tracker and rules; only the encoder differs. Per frame the
two sets of masks are paired one to one (IoU, Hungarian): found (teacher's people the lite one also has, IoU >= 0.5),
extra (lite's people the teacher has not), mask IoU of the pairs. Over the stretch, numbers are compared like a
tracker's: IDF1 (the best one-to-one map of teacher ids to lite ids over all frames) and id switches (a teacher id
whose paired lite id changes). Seconds a frame for both. A side-by-side film of the first stretch: teacher left.

usage (venv_sam3, the card): sam31_lite_eval.py CKPT [N_STRETCHES] [TICKS] [--cache DIR] [--out JSON] [--preview JPG]
                             [--film MP4|none]   (the teacher's answers are kept in --cache and reused)"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
COL = [(66, 197, 245), (80, 220, 100), (245, 160, 66), (220, 90, 220), (240, 220, 70), (160, 100, 255), (60, 200, 200),
       (200, 160, 90), (120, 230, 180), (250, 120, 160), (90, 140, 250), (230, 230, 230)]


def stretches(n, ticks):
    """The most crowded `ticks`-long stretch of each 23.09 window camera, up to n."""
    import sam31_distill as SD
    out = []
    for d in sorted((ROOT / 'data' / 'sam31_seg').glob(SD.TEST_DAY + '_*/cam*')):
        rows = np.load(d / 'chunks.npz')['rows']
        t, c = np.unique(rows[:, 1].astype(int), return_counts=True)
        dense = np.zeros(int(t.max()) + 1); dense[t] = c
        k0 = int(np.argmax(np.convolve(dense, np.ones(ticks), 'valid')))
        out.append((d, k0, float(np.convolve(dense, np.ones(ticks), 'valid')[k0] / ticks)))
    out.sort(key=lambda x: -x[2])
    return out[:n]


def run(pred, folder):
    import torch
    import sam31_video as SV
    res, t0 = {}, time.perf_counter()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        sid = pred.handle_request(dict(type='start_session', resource_path=str(folder), offload_video_to_cpu=True))['session_id']
        first = pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
        res[0] = SV.masks_of(first.get('outputs', {}) or {}, (252, 252))
        for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
            res[int(resp.get('frame_index', len(res)))] = SV.masks_of(resp.get('outputs', {}) or {}, (252, 252))
        pred.handle_request(dict(type='close_session', session_id=sid))
    torch.cuda.synchronize()
    return res, (time.perf_counter() - t0) / max(1, len(res))


def compare(T, S):
    from scipy.optimize import linear_sum_assignment
    found = total = extra = 0
    ious, pairs = [], []                       # (frame, teacher id, lite id)
    for k in sorted(T):
        a, b = T[k], S.get(k, [])
        total += len(a)
        if not a or not b:
            extra += len(b)
            continue
        A = np.stack([m.ravel() for _, _, m in a]).astype(np.float32)
        B = np.stack([m.ravel() for _, _, m in b]).astype(np.float32)
        inter = A @ B.T
        iou = inter / (A.sum(1)[:, None] + B.sum(1)[None] - inter + 1e-6)
        r, c = linear_sum_assignment(-iou)
        ok = [(i, j) for i, j in zip(r, c) if iou[i, j] >= 0.5]
        found += len(ok)
        extra += len(b) - len(ok)
        for i, j in ok:
            ious.append(iou[i, j]); pairs.append((k, a[i][0], b[j][0]))
    # IDF1: best one-to-one map of ids over the stretch
    tids = sorted({p[1] for p in pairs} | {i for k in T for i, _, _ in T[k]})
    sids = sorted({p[2] for p in pairs} | {i for k in S for i, _, _ in S[k]})
    M = np.zeros((len(tids), len(sids)))
    ti, si = {t: i for i, t in enumerate(tids)}, {s: i for i, s in enumerate(sids)}
    for _, t, s in pairs:
        M[ti[t], si[s]] += 1
    idtp = 0
    if M.size:
        r, c = linear_sum_assignment(-M)
        idtp = M[r, c].sum()
    n_t = sum(len(v) for v in T.values()); n_s = sum(len(v) for v in S.values())
    idf1 = 2 * idtp / max(1, n_t + n_s)
    last, switches = {}, 0
    for _, t, s in sorted(pairs):
        if t in last and last[t] != s:
            switches += 1
        last[t] = s
    return {'teacher_people': int(total), 'found': round(found / max(1, total), 4), 'extra': int(extra),
            'precision': round(found / max(1, found + extra), 4), 'mask_iou': round(float(np.mean(ious)) if ious else 0, 4),
            'idf1': round(float(idf1), 4), 'id_switches': int(switches),
            'teacher_ids': len({i for k in T for i, _, _ in T[k]}), 'lite_ids': len({i for k in S for i, _, _ in S[k]})}


def film(folder, T, S, out):
    import cv2
    import day_proxy
    W, H = 960, 960
    enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (2 * W, H + 50),
                            '-r', '12.5', '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '24', '-pix_fmt', 'yuv420p', str(out)],
                           stdin=subprocess.PIPE)
    for k in sorted(T):
        img = cv2.resize(cv2.imread(str(folder / ('%05d.jpg' % k))), (W, H))
        halves = []
        for res, name in ((T[k], 'SAM 3.1 (teacher)'), (S.get(k, []), 'SAM 3.1 lite (student encoder)')):
            vis, over = img.copy(), img.copy()
            for i, p, m in res:
                m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                col = COL[i % len(COL)]
                over[m] = col
                cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(vis, cs, -1, col, 2)
                ys, xs = np.nonzero(m)
                if len(ys):
                    cv2.putText(vis, str(i), (int(xs.mean()) - 10, max(20, int(ys.min()) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 5)
                    cv2.putText(vis, str(i), (int(xs.mean()) - 10, max(20, int(ys.min()) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            vis = cv2.addWeighted(vis, 0.65, over, 0.35, 0)
            bar = np.zeros((50, W, 3), np.uint8)
            cv2.putText(bar, name, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            halves.append(np.vstack([bar, vis]))
        enc.stdin.write(np.hstack(halves).tobytes())
    enc.stdin.close(); enc.wait()


def tune(pred):
    """RA_S31_NEWDET / RA_S31_SCORE: SAM 3.1's thresholds for a new person and for a detection (0.65 / 0.4),
    lowered for a student whose scores run a little lower than the teacher's."""
    import os
    m = pred.model
    if os.environ.get('RA_S31_NEWDET'):
        m.new_det_thresh = float(os.environ['RA_S31_NEWDET'])
    if os.environ.get('RA_S31_SCORE'):
        m.score_threshold_detection = float(os.environ['RA_S31_SCORE'])
    return m.new_det_thresh, m.score_threshold_detection


def save_res(path, res):
    import pickle
    pickle.dump({k: [(i, p, np.packbits(m)) for i, p, m in v] for k, v in res.items()}, open(path, 'wb'))


def load_res(path):
    import pickle
    d = pickle.load(open(path, 'rb'))
    return {k: [(i, p, np.unpackbits(m)[:252 * 252].reshape(252, 252).astype(bool)) for i, p, m in v] for k, v in d.items()}


def preview(folder, T, S, out):
    """The most crowded frame: teacher | student, side by side."""
    import cv2
    k = max(T, key=lambda q: len(T[q]))
    img = cv2.resize(cv2.imread(str(folder / ('%05d.jpg' % k))), (640, 640))
    halves = []
    for res, name in ((T[k], 'SAM 3.1'), (S.get(k, []), 'SAM 3.1 + student encoder')):
        vis, over = img.copy(), img.copy()
        for i, p, m in res:
            m = cv2.resize(m.astype(np.uint8), (640, 640), interpolation=cv2.INTER_NEAREST) > 0
            over[m] = COL[i % len(COL)]
            cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, COL[i % len(COL)], 2)
        vis = cv2.addWeighted(vis, 0.6, over, 0.4, 0)
        cv2.rectangle(vis, (0, 0), (640, 34), (0, 0, 0), -1)
        cv2.putText(vis, '%s: %d people' % (name, len(res)), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        halves.append(vis)
    cv2.imwrite(str(out), np.hstack(halves), [cv2.IMWRITE_JPEG_QUALITY, 85])


def main(ckpt, n=3, ticks=240, cache=None, out=None, preview_path=None, film_path='', heads=None):
    import cv2
    import torch
    import sam31_distill as SD
    import sam31_segment as S
    n, ticks = int(n), int(ticks)
    cache = Path(cache) if cache else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    student = SD.load_student(str(ROOT / ckpt) if not Path(ckpt).is_absolute() else ckpt)
    pred = S.build()
    tune(pred)
    tri = pred.model.detector.backbone.vision_backbone
    teacher_trunk = tri.trunk
    if heads:                                   # stage 2: the short heads too (only the cached teacher answers are used then)
        import sam31_heads as SH
        SH.install(pred, torch.load(heads, map_location='cpu', weights_only=False))
    if os.environ.get('RA_TRACKER_CKPT') and Path(os.environ['RA_TRACKER_CKPT']).exists():    # 07.10: a tracker taught on clips
        import sam31_e2e_trk as ET
        tk = torch.load(os.environ['RA_TRACKER_CKPT'], map_location='cpu', weights_only=False)
        own = dict(ET.find_tracker(pred).named_parameters())
        with torch.no_grad():
            for n, v in tk['state'].items():
                if n in own and own[n].shape == v.shape:
                    own[n].copy_(v.to(own[n].device, own[n].dtype))
    rep = {'ckpt': ckpt, 'stretches': []}
    tmp = ROOT / 'data' / 'logs' / 'lite_eval_frames'
    for si, (d, k0, dens) in enumerate(stretches(n, ticks)):
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        cap = cv2.VideoCapture(str(d / 'video.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k0)
        for k in range(ticks):
            ok, f = cap.read()
            if not ok:
                break
            cv2.imwrite(str(tmp / ('%05d.jpg' % k)), cv2.resize(f, (1008, 1008), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
        cap.release()
        key = cache / ('%s_%s_%d_%d.pkl' % (d.parent.name, d.name, k0, ticks)) if cache else None
        if key is not None and key.exists():
            T, t_sec = load_res(key), None
        else:
            tri.trunk = teacher_trunk
            T, t_sec = run(pred, tmp)
            if key is not None:
                save_res(key, T)
        torch.cuda.empty_cache()
        tri.trunk = student
        Sres, s_sec = run(pred, tmp)
        torch.cuda.empty_cache()
        r = compare(T, Sres)
        r.update({'window': d.parent.name + '/' + d.name, 'start_tick': k0, 'people_per_frame': round(dens, 2),
                  'teacher_s_per_frame': round(t_sec, 3) if t_sec else None, 'lite_s_per_frame': round(s_sec, 3)})
        rep['stretches'].append(r)
        print(json.dumps(r), flush=True)
        if si == 0 and preview_path:
            preview(tmp, T, Sres, preview_path)
        if si == 0 and film_path != 'none':
            film(tmp, T, Sres, Path(film_path) if film_path else ROOT / 'data' / 'logs' / ('sam31_lite_%s.mp4' % Path(ckpt).parent.name))
    keys = ('found', 'precision', 'mask_iou', 'idf1')
    w = np.array([r['teacher_people'] for r in rep['stretches']], float)
    rep['mean'] = {k: round(float(np.average([r[k] for r in rep['stretches']], weights=w)), 4) for k in keys}
    rep['mean']['id_switches'] = int(sum(r['id_switches'] for r in rep['stretches']))
    json.dump(rep, open(out or ROOT / 'data' / 'logs' / 'sam31_lite_eval.json', 'w'), indent=1)
    print(json.dumps(rep['mean']))


if __name__ == '__main__':
    a = sys.argv[1:]
    kw = {}
    for flag, name in (('--cache', 'cache'), ('--out', 'out'), ('--preview', 'preview_path'), ('--film', 'film_path'),
                       ('--heads', 'heads')):
        if flag in a:
            i = a.index(flag)
            kw[name] = a[i + 1]
            del a[i:i + 2]
    main(*a, **kw)
