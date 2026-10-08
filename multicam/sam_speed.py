"""How fast can the live small SAM go, and what does it lose (08.10.2026): the same door stretch, every 3rd frame, a few
variants -- seconds per frame, how busy the card is, and the people found against SAM 3.1's own tracks of that stretch.

usage (venv_sam3, the card): sam_speed.py [TAG] [N_FRAMES]   -> data/logs/sam_speed.json
The night teacher, if running, is suspended for the test and resumed after."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDENT = str(ROOT / 'runs' / 's31micro_a' / 'last.pt')


def gpu_sampler(stop, out):
    while not stop.is_set():
        try:
            r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, timeout=5).stdout.strip().split(',')
            out.append((float(r[0]), float(r[1])))
        except Exception:
            pass
        time.sleep(0.5)


def night_procs():
    try:
        import psutil
    except ImportError:
        return []
    return [p for p in psutil.process_iter(['cmdline']) if p.info['cmdline'] and any('door_night.py' in c for c in p.info['cmdline'])]


def frames(tag, n, stride, dst):
    import cv2
    shutil.rmtree(dst, ignore_errors=True)
    (dst / 'sam_in').mkdir(parents=True)
    cap = cv2.VideoCapture(str(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'video.mp4'))
    i = k = 0
    ks = []
    while k < n:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            cv2.imwrite(str(dst / 'sam_in' / ('%05d.jpg' % k)), cv2.resize(f, (1008, 1008), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            ks.append(i)
            k += 1
        i += 1
    cap.release()
    return ks


def found(tag, ks, dst):
    """teacher people (SAM 3.1 on this stretch) matched by the variant at IoU >= 0.5, on the same frames"""
    import sam31_reid as R
    T = R.Masks(ROOT / 'data' / 'sam31_door' / tag / 'cam1' / 'chunks.npz')
    S = R.Masks(dst / 'chunks.npz')
    by_t, by_s = {}, {}
    for r in range(len(T.rows)):
        by_t.setdefault(int(T.rows[r, 1]), []).append(r)
    for r in range(len(S.rows)):
        by_s.setdefault(int(S.rows[r, 1]), []).append(r)

    def full(M, r):
        m = np.zeros((1224, 2176), bool)
        x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
        c = M.crop(r).astype(bool)
        m[y1:y1 + c.shape[0], x1:x1 + c.shape[1]] = c[:1224 - y1, :2176 - x1]
        return m
    n_t = n_hit = n_s = 0
    ious = []
    for j, kf in enumerate(ks):
        tm = [full(T, r) for r in by_t.get(kf, []) if T.crop(r).sum() > 1500]
        sm = [full(S, r) for r in by_s.get(j, []) if S.crop(r).sum() > 1500]
        n_t += len(tm); n_s += len(sm)
        used = set()
        for a in tm:
            best, bi = 0, None
            for i, b in enumerate(sm):
                if i in used:
                    continue
                inter = (a & b).sum()
                iou = inter / max(1, (a | b).sum())
                if iou > best:
                    best, bi = iou, i
            if best >= 0.5:
                used.add(bi); n_hit += 1; ious.append(best)
    return {'teacher': n_t, 'found': n_hit, 'pred': n_s, 'recall': round(n_hit / max(1, n_t), 3),
            'precision': round(n_hit / max(1, n_s), 3), 'mask_iou': round(float(np.mean(ious)), 3) if ious else None}


def main(tag='door_20260919_32350', n='160'):
    import torch
    import door_micro as DM
    import sam31_segment as SG
    n = int(n)
    stopped = night_procs()
    for p in stopped:
        try:
            p.suspend()
        except Exception:
            pass
    print('night teacher suspended:', [p.pid for p in stopped], flush=True)
    res = {}
    try:
        variants = [('base', None, {}), ('e2e_e heads', str(ROOT / 'runs' / 's31e2e_e' / 'heads.pt'), {'RA_S31_NEWDET': '0.5'}),
                    ('base, 8 objects', None, {'max_obj': 8})]
        if os.environ.get('RA_SPEED_VARIANTS') == 'compile':          # 08.10: SAM 3.1's own torch.compile
            variants = [('base', None, {}), ('compiled', None, {'RA_S31_COMPILE': '1'})]
        for name, heads, opt in variants:
            for k_, v_ in opt.items():
                if k_.startswith('RA_'):
                    os.environ[k_] = v_
            dst = ROOT / 'data' / 'logs' / 'sam_speed' / name.replace(' ', '_').replace(',', '')
            ks = frames(tag, n, 3, dst)
            pred = DM.build(STUDENT, heads)
            if opt.get('max_obj'):
                for m in pred.model.modules():
                    if hasattr(m, 'max_num_objects'):
                        m.max_num_objects = opt['max_obj']
            sess = DM.sessions_of(len(ks), 120, 8)
            tw = time.time()
            SG.label(pred, dst, sess[:1], None)              # warm up (the first session builds caches / compiles)
            SG.label(pred, dst, sess[:1], None)
            print(name, 'warm-up %.0f s' % (time.time() - tw), flush=True)
            stop, util = threading.Event(), []
            th = threading.Thread(target=gpu_sampler, args=(stop, util), daemon=True)
            th.start()
            torch.cuda.synchronize()
            t0 = time.time()
            SG.label(pred, dst, sess, None)
            torch.cuda.synchronize()
            sec = time.time() - t0
            stop.set(); th.join()
            frames_done = sum(e - s for s, e, _ in sess)
            q = found(tag, ks, dst)
            res[name] = dict(q, s_per_frame=round(sec / frames_done, 3), gpu_util=round(float(np.mean([u for u, _ in util])), 1) if util else None,
                             gpu_mem=round(float(np.max([m for _, m in util]))) if util else None)
            print(name, json.dumps(res[name]), flush=True)
            del pred
            torch.cuda.empty_cache()
            for k_ in opt:
                if k_.startswith('RA_'):
                    os.environ.pop(k_, None)
    finally:
        for p in stopped:
            try:
                p.resume()
            except Exception:
                pass
        print('night teacher resumed', flush=True)
    json.dump(res, open(ROOT / 'data' / 'logs' / ('sam_speed_%s.json' % os.environ.get('RA_SPEED_VARIANTS', 'heads')), 'w'), indent=1)
    print('done', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
