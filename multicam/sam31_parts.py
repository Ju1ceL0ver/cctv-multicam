"""Where SAM 3.1's time goes, per part, on one people-full stretch (05.10.2026): backbone, detector, tracker
propagation (memory attention + mask decoder), planning (association, hotstart), memory encoding.

usage (venv_sam3, the card): sam31_parts.py [N_FRAMES]  -> data/logs/sam31_parts.json"""
import glob
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main(n=48):
    import cv2
    import torch
    import sam31_segment as S
    n = int(n)
    vids = sorted(glob.glob(str(ROOT / 'data' / 'sam31_seg' / '*' / 'cam1' / 'video.mp4')))
    # the most crowded spot of the first window: start where the window's chunks have most people
    base = Path(vids[0]).parent
    rows = np.load(base / 'chunks.npz')['rows']
    ticks, cnt = np.unique(rows[:, 1].astype(int), return_counts=True)
    k0 = int(ticks[np.argmax(np.convolve(cnt, np.ones(n), 'same'))]) - n // 2
    sub = ROOT / 'data' / 'logs' / 'sam31_parts_frames'
    shutil.rmtree(sub, ignore_errors=True); sub.mkdir(parents=True)
    cap = cv2.VideoCapture(str(base / 'video.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, k0))
    for k in range(n):
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(str(sub / ('%05d.jpg' % k)), cv2.resize(f, (1008, 1008), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
    pred = S.build()
    m = pred.model
    T = defaultdict(list)

    def wrap(obj, name, key):
        f = getattr(obj, name)

        def g(*a, **kw):
            torch.cuda.synchronize(); t = time.perf_counter()
            r = f(*a, **kw)
            torch.cuda.synchronize(); T[key].append(time.perf_counter() - t)
            return r
        setattr(obj, name, g)
    wrap(m, 'run_backbone_and_detection', 'backbone+detector')
    wrap(m, 'run_tracker_propagation', 'tracker propagate')
    wrap(m, 'run_tracker_update_planning_phase', 'planning (assoc+hotstart+memory enc)')
    wrap(m, 'run_tracker_update_execution_phase', 'execution (new objects)')
    wrap(m, '_tracker_update_memories', '  of which memory encoder')
    bb = m.detector.backbone
    for nm in ('forward_image',):
        if hasattr(bb, nm):
            wrap(bb, nm, '  of which backbone')
    ntrk = []
    with torch.autocast('cuda', dtype=torch.bfloat16):
        sid = pred.handle_request(dict(type='start_session', resource_path=str(sub), offload_video_to_cpu=True))['session_id']
        pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
        t0 = time.perf_counter()
        for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
            ntrk.append(len((resp.get('outputs') or {}).get('out_obj_ids', [])))
        total = time.perf_counter() - t0
        pred.handle_request(dict(type='close_session', session_id=sid))
    rep = {'frames': n, 'start_tick': k0, 'window': base.parent.name, 'total_s_per_frame': round(total / max(1, len(ntrk)), 3),
           'objects_mean': round(float(np.mean(ntrk)), 1) if ntrk else 0, 'objects_max': int(max(ntrk)) if ntrk else 0,
           'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
           'parts_ms': {k: [round(1000 * float(np.median(v[2:] or v)), 1), len(v)] for k, v in T.items()}}
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'sam31_parts.json', 'w'), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main(*sys.argv[1:])
