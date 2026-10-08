"""09.10: does fast_planning_copy (sam31_segment) change anything SAM 3.1 outputs? One 240-frame session of a busy door
stretch, three runs in one process: original deep copy twice (the run-to-run noise of the card), then the fast copy.
Every frame's object ids and masks compared; the time of each run printed.

usage (venv_sam3, the card): sam31_fastcopy_check.py [TAG] [N]"""
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ['RA_S31_FASTCOPY'] = '0'                  # build with the original first
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch


def run(pred, src):
    import sam31_video as SV
    out, t0 = [], time.time()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        sid = pred.handle_request(dict(type='start_session', resource_path=str(src), offload_video_to_cpu=True))['session_id']
        first = pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
        resps = [(0, first.get('outputs', {}) or {})]
        for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
            resps.append((int(resp.get('frame_index', len(resps))), resp.get('outputs', {}) or {}))
        pred.handle_request(dict(type='close_session', session_id=sid))
    torch.cuda.synchronize()
    for k, o in resps:
        ms = SV.masks_of(o, (1008, 1008))
        out.append((k, [(int(i), hashlib.md5(np.packbits(m.astype(bool)).tobytes()).hexdigest(), m.astype(bool)[::4, ::4].copy()) for i, p, m in ms]))
    torch.cuda.empty_cache()
    return out, time.time() - t0


def compare(a, b):
    same_ids = same_masks = frames = 0
    worst = 1.0
    for (ka, xa), (kb, xb) in zip(a, b):
        frames += 1
        ia, ib = [x[0] for x in xa], [x[0] for x in xb]
        if ia == ib:
            same_ids += 1
            if all(p[1] == q[1] for p, q in zip(xa, xb)):
                same_masks += 1
            else:
                for p, q in zip(xa, xb):
                    u = (p[2] | q[2]).sum()
                    worst = min(worst, (p[2] & q[2]).sum() / u if u else 1.0)
    return {'frames': frames, 'same_ids': same_ids, 'same_masks_bitwise': same_masks, 'worst_mask_iou': round(float(worst), 4)}


def main(tag='door_20260919_32350', n='240'):
    import door_micro as DM
    import sam31_segment as SG
    import sam_speed as SP
    dst = ROOT / 'data' / 'logs' / 'fastcopy_check'
    SP.frames(tag, int(n), 3, dst)
    pred = DM.build(str(ROOT / 'runs' / 's31micro_a' / 'last.pt'), None)
    run(pred, dst / 'sam_in')                         # warm
    a, ta = run(pred, dst / 'sam_in')
    b, tb = run(pred, dst / 'sam_in')
    SG.fast_planning_copy()
    c, tc = run(pred, dst / 'sam_in')
    print('original vs original (noise):', compare(a, b), flush=True)
    print('original vs fast copy:', compare(a, c), flush=True)
    print('seconds: original %.1f, %.1f; fast copy %.1f' % (ta, tb, tc), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
