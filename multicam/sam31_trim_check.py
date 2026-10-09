"""09.10: drop SAM 3.1's cached outputs of frames already handed out (state['cached_frame_outputs'], ~2 GB in a busy
240-frame session) -- same masks and ids frame by frame? peak memory? One session without, one with (keep the last K).

usage (venv_sam3, the card): sam31_trim_check.py [TAG] [N] [KEEP]"""
import hashlib
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import numpy as np
import torch


def run(pred, src, keep=None):
    import sam31_video as SV
    import sam31_segment as SG
    out, t0 = [], time.time()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        sid = pred.handle_request(dict(type='start_session', resource_path=str(src), offload_video_to_cpu=True))['session_id']
        first = pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
        resps = [(0, SV.masks_of(first.get('outputs', {}) or {}, (1008, 1008)))]
        for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
            kl = int(resp.get('frame_index', len(resps)))
            resps.append((kl, SV.masks_of(resp.get('outputs', {}) or {}, (1008, 1008))))
            if keep is not None:
                SG.trim_cached_outputs(pred, sid, kl, keep)
        pred.handle_request(dict(type='close_session', session_id=sid))
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    for k, ms in resps:
        out.append((k, [(int(i), hashlib.md5(np.packbits(m.astype(bool)).tobytes()).hexdigest(), m.astype(bool)[::4, ::4].copy()) for i, p, m in ms]))
    torch.cuda.empty_cache()
    return out, time.time() - t0, peak


def main(tag='door_20260919_32350', n='240', keep='32'):
    import door_micro as DM
    import sam31_fastcopy_check as FC
    import sam_speed as SP
    dst = ROOT / 'data' / 'logs' / 'trim_check'
    SP.frames(tag, int(n), 3, dst)
    pred = DM.build(str(ROOT / 'runs' / 's31micro_a' / 'last.pt'), None)
    run(pred, dst / 'sam_in')                       # warm
    a, ta, pa = run(pred, dst / 'sam_in')
    b, tb, pb = run(pred, dst / 'sam_in', int(keep))
    print('as is: %.1f s, peak %.2f GB; trimmed (keep %s): %.1f s, peak %.2f GB' % (ta, pa, keep, tb, pb), flush=True)
    print('as is vs trimmed:', FC.compare(a, b), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
