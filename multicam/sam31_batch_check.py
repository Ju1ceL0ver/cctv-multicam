"""09.10: the small SAM's detector a few frames at a time (batched_grounding_batch_size; it was set to 1 for the big
ViT-L teacher, whose batch of 16 overflowed the card). Each frame's detection does not depend on the others, so the
outputs should stay the same -- checked frame by frame against batch 1 -- while the card works in bigger pieces.
One 240-frame session of a busy door stretch per batch size; time and peak memory printed.

usage (venv_sam3, the card; RA_S31_COMPILE as live): sam31_batch_check.py [TAG] [N] [SIZES]"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch


def set_batch(pred, b):
    for m in pred.model.modules():
        if hasattr(m, 'batched_grounding_batch_size'):
            m.batched_grounding_batch_size = b


def main(tag='door_20260919_32350', n='240', sizes='1,2,4'):
    import door_micro as DM
    import sam31_fastcopy_check as FC
    import sam_speed as SP
    dst = ROOT / 'data' / 'logs' / 'batch_check'
    SP.frames(tag, int(n), 3, dst)
    pred = DM.build(str(ROOT / 'runs' / 's31micro_a' / 'last.pt'), None)
    ref = None
    for b in [int(x) for x in sizes.split(',')]:
        set_batch(pred, b)
        FC.run(pred, dst / 'sam_in')                     # warm (compiles this size)
        FC.run(pred, dst / 'sam_in')
        torch.cuda.reset_peak_memory_stats()
        out, sec = FC.run(pred, dst / 'sam_in')
        peak = torch.cuda.max_memory_allocated() / 2 ** 30
        cmp = FC.compare(ref, out) if ref is not None else '-'
        if ref is None:
            ref = out
        print('batch %d: %.1f s (%.0f ms/frame), GPU peak %.2f GB, vs batch 1: %s' % (b, sec, 1000 * sec / len(out), peak, cmp), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
