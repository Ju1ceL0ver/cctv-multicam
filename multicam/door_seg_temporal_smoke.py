"""Finite compatibility check, no optimizer update or checkpoint writes.

python door_seg_temporal_smoke.py runs/doorseg_c_background/last.pt
Uses four ordered TRAIN frames, real YOLO criterion and BF16 on the GPU.
"""
import json
import sys

import cv2
import torch

import door_seg as S
from door_seg_sequence import DoorSegTemporal, sequence_starts, sample_sequence, sequence_loss


def main(checkpoint):
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.manual_seed(7)
    source = S.Source(S.sources()[0])
    try:
        starts = sequence_starts(source, 4)
        if not starts:
            raise ValueError('first TRAIN source has no continuous four-frame sequence')
        # Prefer a co-visible pair, for an actual contrastive anchor check.
        at = next((at for at in starts[:200] if len(source.people(*at)) >= 2), starts[0])
        frames = sample_sequence(source, 4, seed=7, at=at)
    finally:
        source.close()
    net = DoorSegTemporal('cuda')
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    S.checkpoint_finite(state)
    net.load(state['model'])
    step = state['step']
    del state
    net.yolo.train()
    criterion = net.yolo.init_criterion()
    for _ in range(step // 2000):
        criterion.update()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, stats = sequence_loss(net, [frames], criterion, 'cuda')
    if not torch.isfinite(loss):
        raise ValueError('nonfinite sequence loss')
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10, error_if_nonfinite=True)
    for name, module in [('residual', net.temporal.residual), ('fingerprint', net.temporal.fingerprint)]:
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()):
            raise ValueError(name + ' has no training signal')
    print(json.dumps(dict(stats, finite=True, loss=float(loss.detach()), grad_norm=float(norm),
                          peak_gb=torch.cuda.max_memory_allocated()/1e9, source=str(source.d),
                          tick=at[1], checkpoint_step=step, updated_weights=False)), flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
