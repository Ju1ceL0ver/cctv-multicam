"""What does maps() consist of? FLOPs per module (torch's FlopCounterMode) and milliseconds per part (relative when the trainer runs)."""
import random
import sys
import time
from collections import defaultdict

import torch
from torch.utils.flop_counter import FlopCounterMode

import slot_v2 as V
import v2_data as VD

dev = 'cuda'
torch.backends.cudnn.benchmark = True
model = V.build()
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu')['ema'])
model.to(dev).eval().to(memory_format=torch.channels_last)
asm = VD.Assembler(dev)
tags = sorted({t for t, _ in VD.window_cams(exclude=('20260918',))})
clip = VD.Clips(tags, 2, 3, 5).sample()
rgb, bgv, sta, cam_ids = asm(clip['frames'][0], ('cam1', 'cam2'), False, random.Random(0))
world = asm.last_world
amp = dict(device_type='cuda', dtype=torch.bfloat16)
with torch.no_grad(), torch.autocast(**amp):
    bg_sem = model.body.vit(bgv)[0]


def ms(fn, n=10):
    with torch.no_grad(), torch.autocast(**amp):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


with torch.no_grad(), torch.autocast(**amp):
    sem = model.body.vit(rgb)
b = model.body
parts = [
    ('ViT on the frame (12 blocks, 3600 tokens) x2 cameras', lambda: model.body.vit(rgb)),
    ('STA convolutions (10-channel 2176x1248 detail input)', lambda: [b.sta.conv4(b.sta.conv3(b.sta.conv2(b.sta.stem(sta))))]),
    ('the whole backbone (ViT + STA + fusion)', lambda: model.body(rgb, sta, cam_ids, bg_sem)),
    ('maps() = backbone + FPN + pixel/ReID/centre/boundary/zone heads + memory tokens', lambda: model.maps(rgb, sta, cam_ids, bg_sem, world)),
]
print('milliseconds for one moment (2 cameras), forward only:')
for name, fn in parts:
    print('  %-84s %7.1f ms' % (name, ms(fn)), flush=True)
print('  %-84s %7.1f ms   (per background change in real use, cached)' % ('ViT on the background frames', ms(lambda: model.body.vit(bgv))))

with FlopCounterMode(model, depth=3, display=False) as fc, torch.no_grad(), torch.autocast(**amp):
    model.maps(rgb, sta, cam_ids, bg_sem, world)
tot = fc.get_total_flops()
by = fc.get_flop_counts()
rows = defaultdict(float)
for mod, ops in by.items():
    if mod.count('.') <= 2 and mod != 'Global':
        rows[mod] = sum(ops.values())
print('\nGFLOPs of maps() for the moment: %.0f' % (tot / 1e9))
for k, v in sorted(rows.items(), key=lambda x: -x[1])[:22]:
    print('  %-46s %8.1f GFLOP  %4.1f %%' % (k[:46], v / 1e9, 100 * v / tot))
n = lambda m: sum(p.numel() for p in m.parameters()) / 1e6
print('\nparameters (M): ViT %.1f, STA %.2f, FPN %.2f, whole model %.1f' % (n(model.body.dinov3), n(model.body.sta), n(model.fpn), n(model)))
