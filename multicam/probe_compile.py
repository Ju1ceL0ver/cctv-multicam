"""Does torch.compile work on this machine when MSVC's environment is loaded (run through vcvars64.bat)? Compiles the maps
of the v2 model and times it against eager (relative when the trainer is running)."""
import sys
import time

import torch

import slot_v2 as V
import v2_data as VD
import random

dev = 'cuda'
model = V.build()
model.load_state_dict(torch.load(sys.argv[1], map_location='cpu')['ema'])
model.to(dev).eval().to(memory_format=torch.channels_last)
asm = VD.Assembler(dev)
tags = sorted({t for t, _ in VD.window_cams(exclude=('20260918',))})
clip = VD.Clips(tags, 2, 3, 5).sample()
rgb, bgv, sta, cam_ids = asm(clip['frames'][0], ('cam1', 'cam2'), False, random.Random(0))
world = asm.last_world
with torch.no_grad():
    bg_sem = model.body.vit(bgv)[0]


def run(fn, n=10):
    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
        for _ in range(3):
            fn(rgb, sta, cam_ids, bg_sem, world)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n):
            out = fn(rgb, sta, cam_ids, bg_sem, world)
        torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n, out


ms0, o0 = run(model.maps)
print('eager maps forward: %.1f ms' % ms0, flush=True)
t0 = time.time()
cm = torch.compile(model.maps)
try:
    ms1, o1 = run(cm)
    print('compiled maps forward: %.1f ms  (compile + warm-up %.0f s)  x%.2f' % (ms1, time.time() - t0, ms0 / ms1), flush=True)
    print('max abs difference of pix: %.5f' % (o0['pix'].float() - o1['pix'].float()).abs().max(), flush=True)
except Exception as e:
    print('compile failed:', repr(e)[:600], flush=True)
