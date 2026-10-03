"""How fast can the v2 model run (forward only, both cameras of a moment)? Run with the card free.

  python bench_infer.py runs/v2_a/epoch_23.pt

Measures per moment: the assembler, maps (ViT + STA + FPN + pixel decoder), decode + cross-camera layer, the whole; for
eager bf16, eager fp16 (NaN check), torch.compile of the maps, and the batch of 2 moments in one call (throughput)."""
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import slot_v2 as V          # noqa: E402
import train_v2 as T         # noqa: E402
import v2_data as VD         # noqa: E402

ck = sys.argv[1] if len(sys.argv) > 1 else 'runs/v2_a/epoch_23.pt'
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
model = V.build()
model.load_state_dict(torch.load(ROOT / ck, map_location='cpu')['ema'])
model.to(dev).eval()
model.to(memory_format=torch.channels_last)
asm = VD.Assembler(dev)
tags = [t for t in sorted({t for t, _ in VD.window_cams(exclude=('20260918',))})]
clip = VD.Clips(tags, 4, 3, 5).sample()
cams = ('cam1', 'cam2')


def prepare(per):
    rgb, bgv, sta, cam_ids = asm(per, cams, False, random.Random(0))
    world = asm.last_world
    with torch.no_grad():
        bg_sem = model.body.vit(bgv)[0]
    return rgb, sta, cam_ids, bg_sem, world


prepared = [prepare(per) for per in clip['frames']]


def one(maps_fn, dtype, x):
    rgb, sta, cam_ids, bg_sem, world = x
    with torch.no_grad(), torch.autocast('cuda', dtype):
        m = maps_fn(rgb, sta, cam_ids, bg_sem, world)
        sc = torch.cat([model.scene.start(cam_ids[b:b + 1]) for b in range(2)], 0)
        r = model.decode(m, cam_ids, None, sc)
        r1, r2 = model.cross_cameras(T.split_b(r, 0), T.split_b(r, 1))
    return m, r1, r2


def bench(name, maps_fn, dtype, n=20):
    for x in prepared[:2]:
        one(maps_fn, dtype, x)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for i in range(n):
        m, r1, r2 = one(maps_fn, dtype, prepared[i % len(prepared)])
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t) / n
    bad = not (torch.isfinite(r1['obj']).all() and torch.isfinite(r2['box']).all())
    print('%-34s %6.1f ms per moment (2 cameras)  = %5.1f moments/s%s' % (name, 1000 * dt, 1 / dt, '   NaN/Inf in the output!' if bad else ''), flush=True)
    return dt


def parts(dtype, n=20):
    acc = {'assembler': 0.0, 'maps': 0.0, 'decode+cross': 0.0}
    per = clip['frames']
    for i in range(n + 2):
        f = per[i % len(per)]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        rgb, bgv, sta, cam_ids = asm(f, cams, False, random.Random(0)); world = asm.last_world
        with torch.no_grad():
            bg_sem = model.body.vit(bgv)[0]
        torch.cuda.synchronize(); t1 = time.perf_counter()
        with torch.no_grad(), torch.autocast('cuda', dtype):
            m = model.maps(rgb, sta, cam_ids, bg_sem, world)
            torch.cuda.synchronize(); t2 = time.perf_counter()
            sc = torch.cat([model.scene.start(cam_ids[b:b + 1]) for b in range(2)], 0)
            r = model.decode(m, cam_ids, None, sc)
            model.cross_cameras(T.split_b(r, 0), T.split_b(r, 1))
        torch.cuda.synchronize(); t3 = time.perf_counter()
        if i >= 2:
            acc['assembler'] += t1 - t0; acc['maps'] += t2 - t1; acc['decode+cross'] += t3 - t2
    print('   split: ' + '   '.join('%s %.1f ms' % (k, 1000 * v / n) for k, v in acc.items()), flush=True)


print('moment = one tick of both cameras; real time at 12.5 ticks/s needs <= 80 ms per moment\n')
bench('eager bf16', model.maps, torch.bfloat16)
parts(torch.bfloat16)
try:
    bench('eager fp16', model.maps, torch.float16)
except Exception as e:
    print('eager fp16 failed:', repr(e)[:200])
try:
    t0 = time.time()
    cm = torch.compile(model.maps)
    bench('torch.compile(maps) bf16', cm, torch.bfloat16)
    print('   (compile + warm-up took %.0f s)' % (time.time() - t0), flush=True)
except Exception as e:
    print('torch.compile(maps) failed:', repr(e)[:300])
try:
    cm2 = torch.compile(model.maps, mode='max-autotune-no-cudagraphs')
    t0 = time.time()
    bench('torch.compile(maps) max-autotune', cm2, torch.bfloat16)
    print('   (compile + warm-up took %.0f s)' % (time.time() - t0), flush=True)
except Exception as e:
    print('max-autotune failed:', repr(e)[:300])
print('\npeak memory %.1f GB' % (torch.cuda.max_memory_allocated() / 2 ** 30))
