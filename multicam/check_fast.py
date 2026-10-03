"""Old clip_step against the new one on the same clips, on the card (run with the trainer stopped).

  python check_fast.py runs/v2_a/epoch_23.pt [clips]

For every clip: the losses of both versions (they draw random numbers differently, so they agree within noise, not
to the last digit), the gradient norm, the seconds; then the count of host-device syncs and kernel launches of the new one."""
import collections
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import slot_v2 as V          # noqa: E402
import train_v2 as T         # noqa: E402
import v2_data as VD         # noqa: E402

ck = sys.argv[1] if len(sys.argv) > 1 else 'runs/v2_a/epoch_23.pt'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
model = V.build()
model.load_state_dict(torch.load(ROOT / ck, map_location='cpu')['model'])
model.to(dev).train()
model.to(memory_format=torch.channels_last)
for n, p in model.body.named_parameters():
    p.requires_grad_(not n.startswith('dinov3'))
asm = VD.Assembler(dev)
tags = [t for t in sorted({t for t, _ in VD.window_cams(exclude=('20260918',))}) if t != '20260923_21601']
amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=True)
near, far = VD.Clips(tags, 4, 3, 21), VD.Clips(tags, 2, 3750, 22)
clips = [(near.sample(), False) for _ in range(N)] + [(far.sample(), True)]


def gnorm():
    return float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in model.parameters() if p.grad is not None)))


def run(fn, clip, is_far, seed):
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.autocast(**amp):
        _, lg = fn(model, asm, clip, dev, True, rng, is_far, True, scale=1.0)
    torch.cuda.synchronize()
    return lg, time.perf_counter() - t, gnorm()


run(T.clip_step_old, clips[0][0], False, 0)
run(T.clip_step, clips[0][0], False, 0)                      # warm-up of both (cuDNN autotune, allocator, pinned pool)
tot = {'old': 0.0, 'new': 0.0}
moments = 0
worst = 0.0
for i, (c, f) in enumerate(clips):
    a, ta, ga = run(T.clip_step_old, c, f, 100 + i)
    b, tb, gb = run(T.clip_step, c, f, 100 + i)
    tot['old'] += ta; tot['new'] += tb; moments += len(c['frames'])
    print('\nclip %d (%s, %d moments): old %.2f s  new %.2f s  (x%.2f)   grad norm old %.3f new %.3f' % (i, 'far' if f else 'near', len(c['frames']), ta, tb, ta / tb, ga, gb))
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        rel = abs((va or 0) - (vb or 0)) / max(1e-3, abs(va or 0), abs(vb or 0))
        worst = max(worst, rel) if k in a and k in b else worst
        print('   %-8s old %9.4f  new %9.4f  %s' % (k, va if va is not None else float('nan'), vb if vb is not None else float('nan'), '' if k in a and k in b else '   <-- only one side'))
print('\nTOTAL %d moments: old %.2f s (%.3f s/moment)   new %.2f s (%.3f s/moment)   speed-up x%.2f' % (moments, tot['old'], tot['old'] / moments, tot['new'], tot['new'] / moments, tot['old'] / tot['new']))

# syncs and launches of the new step
lines = collections.Counter()
torch.cuda.set_sync_debug_mode('warn')
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter('always')
    run(T.clip_step, clips[1][0], False, 7)
torch.cuda.set_sync_debug_mode(0)
for w in W:
    lines['%s:%d' % (Path(w.filename).name, w.lineno)] += 1
print('\nimplicit host-device syncs in one near clip (new): %d' % sum(lines.values()))
for k, v in lines.most_common(15):
    print('  %4d  %s' % (v, k))
from torch.profiler import ProfilerActivity, profile  # noqa: E402
with profile(activities=[ProfilerActivity.CPU]) as prof:
    run(T.clip_step, clips[1][0], False, 8)
ka = prof.key_averages()
launches = sum(e.count for e in ka if e.key in ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC', 'cudaLaunchKernelExC_v11060'))
calls = {e.key: e.count for e in ka if e.key in ('cudaStreamSynchronize', 'cudaMemcpyAsync', 'aten::item', 'aten::nonzero', 'aten::_local_scalar_dense')}
print('kernel launches per moment (new): %d;  calls per clip: %s' % (launches / len(clips[1][0]['frames']), calls))
print('worst relative difference of a loss term between the versions: %.0f %%' % (100 * worst))

# ---- the same clips with torch.compile on the maps (ViT is frozen and inside; convolutions + elementwise get fused)
if 'compile' in sys.argv:
    t0 = time.time()
    try:
        model.maps = torch.compile(model.maps)
        for i in range(3):
            run(T.clip_step, clips[0][0], False, 5)           # compile + recompiles happen here
        print('\ncompile and warm-up: %.0f s' % (time.time() - t0), flush=True)
        tc = 0.0
        for i, (c, f) in enumerate(clips):
            b, tb, gb = run(T.clip_step, c, f, 100 + i)
            tc += tb
        print('COMPILED maps: %d moments in %.2f s (%.3f s/moment) vs eager new %.3f  -> x%.2f' % (moments, tc, tc / moments, tot['new'] / moments, tot['new'] / tc), flush=True)
    except Exception as e:
        import traceback
        print('compile failed:', repr(e)[:400]); traceback.print_exc()
