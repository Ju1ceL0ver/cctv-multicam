"""Where does a v2 training step go? Run with the trainer stopped (it needs the card alone).

  python prof_v2.py runs/v2_a/epoch_22.pt

Three measurements on the same clips, the same way train_v2 runs them (bf16 autocast, ViT frozen):
  1. phases: every big piece of clip_step timed with a card sync around it (upper bounds, they break the overlap);
  2. torch.profiler: card-busy share of the wall time, the top CPU ops and kernels, counts of syncs and launches;
  3. every implicit host-device sync with the line of code that caused it (torch.cuda.set_sync_debug_mode)."""
import collections
import json
import random
import sys
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import slot_v2 as V          # noqa: E402
import train_slots as TS     # noqa: E402
import train_v2 as T         # noqa: E402
import v2_data as VD         # noqa: E402

ck = sys.argv[1] if len(sys.argv) > 1 else 'runs/v2_a/epoch_22.pt'
N_CLIPS = int(sys.argv[2]) if len(sys.argv) > 2 else 4
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
rng = random.Random(0)
tags = sorted({t for t, _ in VD.window_cams(exclude=('20260918',))})
tags = [t for t in tags if t != '20260923_21601']
amp = dict(device_type='cuda', dtype=torch.bfloat16, enabled=True)
t0 = time.time()
near = VD.Clips(tags, 4, 3, 7)
far = VD.Clips(tags, 2, 3750, 8)
clips = [near.sample() for _ in range(N_CLIPS + 2)]
fars = [far.sample() for _ in range(2)]
print('sampled %d clips in %.0f s (main process; the trainer does this in workers)' % (len(clips) + len(fars), time.time() - t0), flush=True)


def step(clip, is_far=False):
    with torch.autocast(**amp):
        T.clip_step(model, asm, clip, dev, True, rng, is_far, True, scale=1.0)


def sync():
    torch.cuda.synchronize()


# warm-up (cuDNN autotune, allocator)
for c in clips[:2]:
    step(c)
sync()
res = {}

# ---- 1. phases -------------------------------------------------------------------------------------------------------
acc = collections.defaultdict(float)
cnt = collections.defaultdict(int)


def timed(name, fn):
    def w(*a, **k):
        sync(); t = time.perf_counter()
        r = fn(*a, **k)
        sync(); acc[name] += time.perf_counter() - t; cnt[name] += 1
        return r
    return w


ORIG = {'match_frame': T.match_frame, 'frame_losses': T.frame_losses, 'ident_losses': T.ident_losses, 'targets_tensor': T.targets_tensor, 'make_dn': TS.make_dn}
T.match_frame = timed('match (hungarian)', T.match_frame)
T.frame_losses = timed('frame_losses', T.frame_losses)
T.ident_losses = timed('ident_losses', T.ident_losses)
T.targets_tensor = timed('targets_tensor', T.targets_tensor)
TS.make_dn = timed('make_dn', TS.make_dn)
model.maps = timed('model.maps (ViT+STA+FPN+pixel)', model.maps)
model.decode = timed('model.decode', model.decode)
model.cross_cameras = timed('cross_cameras', model.cross_cameras)
model.next_tracks = timed('next_tracks', model.next_tracks)
model.pooled = timed('pooled', model.pooled)
_orig_call = VD.Assembler.__call__


def asm_timed(self, *a, **k):
    sync(); t = time.perf_counter()
    r = _orig_call(self, *a, **k)
    sync(); acc['assembler (frames -> inputs)'] += time.perf_counter() - t; cnt['assembler (frames -> inputs)'] += 1
    return r


VD.Assembler.__call__ = asm_timed
wall = 0.0
frames = 0
for c in clips[2:]:
    sync(); t = time.perf_counter(); step(c); sync(); wall += time.perf_counter() - t; frames += len(c['frames'])
print('\n== 1. phases, %d clips (%d moments, both cameras), %.2f s per moment, %.2f s per T=4 clip' % (N_CLIPS, frames, wall / frames, wall / N_CLIPS))
inner = sum(acc.values())
for k, v in sorted(acc.items(), key=lambda x: -x[1]):
    print('  %-40s %6.2f s   %4.1f %%   x%d' % (k, v, 100 * v / wall, cnt[k]))
print('  %-40s %6.2f s   %4.1f %%   (backward, the rest of the Python, the syncs)' % ('everything else', wall - inner, 100 * (wall - inner) / wall))
res['phases'] = {k: round(v / frames, 4) for k, v in acc.items()}
res['s_per_moment'] = round(wall / frames, 3)
# put the originals back for the next measurements
T.match_frame, T.frame_losses, T.ident_losses, T.targets_tensor, TS.make_dn = (ORIG[k] for k in ('match_frame', 'frame_losses', 'ident_losses', 'targets_tensor', 'make_dn'))
for name in ('maps', 'decode', 'cross_cameras', 'next_tracks', 'pooled'):
    delattr(model, name)
VD.Assembler.__call__ = _orig_call
sync()

# ---- 2. profiler -----------------------------------------------------------------------------------------------------
from torch.profiler import ProfilerActivity, profile  # noqa: E402
for c in clips[:1]:
    step(c)
sync()
t = time.perf_counter()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    for c in clips[2:4]:
        step(c)
    sync()
wall2 = time.perf_counter() - t
ka = prof.key_averages()
cuda_total = sum(e.self_device_time_total for e in ka) / 1e6 if hasattr(ka[0], 'self_device_time_total') else sum(e.self_cuda_time_total for e in ka) / 1e6
print('\n== 2. profiler, 2 clips: wall %.2f s, card busy with kernels %.2f s = %.0f %% of the wall time' % (wall2, cuda_total, 100 * cuda_total / wall2))


def col(e, name):
    return getattr(e, name, getattr(e, name.replace('device', 'cuda'), 0))


print('  top CPU ops (self CPU time):')
for e in sorted(ka, key=lambda e: -e.self_cpu_time_total)[:14]:
    print('   %-46s %7.0f ms  x%d' % (e.key[:46], e.self_cpu_time_total / 1e3, e.count))
print('  top CUDA kernels/ops (self device time):')
for e in sorted(ka, key=lambda e: -col(e, 'self_device_time_total'))[:12]:
    print('   %-46s %7.0f ms  x%d' % (e.key[:46], col(e, 'self_device_time_total') / 1e3, e.count))
launches = sum(e.count for e in ka if e.key in ('cudaLaunchKernel', 'cuLaunchKernel', 'cudaLaunchKernelExC', 'cudaLaunchKernelExC_v11060'))
syncs = {e.key: e.count for e in ka if e.key in ('cudaStreamSynchronize', 'cudaDeviceSynchronize', 'aten::item', 'aten::_local_scalar_dense', 'aten::nonzero', 'cudaMemcpyAsync', 'cudaMemcpy', 'aten::to', 'aten::index_put_', 'aten::index')}
print('  kernel launches: %d (%.0f per moment)' % (launches, launches / sum(len(c['frames']) for c in clips[2:4])))
print('  calls:', json.dumps(syncs))
res['gpu_busy'] = round(cuda_total / wall2, 3)
res['launches_per_moment'] = round(launches / sum(len(c['frames']) for c in clips[2:4]))
prof.export_chrome_trace(str(ROOT / 'data/logs/prof_v2_trace.json'))

# ---- 3. syncs by line ------------------------------------------------------------------------------------------------
import traceback  # noqa: E402
lines = collections.Counter()
torch.cuda.set_sync_debug_mode('warn')
with warnings.catch_warnings(record=True) as W:
    warnings.simplefilter('always')
    step(clips[4])
    sync()
torch.cuda.set_sync_debug_mode(0)
for w in W:
    lines['%s:%d' % (Path(w.filename).name, w.lineno)] += 1
print('\n== 3. implicit host-device syncs in one T=4 clip: %d, by line' % sum(lines.values()))
for k, v in lines.most_common(25):
    print('  %4d  %s' % (v, k))
res['syncs_per_clip'] = sum(lines.values())
res['sync_lines'] = dict(lines.most_common(25))
json.dump(res, open(ROOT / 'data/logs/prof_v2.json', 'w'), indent=1)
print('\nwrote data/logs/prof_v2.json and prof_v2_trace.json (open the trace in chrome://tracing or ui.perfetto.dev)')
