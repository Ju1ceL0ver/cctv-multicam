"""cuDNN attention against the memory-efficient kernel inside the real layers (short; relative numbers when the trainer runs)."""
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

import slot_v2 as V

dev = 'cuda'
dt = torch.bfloat16
B, N, K, D, H = 2, 100, 18936, 192, 8


def timed(fn, n=15, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


layer = V.DecoderLayer().to(dev).train()
q = torch.randn(B, N, D, device=dev, requires_grad=True); mem = torch.randn(B, K, D, device=dev, requires_grad=True)
qpos = torch.randn(B, N, D, device=dev); mpos = torch.randn(B, K, D, device=dev)
mask = ((torch.rand(B, N, K, device=dev) < 0.7).float() * -1e4).to(dt).repeat_interleave(H, 0)
smask = ((torch.rand(B, N, N, device=dev) < 0.2).float() * -1e4).to(dt).repeat_interleave(H, 0)


def one():
    with torch.autocast('cuda', dt):
        o = layer(q, qpos, mem, mpos, mask, smask)
    o.float().sum().backward()


def out_of(backends):
    with torch.no_grad(), torch.autocast('cuda', dt), sdpa_kernel(backends, set_priority=True):
        return layer(q, qpos, mem, mpos, mask, smask).float()


print('one decoder layer (cross + self + ff), forward + backward:')
res = {}
for name, be in (('efficient (now)', [SDPBackend.EFFICIENT_ATTENTION]), ('math', [SDPBackend.MATH]),
                 ('cudnn first, efficient, math', [SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])):
    try:
        with sdpa_kernel(be, set_priority=True):
            ms = timed(one)
        res[name] = out_of(be)
        print('  %-32s %7.2f ms' % (name, ms), flush=True)
    except Exception as e:
        print('  %-32s failed: %s' % (name, repr(e)[:120]))
base = res.get('efficient (now)')
for k, v in res.items():
    if base is not None and k != 'efficient (now)':
        print('  max abs difference of the layer output, %s vs efficient: %.5f (output scale %.2f)' % (k, (v - base).abs().max(), base.abs().mean()))

# the ViT's own attention: 3600 tokens, 3 heads x 64, batch 2 (both cameras), forward only (the ViT is frozen) and forward + backward
print('\nthe ViT block attention (3600 tokens, 3 heads x 64 dims, batch 2):')
qv = torch.randn(2, 3, 3600, 64, device=dev, dtype=dt, requires_grad=True); kv = torch.randn_like(qv, requires_grad=True); vv = torch.randn_like(qv, requires_grad=True)
for name, be in (('efficient', [SDPBackend.EFFICIENT_ATTENTION]), ('cudnn', [SDPBackend.CUDNN_ATTENTION]), ('math', [SDPBackend.MATH])):
    try:
        with sdpa_kernel(be):
            f = lambda: F.scaled_dot_product_attention(qv, kv, vv)
            print('  %-10s forward %6.2f ms' % (name, timed(f)), end='')
            def fb():
                F.scaled_dot_product_attention(qv, kv, vv).float().sum().backward()
            print('   forward+backward %6.2f ms' % timed(fb))
    except Exception as e:
        print('  %-10s failed: %s' % (name, repr(e)[:100]))
