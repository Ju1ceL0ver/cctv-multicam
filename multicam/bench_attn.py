"""The decoder's cross-attention is the biggest single cost of a step (see bench_decoder.py): 18 ms per layer for forward +
backward at B=2, 8 heads x 24 dims, N=100 queries, K=18936 memory tokens. Which way of computing it is fastest, and is the
result the same?  (Short, fine to run next to the trainer: the numbers are relative.)

  python bench_attn.py"""
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

dev = 'cuda'
B, H, N, K, Dh = 2, 8, 100, 18936, 24
g = torch.Generator(device=dev).manual_seed(0)
dt = torch.bfloat16


def make(h=H, dh=Dh, dtype=dt):
    q = torch.randn(B, h, N, dh, device=dev, dtype=dtype, generator=g, requires_grad=True)
    k = torch.randn(B, h, K, dh, device=dev, dtype=dtype, generator=g, requires_grad=True)
    v = torch.randn(B, h, K, dh, device=dev, dtype=dtype, generator=g, requires_grad=True)
    return q, k, v


mask_b = torch.rand(B, 1, N, K, device=dev, generator=g) < 0.7
bias = (mask_b.float() * -1e4).to(dt).expand(B, H, N, K).contiguous()


def timed(fn, n=20, warm=4):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t) / n


def fb(attn, q, k, v, *rest):
    def f():
        o = attn(q, k, v, *rest)
        o.float().sum().backward()
    return f


def sdpa(q, k, v, m):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m)


def manual(q, k, v, m):
    s = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5) + m
    return torch.matmul(torch.softmax(s.float(), -1).to(q.dtype), v)


def manual_bf16_softmax(q, k, v, m):
    s = torch.matmul(q, k.transpose(-1, -2)) * (q.shape[-1] ** -0.5) + m
    return torch.matmul(torch.softmax(s, -1), v)


q, k, v = make()
ref = None
with torch.no_grad():
    ref = manual(q.float(), k.float(), v.float(), bias.float())            # fp32 truth
print('cross-attention B=%d heads=%d N=%d K=%d head_dim=%d, forward + backward (ms) and max abs error against fp32\n' % (B, H, N, K, Dh))


def row(name, fn_attn, args, ref_out=None):
    try:
        f = fb(fn_attn, *args)
        ms = timed(f)
        with torch.no_grad():
            o = fn_attn(*args)
        err = (o.float() - ref).abs().max().item() if ref_out is not None and o.shape == ref.shape else float('nan')
        print('  %-46s %7.2f ms   err %.4f' % (name, ms, err), flush=True)
    except Exception as e:
        print('  %-46s failed: %s' % (name, repr(e)[:100]), flush=True)


with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    row('SDPA efficient (what runs now)', sdpa, (q, k, v, bias), ref)
with sdpa_kernel(SDPBackend.MATH):
    row('SDPA math', sdpa, (q, k, v, bias), ref)
try:
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        row('SDPA cuDNN attention', sdpa, (q, k, v, bias), ref)
except Exception as e:
    print('  cuDNN attention unavailable:', repr(e)[:80])
row('matmul + fp32 softmax + matmul', manual, (q, k, v, bias), ref)
row('matmul + bf16 softmax + matmul', manual_bf16_softmax, (q, k, v, bias), ref)
# head dim padded to 32 (kernels like multiples of 32 / 64)
pad = lambda t: F.pad(t, (0, 8))
qp, kp, vp = [pad(t.detach()).requires_grad_(True) for t in (q, k, v)]
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    scale = Dh ** -0.5
    row('SDPA efficient, head_dim padded 24 -> 32', lambda a, b, c, m: F.scaled_dot_product_attention(a, b, c, attn_mask=m, scale=scale)[..., :Dh], (qp, kp, vp, bias), ref)
# other head layouts (would change the model, for information)
for h, dh in ((4, 48), (6, 32), (12, 16)):
    q2, k2, v2 = make(h, dh)
    b2 = (mask_b.float() * -1e4).to(dt).expand(B, h, N, K).contiguous()
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        row('for information: %d heads x %d dims (efficient)' % (h, dh), sdpa, (q2, k2, v2, b2))
    with sdpa_kernel(SDPBackend.MATH):
        row('for information: %d heads x %d dims (math)' % (h, dh), sdpa, (q2, k2, v2, b2))
# the same with a 4x smaller memory (levels 16 + 32)
K2 = 4536
q3 = q.detach().clone().requires_grad_(True)
k3 = k.detach()[:, :, :K2].clone().requires_grad_(True); v3 = v.detach()[:, :, :K2].clone().requires_grad_(True)
b3 = bias[:, :, :, :K2].contiguous()
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    row('K = %d (levels 16 + 32 only), efficient' % K2, sdpa, (q3, k3, v3, b3))
with sdpa_kernel(SDPBackend.MATH):
    row('K = %d (levels 16 + 32 only), math' % K2, sdpa, (q3, k3, v3, b3))

# the whole decoder layer with the math backend around it
import slot_v2 as V
layer = V.DecoderLayer().to(dev).train()
qq = torch.randn(B, N, 192, device=dev, requires_grad=True); mem = torch.randn(B, K, 192, device=dev, requires_grad=True)
qpos = torch.randn(B, N, 192, device=dev); mpos = torch.randn(B, K, 192, device=dev)
mask_f = (mask_b[:, 0].float() * -1e4).to(dt).repeat_interleave(H, 0)


def one_layer():
    with torch.autocast('cuda', dt):
        o = layer(qq, qpos, mem, mpos, mask_f, None)
    o.float().sum().backward()


print('\nwhole decoder layer (cross + self + ff), forward + backward:')
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    print('  efficient (now)   %7.2f ms' % timed(one_layer))
with sdpa_kernel(SDPBackend.MATH):
    print('  math              %7.2f ms' % timed(one_layer))
print('\npeak memory %.2f GB' % (torch.cuda.max_memory_allocated() / 2 ** 30))
