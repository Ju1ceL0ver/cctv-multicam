"""Where do the decoder's milliseconds go? Real shapes of one decode() call of the v2 model (both cameras in a batch),
forward + backward, bf16, on a free card:

  python bench_decoder.py

  memory tokens K = 14400 + 3600 + 920 (strides 8, 16, 32 of 1280 x 720) + 16 scene tokens, queries N ~ 100 (tracks +
  48 proposals + denoising hints), heads 8 x 24 dims, batch 2.
Compared: the cross-attention as it is (nn.MultiheadAttention with a float mask of shape B*heads x N x K), the same through
F.scaled_dot_product_attention with the mask broadcast over the heads, no mask at all, a boolean mask, Mask2Former's
round-robin (one level per layer), and the parts around the attention (mask builder, K/V projections)."""
import time

import torch
import torch.nn.functional as F

import slot_v2 as V

dev = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
B, N, D, H = 2, 100, 192, 8
K = 14400 + 3600 + 920 + 16
sizes = [(90, 160), (45, 80), (23, 40)]
amp = dict(device_type='cuda', dtype=torch.bfloat16)
g = torch.Generator(device=dev).manual_seed(0)


def t_(fn, n=15, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1000 * (time.perf_counter() - t0) / n


def make(k):
    q = torch.randn(B, N, D, device=dev, requires_grad=True)
    mem = torch.randn(B, k, D, device=dev, requires_grad=True)
    mpos = torch.randn(B, k, D, device=dev)
    qpos = torch.randn(B, N, D, device=dev)
    return q, qpos, mem, mpos


layer = V.DecoderLayer().to(dev).train()
q, qpos, mem, mpos = make(K)
mask_b = torch.rand(B, N, K, device=dev) < 0.7                     # True = not allowed
mask_f = (mask_b.float() * -1e4).to(torch.bfloat16).repeat_interleave(H, 0)


def run_layer(attn_mask):
    def f():
        with torch.autocast(**amp):
            out = layer(q, qpos, mem, mpos, attn_mask, None)
        out.float().sum().backward()
    return f


print('B=%d  N=%d  K=%d  heads=%d x %d dims\n' % (B, N, K, H, D // H))
print('one full decoder layer (cross + self + ff), forward + backward:')
print('  float mask, nn.MultiheadAttention (now)   %7.2f ms' % t_(run_layer(mask_f)))
print('  no mask                                   %7.2f ms' % t_(run_layer(None)))
K2 = 3600 + 920 + 16
q2, qpos2, mem2, mpos2 = make(K2)
mask2 = (torch.rand(B, N, K2, device=dev) < 0.7).float().mul(-1e4).to(torch.bfloat16).repeat_interleave(H, 0)


def run_layer2():
    with torch.autocast(**amp):
        out = layer(q2, qpos2, mem2, mpos2, mask2, None)
    out.float().sum().backward()


print('  float mask, K = %5d (levels 16 + 32 only) %7.2f ms' % (K2, t_(run_layer2)))
K3 = 14400 + 16
q3, qpos3, mem3, mpos3 = make(K3)
mask3 = (torch.rand(B, N, K3, device=dev) < 0.7).float().mul(-1e4).to(torch.bfloat16).repeat_interleave(H, 0)


def run_layer3():
    with torch.autocast(**amp):
        out = layer(q3, qpos3, mem3, mpos3, mask3, None)
    out.float().sum().backward()


print('  float mask, K = %5d (level 8 only)        %7.2f ms' % (K3, t_(run_layer3)))

# the attention alone
qh = torch.randn(B, H, N, D // H, device=dev, dtype=torch.bfloat16, requires_grad=True)
kh = torch.randn(B, H, K, D // H, device=dev, dtype=torch.bfloat16, requires_grad=True)
vh = torch.randn(B, H, K, D // H, device=dev, dtype=torch.bfloat16, requires_grad=True)
bias4 = mask_f.view(B, H, N, K)
bias_bc = (mask_b.float() * -1e4).to(torch.bfloat16)[:, None]          # B x 1 x N x K, broadcast over the heads


def sdpa(mask, backend=None):
    def f():
        o = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        o.float().sum().backward()
    return f


print('\nthe cross-attention alone (forward + backward):')
print('  SDPA, float mask B x H x N x K (as now)   %7.2f ms' % t_(sdpa(bias4)))
try:
    print('  SDPA, mask broadcast over heads           %7.2f ms' % t_(sdpa(bias_bc)))
except Exception as e:
    print('  SDPA broadcast mask failed:', repr(e)[:120])
print('  SDPA, boolean mask                        %7.2f ms' % t_(sdpa(~mask_b[:, None])))
print('  SDPA, no mask                             %7.2f ms' % t_(sdpa(None)))
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    for be in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH):
        with sdpa_kernel(be):
            print('  %-42s%7.2f ms' % (str(be).split('.')[-1] + ' (float mask)', t_(sdpa(bias4))))
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        print('  FLASH_ATTENTION (no mask)                 %7.2f ms' % t_(sdpa(None)))
except Exception as e:
    print('  backend comparison failed:', repr(e)[:160])

# the parts around it
mv = torch.randn(B, N, 64, device=dev)
pix = [torch.randn(B, 64, h, w, device=dev) for h, w in sizes]


def build_mask():
    m = torch.cat([torch.einsum('bsc,bchw->bshw', mv, p).flatten(2) for p in pix], 2) < 0
    m[m.all(-1)] = False
    am = torch.cat([m, torch.zeros(*m.shape[:2], 16, dtype=torch.bool, device=dev)], 2)
    return (am.float() * -1e4).to(torch.bfloat16).repeat_interleave(H, 0)


print('\naround the attention:')
print('  mask builder (einsum + bool fix + float + repeat) %6.2f ms   (per layer, forward only)' % t_(build_mask))


def kv_proj():
    with torch.no_grad(), torch.autocast(**amp):
        return layer.cross.in_proj_weight, F.linear(mem + mpos, layer.cross.in_proj_weight[D:2 * D]), F.linear(mem, layer.cross.in_proj_weight[2 * D:])


print('  K and V projections of the memory                %6.2f ms   (per layer, forward only)' % t_(kv_proj))
print('\npeak memory %.2f GB' % (torch.cuda.max_memory_allocated() / 2 ** 30))
