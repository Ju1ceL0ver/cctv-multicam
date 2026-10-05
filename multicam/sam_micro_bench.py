"""How fast SAM 3.1's own structure runs at our sizes (06.10.2026): 1280 x 720, width 256, random weights, the
3060, fp16, both cameras in one batch. Standard PyTorch layers with the same shapes and operations as SAM 3.1's
modules, to price the design before building it.

parts: a micro backbone (several candidates; maps at strides 4/8/16/32) -> two necks to 256 (detector, propagation)
-> detector: encoder layers over the 80x45 tokens with cross-attention to the constant "person" prompt, decoder
layers of 50 queries + presence, pixel decoder to stride 4, mask = query . pixel map -> tracker: memory attention
layers (self + cross to the memory: 4 conditioning + 6 recent frames + object pointers), SAM's two-way mask decoder
for 16 multiplexed people x (1 + 3 candidate) tokens, masks at stride 4 -> memory encoder (16 masks down to stride 16,
fused with the frame, pooled to the memory grid).

usage (venv_sam3, the card): sam_micro_bench.py  -> data/logs/sam_micro_bench.json"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
W, H = 1280, 720
D = 256
B = 2                                   # both cameras


class MHA(nn.Module):
    def __init__(self, d=D, h=8):
        super().__init__()
        self.h = h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q, k, v):
        b, n, d = q.shape
        sp = lambda x: x.view(b, -1, self.h, d // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(sp(self.q(q)), sp(self.k(k)), sp(self.v(v)))
        return self.o(y.transpose(1, 2).reshape(b, n, d))


class Layer(nn.Module):
    """pre-norm: self-attention, cross-attention, FFN 2048 (SAM 3.1's encoder/decoder/memory-attention layers)."""

    def __init__(self, cross=True, ffn=2048):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.sa = MHA()
        self.ca = MHA() if cross else None
        self.ff = nn.Sequential(nn.Linear(D, ffn), nn.ReLU(), nn.Linear(ffn, D))

    def forward(self, x, mem=None):
        y = self.n1(x)
        x = x + self.sa(y, y, y)
        if self.ca is not None and mem is not None:
            x = x + self.ca(self.n2(x), mem, mem)
        return x + self.ff(self.n3(x))


class TwoWay(nn.Module):
    """SAM's mask decoder layer: tokens self-attn, tokens->image, MLP, image->tokens."""

    def __init__(self):
        super().__init__()
        self.sa, self.t2i, self.i2t = MHA(), MHA(), MHA()
        self.mlp = nn.Sequential(nn.Linear(D, 2048), nn.ReLU(), nn.Linear(2048, D))
        self.ns = nn.ModuleList(nn.LayerNorm(D) for _ in range(4))

    def forward(self, t, img):
        t = self.ns[0](t + self.sa(t, t, t))
        t = self.ns[1](t + self.t2i(t, img, img))
        t = self.ns[2](t + self.mlp(t))
        img = self.ns[3](img + self.i2t(img, t, t))
        return t, img


class CX(nn.Module):
    def __init__(self, d=D):
        super().__init__()
        self.dw = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.n = nn.GroupNorm(1, d)
        self.p1, self.p2 = nn.Conv2d(d, 4 * d, 1), nn.Conv2d(4 * d, d, 1)

    def forward(self, x):
        return x + self.p2(F.gelu(self.p1(self.n(self.dw(x)))))


def neck(chs):
    return nn.ModuleList(nn.Sequential(nn.Conv2d(c, D, 1), nn.Conv2d(D, D, 3, padding=1)) for c in chs)


class Detector(nn.Module):
    def __init__(self, enc=2, dec=3, queries=50):
        super().__init__()
        self.enc = nn.ModuleList(Layer() for _ in range(enc))
        self.dec = nn.ModuleList(Layer() for _ in range(dec))
        self.q = nn.Parameter(torch.randn(1, queries + 1, D))
        self.prompt = nn.Parameter(torch.randn(1, 4, D))          # the constant "person" prompt tokens
        self.pix = nn.ModuleList([nn.Conv2d(D + 64, 64, 3, padding=1), nn.Conv2d(64 + 32, 64, 3, padding=1)])
        self.qp = nn.Linear(D, 64)
        self.box = nn.Linear(D, 4)
        self.score = nn.Linear(D, 1)

    def forward(self, f16, f8, f4):
        b, d, h, w = f16.shape
        x = f16.flatten(2).transpose(1, 2)
        pr = self.prompt.expand(b, -1, -1)
        for L in self.enc:
            x = L(x, pr)
        q = self.q.expand(b, -1, -1)
        for L in self.dec:
            q = L(q, x)
        p = x.transpose(1, 2).reshape(b, d, h, w)
        p = self.pix[0](torch.cat([F.interpolate(p, size=f8.shape[-2:]), f8], 1))
        p = self.pix[1](torch.cat([F.interpolate(p, size=f4.shape[-2:]), f4], 1))
        masks = torch.einsum('bqd,bdhw->bqhw', self.qp(q[:, 1:]), p)
        return masks, self.box(q), self.score(q), x


class Tracker(nn.Module):
    def __init__(self, layers=2, objs=16, mem_stride=2):
        super().__init__()
        self.layers = nn.ModuleList(Layer() for _ in range(layers))
        self.dec = nn.ModuleList(TwoWay() for _ in range(2))
        self.tok = nn.Parameter(torch.randn(1, objs * 4 + 1, D))
        self.up = nn.Sequential(nn.ConvTranspose2d(D, 64, 2, 2), nn.GELU(), nn.ConvTranspose2d(64, 32, 2, 2))
        self.hyper = nn.Linear(D, 32)
        self.objs, self.mem_stride = objs, mem_stride
        # memory encoder: 16 masks (x2 channels as in SAM 3.1) down to stride 16, fused with the frame
        chans = [2 * objs, 64, D]
        self.down = nn.Sequential(*[m for i in range(2) for m in (nn.Conv2d(chans[i], chans[i + 1], 3, 2, 1), nn.GELU())])
        self.fuse = nn.Sequential(CX(), CX())

    def forward(self, f16, f4, memory):
        b, d, h, w = f16.shape
        x = f16.flatten(2).transpose(1, 2)
        for L in self.layers:
            x = L(x, memory)
        t = self.tok.expand(b, -1, -1)
        img = x
        for L in self.dec:
            t, img = L(t, img)
        up = self.up(img.transpose(1, 2).reshape(b, d, h, w))
        masks = torch.einsum('bqc,bchw->bqhw', self.hyper(t[:, 1:1 + self.objs]), up)
        return masks

    def encode_memory(self, f16, masks_full):
        m = self.down(masks_full)
        m = self.fuse(m + f16)
        if self.mem_stride > 1:
            m = F.avg_pool2d(m, self.mem_stride)
        return m.flatten(2).transpose(1, 2)


def timed(f, n=30):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(n):
        f()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / n / B                           # ms a frame


@torch.no_grad()
def main():
    import timm
    torch.backends.cudnn.benchmark = True
    dev = 'cuda'
    rep = {'input': [W, H], 'batch_frames': B, 'note': 'ms per frame, fp16, random weights, PyTorch eager (no TensorRT)'}
    img = torch.randn(B, 3, H + 16, W, device=dev, dtype=torch.half).contiguous(memory_format=torch.channels_last)  # 736: /32
    backs = {}
    for name in ('mobilenetv4_conv_small', 'hgnetv2_b0', 'starnet_s2', 'mobilenetv4_conv_medium'):
        try:
            m = timm.create_model(name, pretrained=False, features_only=True, out_indices=(-4, -3, -2, -1))
        except Exception as ex:
            backs[name] = 'n/a: %s' % str(ex)[:60]
            continue
        red = m.feature_info.reduction()
        m = m.to(dev).half().eval().to(memory_format=torch.channels_last)
        params = sum(p.numel() for p in m.parameters()) / 1e6
        backs[name] = {'ms': round(timed(lambda: m(img)), 2), 'params_m': round(params, 2), 'strides': red}
        print(name, backs[name], flush=True)
    rep['backbones'] = backs
    # shapes from strides 4/8/16/32 of a 1280x736 frame
    f4 = torch.randn(B, 32, 184, 320, device=dev, dtype=torch.half)
    f8 = torch.randn(B, 64, 92, 160, device=dev, dtype=torch.half)
    f16 = torch.randn(B, D, 46, 80, device=dev, dtype=torch.half)
    nk = neck([64, 128, 256, 512]).to(dev).half()
    c = [torch.randn(B, ch, 736 // s, 1280 // s, device=dev, dtype=torch.half) for ch, s in ((64, 4), (128, 8), (256, 16), (512, 32))]
    rep['necks_x2_ms_full256'] = round(2 * timed(lambda: [nk[i](c[i]) for i in range(3)]), 2)
    thin = nn.ModuleList([nn.Sequential(nn.Conv2d(64, 32, 1), nn.Conv2d(32, 32, 3, padding=1)),
                          nn.Sequential(nn.Conv2d(128, 64, 1), nn.Conv2d(64, 64, 3, padding=1)),
                          nn.Sequential(nn.Conv2d(256, D, 1), nn.Conv2d(D, D, 3, padding=1))]).to(dev).half()
    rep['necks_x2_ms'] = round(2 * timed(lambda: [thin[i](c[i]) for i in range(3)]), 2)
    parts = {}
    for enc, dec in ((2, 3), (3, 4), (6, 6)):
        det = Detector(enc, dec).to(dev).half().eval()
        parts['detector_%d+%d' % (enc, dec)] = round(timed(lambda: det(f16, f8, f4)), 2)
    masks_full = torch.randn(B, 32, 184, 320, device=dev, dtype=torch.half)
    for layers, mem_stride in ((2, 2), (2, 1), (4, 1)):
        trk = Tracker(layers, mem_stride=mem_stride).to(dev).half().eval()
        tokens = (46 // mem_stride) * (80 // mem_stride)
        memory = torch.randn(B, 10 * tokens + 16 * 4, D, device=dev, dtype=torch.half)   # 4 cond + 6 recent + 16 pointers x 4
        parts['tracker_%dlayers_memgrid_%dx%d' % (layers, 80 // mem_stride, 46 // mem_stride)] = round(timed(lambda: trk(f16, f4, memory)), 2)
        parts['memory_encoder_stride%d' % mem_stride] = round(timed(lambda: trk.encode_memory(f16, masks_full)), 2)
    rep['parts'] = parts
    best = min((v['ms'] for v in backs.values() if isinstance(v, dict)), default=0)
    rep['total_light_ms'] = round(best + rep['necks_x2_ms'] + parts['detector_2+3'] + parts['tracker_2layers_memgrid_40x23']
                                  + parts['memory_encoder_stride2'], 1)
    print(json.dumps(rep, indent=1), flush=True)
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'sam_micro_bench.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
