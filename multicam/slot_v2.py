"""The slot model, v2: both cameras, people carried from frame to frame, their place on the floor.

Per camera frame (both cameras of one moment go through as one batch):
  * ViT-Tiny (DEIMv2-S, DINOv3-distilled) on RGB 1280 x 720 -- the meaning; its layer-3 tokens minus the same
    ViT's tokens of the empty hall (the long background, cached when predicting) -- what is new in the frame;
  * STA convolutions on 10 channels at 2176 x 1248 -- the detail: RGB, the empty hall, the hall a moment ago
    (BSUV-Net: backgrounds as images, not differences), the depth of the empty hall (Depth Anything V2 Large);
  * both joined per stride (DEIMv2), plus a learned vector of the camera.
Heads on the maps: per-pixel mask embedding (stride 4), identity and zone maps, people's centres and sizes
(stride 8) and the boundaries between touching people (stride 4).
Slots: proposals -- one per peak of the centre map (a person the ViT alone would miss still gets a slot) --
and tracks, the slots of the previous frame carried on with their memory. A small scene memory per camera
(tokens updated with a gate each frame) is part of what the slots look at. Decoder: masked attention
(Mask2Former), boxes refined per layer, denoising hints while training (DINO).
Both cameras' slots then meet in a cross-camera layer, attention biased by their distance on the floor.
Per slot: person-ness, box, mask, state (seen / hidden / gone), place (x, y on the floor and height of the
body's centre, their uncertainties, velocity), identity (clothes + body shape), zone (outside / inside / door).
SameHead: is slot a the same person as memory b -- from both identities and the context (time apart, floor
distance, cameras, how visible). World people, permanent staff and the delay module live in v2_track.py."""
import contextlib
import math

import torch
import torch.nn as nn
import torch.utils.checkpoint
import torch.nn.functional as F

from slot_model import (DEIMv2Tiny, FPN, DecoderLayer, ResBlock, conv_bn, inverse_sigmoid, mlp, sine_pos, tower,
                        IMAGENET)

D, P, E, R = 192, 128, 64, 128
STATES = 3            # seen, hidden, gone
ZONES = 3             # outside, inside, doorway
PROPOSALS = 48
SCENE = 16
REID = 256
XY_C, XY_S = (4.0, 4.0), 6.0      # the common floor frame (m): centre and half-size, so places are about -1..1
Z_S = 1.0                         # height of the body's centre (m) / Z_S
STA_IN = 10


_XYC = {}


def _xyc(dev):
    """XY_C as a tensor on dev, made once (a tensor built on the card every call forced a host-to-device sync)."""
    k = str(dev)
    if k not in _XYC:
        _XYC[k] = torch.tensor(XY_C, device=dev)
    return _XYC[k]


def attn_backend():
    """Prefer cuDNN's attention kernel (4x faster than the memory-efficient one for the decoder's cross-attention: 100 queries
    over 19 000 memory tokens, 24-dim heads; measured in bench_attn.py, same result to 5e-4), fall back to the others."""
    if torch.cuda.is_available():
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            return sdpa_kernel([SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH], set_priority=True)
        except Exception:
            pass
    return contextlib.nullcontext()


def box_pos(box, dim=D):
    """Sine embedding of cx cy w h (0..1) -> dim."""
    freq = torch.exp(torch.arange(dim // 8, device=box.device, dtype=torch.float32) * (-math.log(1000.0) / (dim // 8)))
    x = box.float()[..., None] * freq * 2 * math.pi
    return torch.cat([x.sin(), x.cos()], -1).flatten(-2).to(box.dtype)


def place_pos(xyz, dim=D):
    """Fourier features of a place on the floor (normalised x, y, z) -> dim."""
    freq = 2.0 ** torch.arange(dim // 6, device=xyz.device, dtype=torch.float32) * 0.5
    x = xyz.float()[..., None] * freq * math.pi
    return torch.cat([x.sin(), x.cos()], -1).flatten(-2)[..., :dim].to(xyz.dtype)


class Backbone(DEIMv2Tiny):
    """DEIMv2Tiny with the ViT on its own (smaller) RGB and the STA on the 10-channel detail input."""

    def __init__(self, drop_path=0.1):
        super().__init__(in_chans=STA_IN, drop_path=drop_path)
        self.diff = nn.Conv2d(192, 192, 1)
        nn.init.zeros_(self.diff.weight); nn.init.zeros_(self.diff.bias)        # step 0: the pretrained branch exactly
        self.cam = nn.Embedding(2, 192)
        nn.init.zeros_(self.cam.weight)

    def vit(self, rgb):
        """rgb: ImageNet-normalised B x 3 x 720 x 1280 -> three maps at stride 16 (layers 3, 7, 11)."""
        B, _, H, W = rgb.shape
        h, w = H // 16, W // 16
        return [f.transpose(1, 2).reshape(B, -1, h, w) for f, _ in self.dinov3(rgb[:, :, :h * 16, :w * 16])]

    def forward(self, rgb, sta, cam, bg_sem=None):
        """bg_sem: the long background's layer-3 map (vit(bg)[0]) -- computed by the caller once per background."""
        sem = self.vit(rgb)
        if bg_sem is not None:
            sem[0] = sem[0] + self.diff(sem[0] - bg_sem)
        sem = [F.interpolate(f, scale_factor=2.0 ** (1 - i), mode='bilinear', align_corners=False) for i, f in enumerate(sem)]
        c1 = self.sta.stem(sta)
        c2 = self.sta.conv2(c1); c3 = self.sta.conv3(c2); c4 = self.sta.conv4(c3)
        ce = self.cam(cam)[:, :, None, None]
        out = [c1]
        for k, (s, d) in enumerate(zip(sem, (c2, c3, c4))):
            s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False)
            out.append(self.norms[k](self.convs[k](torch.cat([s, d], 1))) + ce)
        return out


class SceneMemory(nn.Module):
    """SCENE tokens per camera: read by the slots, then updated from them with a gate (GRU-like)."""

    def __init__(self):
        super().__init__()
        self.init = nn.Parameter(torch.randn(2, SCENE, D) * 0.02)
        self.read = nn.MultiheadAttention(D, 8, batch_first=True)
        self.gate = nn.Linear(2 * D, D)
        self.cand = nn.Linear(2 * D, D)
        self.norm = nn.LayerNorm(D)

    def start(self, cam):
        return self.init[cam]

    def update(self, scene, slots, pad=None):
        new = self.read(scene, slots, slots, key_padding_mask=pad, need_weights=False)[0]
        z = torch.sigmoid(self.gate(torch.cat([scene, new], -1)))
        return self.norm((1 - z) * scene + z * torch.tanh(self.cand(torch.cat([scene, new], -1))))


class TrackUpdate(nn.Module):
    """A track's memory for the next frame from its memory and this frame's slot (gated)."""

    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(2 * D, D)
        self.cand = mlp(2 * D, D, D, depth=2)
        self.norm = nn.LayerNorm(D)

    def forward(self, h, q):
        x = torch.cat([h, q], -1)
        z = torch.sigmoid(self.gate(x))
        return self.norm((1 - z) * h + z * self.cand(x))


class CrossCamera(nn.Module):
    """Self-attention over both cameras' slots of one moment; a cross-camera pair's attention is lowered by
    alpha x its distance on the floor (alpha learned, starts gentle)."""

    def __init__(self, layers=2):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(D, 8, 4 * D, 0.0, batch_first=True, norm_first=True)
                                     for _ in range(layers)])
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.heads = 8

    def forward(self, q, cam, xy, pad):
        """q: N x S x D (both cameras' slots), cam: N x S, xy: N x S x 2 (metres), pad: N x S (True = empty)."""
        d = torch.cdist(xy.float(), xy.float())
        other = cam[:, :, None] != cam[:, None, :]
        bias = -F.softplus(self.alpha) * d * other
        bias = bias.masked_fill(pad[:, None, :], -1e4).to(q.dtype)
        bias = bias.repeat_interleave(self.heads, 0)
        for l in self.layers:
            q = l(q, src_mask=bias)
        return q


class SameHead(nn.Module):
    """P(same person) of a slot and a remembered person: identities and context."""

    CTX = 8   # dt (min), floor distance (m), their uncertainty, same camera, visibility a, b, zone a, b

    def __init__(self, hidden=512):
        super().__init__()
        self.net = mlp(4 * REID + self.CTX, hidden, 1, depth=4, drop=0.1, norm=True)

    def forward(self, a, b, ctx):
        """a, b: ... x 2*REID (clothes | shape), ctx: ... x CTX -> logit."""
        a = torch.cat([F.normalize(a[..., :REID], dim=-1), F.normalize(a[..., REID:], dim=-1)], -1)
        b = torch.cat([F.normalize(b[..., :REID], dim=-1), F.normalize(b[..., REID:], dim=-1)], -1)
        return self.net(torch.cat([a * b, (a - b).abs(), ctx], -1)).squeeze(-1)


class ConvBackbone(Backbone):
    """A light convolutional net (timm, pretrained) in place of the ViT: its maps at strides 8, 16, 32 of the
    1280 x 720 RGB, each projected to the ViT's 192 channels; the stride-8 map takes the background difference.
    The STA branch, the joins and everything after are Backbone's. 29.09: RepViT-M0.9 (4.7 M) fwd+bwd 134 ms
    against the ViT's 255 ms for the two frames of a moment (bb_bench3.json) -- a person is a local thing."""

    def __init__(self, name, drop_path=0.1, pretrained=True):
        super().__init__(drop_path=drop_path)
        del self.dinov3
        import timm
        self.cnn = timm.create_model(name, pretrained=pretrained, features_only=True)
        red, ch = self.cnn.feature_info.reduction(), self.cnn.feature_info.channels()
        self.idx = [red.index(s) for s in (8, 16, 32)]
        self.proj = nn.ModuleList([nn.Sequential(nn.Conv2d(ch[i], 192, 1, bias=False), nn.BatchNorm2d(192)) for i in self.idx])
        self.name = name

    def vit(self, rgb):
        o = self.cnn(rgb)
        return [p(o[i]) for p, i in zip(self.proj, self.idx)]

    def forward(self, rgb, sta, cam, bg_sem=None):
        sem = self.vit(rgb)
        if bg_sem is not None:
            sem[0] = sem[0] + self.diff(sem[0] - bg_sem)
        c1 = self.sta.stem(sta)
        c2 = self.sta.conv2(c1); c3 = self.sta.conv3(c2); c4 = self.sta.conv4(c3)
        ce = self.cam(cam)[:, :, None, None]
        out = [c1]
        for k, (s, d) in enumerate(zip(sem, (c2, c3, c4))):
            s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False)
            out.append(self.norms[k](self.convs[k](torch.cat([s, d], 1))) + ce)
        return out


class SlotV2(nn.Module):
    def __init__(self, layers=6, teacher_dims=(3840, 1024), backbone=None, pretrained_backbone=True, slim=False, rgb_size=None):
        """slim (29.09): lighter identity heads -- ReID towers one block deep, its output MLP 256 wide, SameHead 128
        wide (1.06 M -> 0.17 M: it only says yes or no) -- with layers=4 about 14 M at work instead of 17."""
        super().__init__()
        self.backbone_name = backbone
        self.rgb_size = tuple(rgb_size) if rgb_size else (1280, 720)     # the RGB backbone's input (w, h); the STA branch keeps 2176 x 1248
        self.body = ConvBackbone(backbone, pretrained=pretrained_backbone) if backbone else Backbone()
        c4, c8, c16, c32 = self.body.channels
        self.fpn = FPN([c8, c16, c32])
        self.pix16 = tower(P, P, 1)
        self.pix8 = tower(P, P, 2)
        self.lat4 = conv_bn(c4, E, 1)
        self.up8 = conv_bn(P, E, 1)
        self.pix4 = nn.Sequential(conv_bn(E, E, 3), nn.Conv2d(E, E, 1))
        self.reid16 = tower(P, R, 1 if slim else 2)
        self.reid8 = tower(P + R, R, 1 if slim else 2)
        self.zone_map = tower(P, 64, 2)
        # people's centres and sizes (stride 8), boundaries between people (stride 4)
        self.cen = nn.Sequential(tower(P, 96, 2), nn.Conv2d(96, 3, 1))
        self.bnd = nn.Sequential(conv_bn(E + E, 48, 3), ResBlock(48), nn.Conv2d(48, 1, 1))
        with torch.no_grad():
            self.cen[-1].bias[0].fill_(-4.6); self.bnd[-1].bias.fill_(-3.0)
        # decoder
        self.proj = nn.ModuleList([nn.Linear(P, D) for _ in range(3)])
        self.level = nn.Parameter(torch.zeros(3, D))
        self.scene_pos = nn.Parameter(torch.randn(SCENE, D) * 0.02)
        self.prop_content = nn.Linear(P, D)
        self.kind = nn.Parameter(torch.zeros(3, D))                 # proposal, track, hint
        self.pos_of_box = mlp(D, D, D, depth=2)
        self.dn_content = mlp(4, D, D, depth=2)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(layers)])
        self.norm = nn.LayerNorm(D)
        self.scene = SceneMemory()
        self.track_update = TrackUpdate()
        self.cross = CrossCamera()
        self.cross_in = mlp(D, D, D, depth=2)
        # per-slot outputs
        self.obj = mlp(D, D, 1, depth=2)
        self.box_delta = mlp(D, D, 4)
        self.mask_embed = mlp(D, D, E)
        self.state = mlp(D, D, STATES, depth=2)
        self.place = mlp(D, D, 8, depth=3)                           # x y z, log-variance x y z, velocity x y
        self.reid_out = mlp(D + R, 256 if slim else 512, 2 * REID, depth=3, drop=0.1, norm=True)
        self.zone_out = mlp(D + 64 + 4, 128, ZONES, depth=3, drop=0.1, norm=True)
        self.to_teacher = nn.ModuleList([mlp(REID, 512, t, depth=2) for t in teacher_dims])
        self.same = SameHead(128 if slim else 512)
        self.world_proj = nn.Linear(D, D)
        nn.init.zeros_(self.world_proj.weight); nn.init.zeros_(self.world_proj.bias)
        self.long_proj = nn.Linear(D, D)                              # a track's long memory (MeMOTR) into its query
        nn.init.zeros_(self.long_proj.weight); nn.init.zeros_(self.long_proj.bias)
        self.grad_checkpoint = False
        with torch.no_grad():
            self.obj[-1].bias.fill_(-4.6)

    def set_checkpointing(self, on=True):
        """Recompute the ViT blocks and decoder layers in the backward pass instead of keeping them."""
        self.grad_checkpoint = on
        if hasattr(self.body, 'dinov3'):
            self.body.dinov3.grad_checkpoint = on

    # ---------------------------------------------------------------- maps
    def maps(self, rgb, sta, cam, bg_sem=None, world=None):
        f4, f8, f16, f32 = self.body(rgb, sta, cam, bg_sem)
        p8, p16, p32 = self.fpn([f8, f16, f32])
        pix8 = self.pix8(p8 + F.interpolate(self.pix16(p16), size=p8.shape[-2:], mode='bilinear', align_corners=False))
        up = F.interpolate(self.up8(pix8), size=f4.shape[-2:], mode='bilinear', align_corners=False)
        pix = self.pix4(self.lat4(f4) + up)
        r16 = self.reid16(p16)
        reid_map = self.reid8(torch.cat([p8, F.interpolate(r16, size=p8.shape[-2:], mode='bilinear', align_corners=False)], 1))
        cen = self.cen(p8)
        bnd = self.bnd(torch.cat([pix, up], 1))
        mem, mpos, sizes = [], [], []
        H, W = rgb.shape[-2:]
        for i, (p, st) in enumerate(((p8, 8), (p16, 16), (p32, 32))):
            # the slots look at a grid the size of the ViT's input (1280 x 720): 3x fewer points than the
            # detail frame's, the masks keep the full detail (stride 4 of the 2176 frame)
            p = F.adaptive_avg_pool2d(p, (-(-H // st), -(-W // st)))
            b, c, h, w = p.shape
            mem.append(self.proj[i](p.flatten(2).transpose(1, 2)) + self.level[i])
            pos = sine_pos(h, w, D, p.device).to(p.dtype)[None].expand(b, -1, -1)
            if world is not None:                        # where in the hall each point of the picture is (PLANET)
                g = F.adaptive_avg_pool2d(world.float(), (h, w)).flatten(2).transpose(1, 2)       # b x hw x 3
                xyz = torch.cat([((g[..., :2] - _xyc(g.device)) / XY_S).clamp(-3, 3), g[..., 2:]], -1)
                pos = pos + self.world_proj(place_pos(xyz)).to(p.dtype)
            mpos.append(pos)
            sizes.append((h, w))
        return {'p8': p8, 'pix': pix, 'reid_map': reid_map, 'zone_map': self.zone_map(p8), 'cen': cen, 'bnd': bnd,
                'mem': torch.cat(mem, 1), 'mpos': torch.cat(mpos, 1), 'sizes': sizes}

    def proposals(self, m, k=PROPOSALS):
        """Top-k peaks of the centre map -> (content, box logit) of proposal slots."""
        heat = m['cen'][:, 0].float().sigmoid()
        B, h, w = heat.shape
        peak = heat * (heat == F.max_pool2d(heat[:, None], 3, 1, 1)[:, 0])
        score, idx = peak.flatten(1).topk(k)
        ys, xs = (idx // w).float(), (idx % w).float()
        size = m['cen'][:, 1:].float().flatten(2).gather(2, idx[:, None].expand(-1, 2, -1)).transpose(1, 2)
        wh = torch.sigmoid(size)                                          # w, h as fractions of the frame
        box = torch.stack([(xs + 0.5) / w, (ys + 0.5) / h, wh[..., 0], wh[..., 1]], -1).clamp(1e-3, 1 - 1e-3)
        feat = m['p8'].flatten(2).gather(2, idx[:, None].expand(-1, m['p8'].shape[1], -1)).transpose(1, 2)
        return self.prop_content(feat) + self.kind[0], inverse_sigmoid(box), score

    # ---------------------------------------------------------------- slots
    def slot_outputs(self, q, box_logit):
        qn = self.norm(q)
        box_logit = box_logit + self.box_delta(qn)
        return {'obj': self.obj(qn).squeeze(-1), 'box': box_logit.sigmoid(), 'box_logit': box_logit,
                'mask_vec': self.mask_embed(qn), 'state': self.state(qn), 'place': self.place(qn)}

    def attn_mask(self, mask_vec, pix_levels):
        m = torch.cat([torch.einsum('bsc,bchw->bshw', mask_vec.detach(), p).flatten(2) for p in pix_levels], 2) < 0
        return m & ~m.all(-1, keepdim=True)          # a slot that would see nothing sees everything (no boolean-index write: it syncs)

    def decode(self, m, cam, track=None, scene=None, dn=None):
        """One frame's slots. track: {'h': B x T x D memory, 'box': B x T x 4 last boxes, 'valid': B x T bool}.
        Returns the final slot outputs (tracks first, then proposals), their raw vectors, aux layers, dn."""
        B = m['pix'].shape[0]
        pc, pb, pscore = self.proposals(m)
        parts_q, parts_b, pad = [pc], [pb], [torch.zeros(B, pc.shape[1], dtype=torch.bool, device=pc.device)]
        T = 0
        if track is not None and track['h'].shape[1]:
            T = track['h'].shape[1]
            parts_q.insert(0, track['h'] + self.long_proj(track['long']) + self.kind[1]); parts_b.insert(0, inverse_sigmoid(track['box']))
            pad.insert(0, ~track['valid'])
        q, box_logit, pad = torch.cat(parts_q, 1), torch.cat(parts_b, 1), torch.cat(pad, 1)
        S = q.shape[1]
        scene = self.scene.start(cam) if scene is None else scene
        if scene.dim() == 2:
            scene = scene[None].expand(B, -1, -1)
        mem = torch.cat([m['mem'].expand(B, -1, -1), scene], 1)
        mpos = torch.cat([m['mpos'].expand(B, -1, -1), self.scene_pos[None].expand(B, -1, -1)], 1)
        if dn is not None:
            db = dn['box'].to(q.dtype)
            q = torch.cat([q, self.dn_content(db) + self.kind[2]], 1)
            box_logit = torch.cat([box_logit, inverse_sigmoid(dn['box'].float()).to(box_logit.dtype)], 1)
            pad = torch.cat([pad, torch.zeros(B, db.shape[1], dtype=torch.bool, device=q.device)], 1)
        N = q.shape[1]
        heads = self.layers[0].heads
        self_block = pad[:, None, :].expand(B, N, N).clone()
        if dn is not None:
            g = torch.full((N,), -1, device=q.device, dtype=torch.long)
            g[S:] = dn['group'].to(q.device)
            blk = torch.zeros(N, N, dtype=torch.bool, device=q.device)
            blk[:S, S:] = True
            blk[S:, S:] = g[S:, None] != g[None, S:]
            self_block = self_block | blk[None]
        eye = torch.eye(N, dtype=torch.bool, device=q.device)[None]
        self_mask = (self_block & ~eye).float() * -1e4
        self_mask = self_mask.to(q.dtype).repeat_interleave(heads, 0)
        pix_levels = [F.adaptive_avg_pool2d(m['pix'].detach(), s) for s in m['sizes']]
        cur = self.slot_outputs(q, box_logit)
        outs = []
        ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
        with attn_backend():
            for layer in self.layers:
                am = self.attn_mask(cur['mask_vec'], pix_levels)
                am = torch.cat([am, torch.zeros(*am.shape[:2], SCENE, dtype=torch.bool, device=am.device)], 2)
                qpos = self.pos_of_box(box_pos(cur['box_logit'].detach().sigmoid()))
                amf = (am.float() * -1e4).to(q.dtype).repeat_interleave(heads, 0)
                if ckpt:
                    q = torch.utils.checkpoint.checkpoint(layer, q, qpos, mem, mpos, amf, self_mask, use_reentrant=False)
                else:
                    q = layer(q, qpos, mem, mpos, amf, self_mask)
                cur = self.slot_outputs(q, cur['box_logit'].detach())
                outs.append(cur)
        split = lambda o, a, b: {k: v[:, a:b] for k, v in o.items()}
        res = dict(split(outs[-1], 0, S))
        res.update(q=q[:, :S], pad=pad[:, :S], tracks=T, aux=[split(o, 0, S) for o in outs[:-1]], prop_score=pscore)
        if dn is not None:
            res['dn'] = [split(o, S, N) for o in outs]
        return res

    def cross_cameras(self, r1, r2):
        """Both cameras' slots of one moment (r1, r2 from decode, same batch size) meet; heads again."""
        q = torch.cat([r1['q'], r2['q']], 1)
        pad = torch.cat([r1['pad'], r2['pad']], 1)
        cam = torch.cat([torch.zeros_like(r1['pad'], dtype=torch.long), torch.ones_like(r2['pad'], dtype=torch.long)], 1)
        pl = torch.cat([r1['place'], r2['place']], 1)
        xy = pl[..., :2].detach().float() * XY_S + _xyc(q.device)
        x = q + self.cross_in(q) + place_pos(pl[..., :3].detach())
        x = self.cross(x, cam, xy, pad)
        q = q + x
        S1 = r1['q'].shape[1]
        out = []
        for r, qq in ((r1, q[:, :S1]), (r2, q[:, S1:])):
            o = self.slot_outputs(qq, r['box_logit'] - self.box_delta(self.norm(r['q'])))   # the same box start
            r = dict(r)
            r.update(o, q=qq)
            out.append(r)
        return out

    LONG = 0.1           # the long memory's update rate per frame (MeMOTR: a slow average of the person)

    def next_tracks(self, r, keep, track=None, extra=None, keep_cpu=None):
        """The slots kept (B x S bool) become next frame's tracks: fast memory updated (gated), long memory a slow
        average, last boxes. extra: B x E x D slot vectors added as tracks too (training: false tracks)."""
        B, S, _ = r['q'].shape
        prev = torch.zeros_like(r['q'])
        long_prev = r['q'].detach().clone()
        if track is not None and r['tracks']:
            prev[:, :r['tracks']] = track['h']
            long_prev[:, :r['tracks']] = track['long']
        h = self.track_update(prev, r['q'])
        long = (1 - self.LONG) * long_prev + self.LONG * r['q']
        if keep_cpu is not None:                     # the caller knows who is kept on the host: no device-to-host stall here
            n = int(keep_cpu.sum(1).max()) if keep_cpu.numel() else 0
        else:
            n = int(keep.sum(1).max().item()) if keep.numel() else 0
        H = r['q'].new_zeros(B, n, D); Lg = r['q'].new_zeros(B, n, D)
        Bx = r['box'].new_zeros(B, n, 4); V = torch.zeros(B, n, dtype=torch.bool, device=keep.device)
        idx = []
        for b in range(B):
            if keep_cpu is not None:
                kc = torch.nonzero(keep_cpu[b]).flatten()
                k = kc.pin_memory().to(keep.device, non_blocking=True) if keep.device.type == 'cuda' else kc.to(keep.device)
            else:
                k = kc = torch.nonzero(keep[b]).flatten()
            H[b, :len(k)] = h[b, k]; Lg[b, :len(k)] = long[b, k]
            Bx[b, :len(k)] = r['box'][b, k].detach(); V[b, :len(k)] = True
            idx.append(kc)
        return {'h': H, 'long': Lg, 'box': Bx.clamp(1e-3, 1 - 1e-3), 'valid': V}, idx

    @staticmethod
    def stack_tracks(tracks, dev, dtype):
        """Per-camera track dicts (batch 1 each, maybe None) -> one batch, padded (valid marks the real ones)."""
        n = max([t['h'].shape[1] for t in tracks if t is not None] + [0])
        if n == 0:
            return None
        out = {'h': torch.zeros(len(tracks), n, D, device=dev, dtype=dtype), 'long': torch.zeros(len(tracks), n, D, device=dev, dtype=dtype),
               'box': torch.full((len(tracks), n, 4), 0.5, device=dev), 'valid': torch.zeros(len(tracks), n, dtype=torch.bool, device=dev)}
        for i, t in enumerate(tracks):
            if t is None:
                continue
            k = t['h'].shape[1]
            for key in out:
                out[key][i, :k] = t[key][0].to(out[key].dtype)
        return out

    def pooled(self, m, r, weights):
        """Identity and zone of each slot, pooled under weights (B x S x h x w, 0..1, stride 4)."""
        q = self.norm(r['q'])
        res = {}
        for name, mp in (('reid', m['reid_map']), ('zone', m['zone_map'])):
            w = F.interpolate(weights.float(), size=mp.shape[-2:], mode='bilinear', align_corners=False)
            w = w / w.flatten(2).sum(-1).clamp_min(1e-3)[..., None, None]
            res[name] = torch.einsum('bshw,bchw->bsc', w.to(mp.dtype), mp)
        v = self.reid_out(torch.cat([q, res['reid'].to(q.dtype)], -1))
        zone = self.zone_out(torch.cat([q, res['zone'].to(q.dtype), r['box'].to(q.dtype)], -1))
        return {'ident': v, 'zone': zone}

    @staticmethod
    def full_masks(m, r):
        return torch.einsum('bsc,bchw->bshw', r['mask_vec'], m['pix'])


def build(pretrained=None, backbone=None, pretrained_backbone=True, layers=6, slim=False, rgb_size=None):
    """The model; pretrained: a DEIMv2-S checkpoint path (COCO detector) for the backbone; backbone: a timm name for
    ConvBackbone instead of the ViT; layers: decoder layers; slim: lighter identity heads."""
    model = SlotV2(layers=layers, backbone=backbone, pretrained_backbone=pretrained_backbone, slim=slim, rgb_size=rgb_size)
    if pretrained:
        sd = torch.load(pretrained, map_location='cpu')
        sd = sd.get('model', sd.get('ema', {}).get('module', sd)) if isinstance(sd, dict) else sd
        missing, unexpected = model.body.load_deimv2(sd)
        print('backbone: %d missing, %d unexpected' % (len(missing), len(unexpected)))
    return model


def load(path, which='ema'):
    """A training checkpoint -> the model it holds (the backbone it was built with)."""
    ck = torch.load(path, map_location='cpu')
    arch = ck.get('arch') or {}
    m = build(backbone=ck.get('backbone'), pretrained_backbone=False, layers=arch.get('layers', 6), slim=arch.get('slim', False),
              rgb_size=arch.get('rgb'))
    m.load_state_dict(ck.get(which, ck['model']))
    return m
