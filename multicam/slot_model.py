"""The slot model, v1: one network, one pass over a frame, every person as a slot.

Input: 5 x 608 x 1088 -- RGB and two background heat maps (slot_data.py), normalised by
data/slot_prep/norm.json.

Backbone (default 'deimv2_vit_tiny', the paper's; NON-COMMERCIAL, see deimv2_vit.py):
  * a ViT-Tiny distilled from DINOv3-S (DEIMv2-S), fed RGB only, re-normalised to ImageNet as it was
    trained: its layers 3, 7, 11 give the meaning, resampled to strides 8, 16, 32;
  * a light convolutional branch (DEIMv2's STA) fed all 5 channels -- the heat maps enter here, so the
    ViT's pretrained patch embedding is untouched; its extra input filters start at zero, so on step 0
    the branch is exactly the pretrained one; it gives the fine detail at strides 4, 8, 16, 32;
  * both joined per stride as in DEIMv2 (concatenate, 1x1, BN). Weights: DEIMv2-S's COCO detector.
Any timm backbone by name works too (ConvNeXt, HGNetv2, DINOv3 ViT...), for the comparison.

Heads -- each has its own convolution blocks (residual, with squeeze-excitation) and its own MLP:
  radio   stride 16 -> 512: copies C-RADIOv4-H (PCA-512) of the same frame
  pixel   strides 4 + 8 + 16 -> per-pixel embedding at stride 4; a slot's mask = its vector . embedding
  reid    stride 8 + 16 -> identity map; pooled under the slot's mask, joined with the slot, MLP ->
          256 clothes (copies TransReID MSMT17) + 256 no-clothes (copies CSCI LTCC)
  inout   stride 8 -> small map; pooled under the mask, joined with the slot and its box, MLP ->
          outside / inside / doorway (the owner's /inout answers)
Decoder: 32 slot queries; per layer cross-attention to strides 8/16/32 (masked to the slot's mask of
the previous layer, as Mask2Former: each slot looks at its person), self-attention between slots, FFN;
boxes refined layer by layer. Slots are plain vectors: v2 hands slots from the previous frame back in."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SLOTS = 32
D = 192              # decoder width
P = 128              # neck width
E = 64               # pixel embedding at stride 4 (a slot's mask = its E-vector . this map)
R = 128              # identity map width
INOUT = 3            # outside, inside, doorway
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


# ---------------------------------------------------------------- building blocks
def conv_bn(cin, cout, k=3, act=True):
    layers = [nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout)]
    return nn.Sequential(*layers, nn.GELU()) if act else nn.Sequential(*layers)


class SE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(c, max(8, c // r)), nn.GELU(), nn.Linear(max(8, c // r), c))

    def forward(self, x):
        return x * torch.sigmoid(self.fc(x.mean((2, 3))))[:, :, None, None]


class ResBlock(nn.Module):
    """3x3 conv-BN-GELU, 3x3 conv-BN, squeeze-excitation, residual."""

    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(conv_bn(c, c), conv_bn(c, c, act=False), SE(c))

    def forward(self, x):
        return F.gelu(x + self.body(x))


def tower(cin, width, depth):
    return nn.Sequential(conv_bn(cin, width, 1), *[ResBlock(width) for _ in range(depth)])


def mlp(cin, hidden, cout, depth=3, drop=0.0, norm=False):
    layers = []
    for i in range(depth - 1):
        layers += [nn.Linear(cin if i == 0 else hidden, hidden)] + ([nn.LayerNorm(hidden)] if norm else []) + [nn.GELU()]
        if drop:
            layers.append(nn.Dropout(drop))
    return nn.Sequential(*layers, nn.Linear(hidden if depth > 1 else cin, cout))


# ---------------------------------------------------------------- backbones
class DEIMv2Tiny(nn.Module):
    """ViT-Tiny (RGB) + STA (all channels) -> strides 4, 8, 16, 32. Parameter names follow DEIMv2's
    DINOv3STAs (dinov3, sta, convs, norms), so its checkpoint loads as it is."""

    def __init__(self, in_chans=5, mean=None, std=None, drop_path=0.1, inplanes=16):
        super().__init__()
        from deimv2_vit import VisionTransformer
        self.dinov3 = VisionTransformer(embed_dim=192, num_heads=3, return_layers=[3, 7, 11], drop_path_rate=drop_path)
        c = inplanes
        self.sta = nn.Module()
        self.sta.stem = nn.Sequential(nn.Conv2d(in_chans, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.GELU(), nn.MaxPool2d(3, 2, 1))
        self.sta.conv2 = nn.Sequential(nn.Conv2d(c, 2 * c, 3, 2, 1, bias=False), nn.BatchNorm2d(2 * c))
        self.sta.conv3 = nn.Sequential(nn.GELU(), nn.Conv2d(2 * c, 4 * c, 3, 2, 1, bias=False), nn.BatchNorm2d(4 * c))
        self.sta.conv4 = nn.Sequential(nn.GELU(), nn.Conv2d(4 * c, 4 * c, 3, 2, 1, bias=False), nn.BatchNorm2d(4 * c))
        self.convs = nn.ModuleList([nn.Conv2d(192 + 2 * c, 192, 1, bias=False), nn.Conv2d(192 + 4 * c, 192, 1, bias=False),
                                    nn.Conv2d(192 + 4 * c, 192, 1, bias=False)])
        self.norms = nn.ModuleList([nn.BatchNorm2d(192) for _ in range(3)])
        # our normalisation -> ImageNet's for the ViT
        m = torch.tensor(mean[:3] if mean is not None else IMAGENET[0], dtype=torch.float32)
        s = torch.tensor(std[:3] if std is not None else IMAGENET[1], dtype=torch.float32)
        self.register_buffer('ours', torch.stack([m, s])[:, :, None, None], persistent=False)
        self.register_buffer('imnet', torch.tensor(IMAGENET, dtype=torch.float32)[:, :, None, None], persistent=False)
        self.channels = [c, 192, 192, 192]
        self.in_chans = in_chans

    def load_deimv2(self, state):
        """A DEIMv2-S checkpoint (full detector or backbone only). The stem's extra input filters (heat
        maps) start at zero: on step 0 the branch computes what it was trained to."""
        sd = {k[len('backbone.'):] if k.startswith('backbone.') else k: v for k, v in state.items()}
        sd = {k: v for k, v in sd.items() if k.split('.')[0] in ('dinov3', 'sta', 'convs', 'norms')}
        w = sd.get('sta.stem.0.weight')
        if w is not None and w.shape[1] != self.in_chans:
            ext = torch.zeros(w.shape[0], self.in_chans, *w.shape[2:], dtype=w.dtype)
            ext[:, :w.shape[1]] = w
            sd['sta.stem.0.weight'] = ext
        missing, unexpected = self.load_state_dict(sd, strict=False)
        return missing, unexpected

    def forward(self, x):
        rgb = x[:, :3] * self.ours[1] + self.ours[0]
        rgb = (rgb - self.imnet[0]) / self.imnet[1]
        B, _, H, W = x.shape
        h, w = H // 16, W // 16
        layers = self.dinov3(rgb[:, :, :h * 16, :w * 16])
        sem = []
        for i, (f, _) in enumerate(layers):
            f = f.transpose(1, 2).reshape(B, -1, h, w)
            sem.append(F.interpolate(f, scale_factor=2.0 ** (1 - i), mode='bilinear', align_corners=False))
        c1 = self.sta.stem(x)
        c2 = self.sta.conv2(c1); c3 = self.sta.conv3(c2); c4 = self.sta.conv4(c3)
        out = [c1]
        for k, (s, d) in enumerate(zip(sem, (c2, c3, c4))):
            if s.shape[-2:] != d.shape[-2:]:
                s = F.interpolate(s, size=d.shape[-2:], mode='bilinear', align_corners=False)
            out.append(self.norms[k](self.convs[k](torch.cat([s, d], 1))))
        return out


class TimmBackbone(nn.Module):
    """Any timm backbone by name -> strides 4, 8, 16, 32 (plain ViTs: ViTDet-style pyramid)."""

    def __init__(self, name, pretrained=True, in_chans=5):
        super().__init__()
        import timm
        kw = {'dynamic_img_size': True} if name.startswith('vit') else {}
        red = timm.create_model(name, pretrained=False, features_only=True, **kw).feature_info.reduction()
        self.plain = all(r == red[-1] for r in red)
        idx = [len(red) - 1] if self.plain else [red.index(s) for s in (4, 8, 16, 32)]
        self.net = timm.create_model(name, pretrained=pretrained, features_only=True, in_chans=in_chans, out_indices=idx, **kw)
        ch = self.net.feature_info.channels()
        if self.plain:
            c = ch[-1]
            self.up4 = nn.Sequential(nn.ConvTranspose2d(c, c // 2, 2, 2), nn.GroupNorm(1, c // 2), nn.GELU(), nn.ConvTranspose2d(c // 2, c // 4, 2, 2))
            self.up8 = nn.ConvTranspose2d(c, c // 2, 2, 2)
            self.channels = [c // 4, c // 2, c, c]
        else:
            self.channels = ch

    def forward(self, x):
        f = self.net(x)
        if not self.plain:
            return f
        s16 = f[-1]
        return [self.up4(s16), self.up8(s16), s16, F.max_pool2d(s16, 2)]


# ---------------------------------------------------------------- neck and decoder
class FPN(nn.Module):
    """Top-down and bottom-up (PAN) over strides 8, 16, 32."""

    def __init__(self, chans, width=P):
        super().__init__()
        self.lat = nn.ModuleList([conv_bn(c, width, 1) for c in chans])
        self.td = nn.ModuleList([ResBlock(width) for _ in chans[:-1]])
        self.down = nn.ModuleList([conv_bn(width, width) for _ in chans[:-1]])
        self.bu = nn.ModuleList([ResBlock(width) for _ in chans[:-1]])

    def forward(self, feats):
        x = [l(f) for l, f in zip(self.lat, feats)]
        for i in range(len(x) - 1, 0, -1):
            x[i - 1] = self.td[i - 1](x[i - 1] + F.interpolate(x[i], size=x[i - 1].shape[-2:], mode='nearest'))
        for i in range(len(x) - 1):
            d = self.down[i](F.max_pool2d(x[i], 2))
            if d.shape[-2:] != x[i + 1].shape[-2:]:
                d = F.interpolate(d, size=x[i + 1].shape[-2:], mode='nearest')
            x[i + 1] = self.bu[i](x[i + 1] + d)
        return x


def sine_pos(h, w, dim, device):
    y, x = torch.meshgrid(torch.linspace(0, 1, h, device=device), torch.linspace(0, 1, w, device=device), indexing='ij')
    freq = torch.exp(torch.arange(dim // 4, device=device) * (-math.log(1000.0) / (dim // 4)))
    px, py = x[..., None] * freq * 2 * math.pi, y[..., None] * freq * 2 * math.pi
    return torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], -1).reshape(h * w, dim)


class DecoderLayer(nn.Module):
    def __init__(self, d=D, heads=8, drop=0.0):
        super().__init__()
        self.cross = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.self_ = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.heads = heads

    def forward(self, q, qpos, mem, mpos, attn_mask=None, self_mask=None):
        q = self.n1(q + self.cross(q + qpos, mem + mpos, mem, attn_mask=attn_mask, need_weights=False)[0])
        q = self.n2(q + self.self_(q + qpos, q + qpos, q, attn_mask=self_mask, need_weights=False)[0])
        return self.n3(q + self.ff(q))


def inverse_sigmoid(x, eps=1e-4):
    x = x.clamp(eps, 1 - eps)
    return torch.log(x / (1 - x))


# ---------------------------------------------------------------- the model
class SlotModel(nn.Module):
    def __init__(self, backbone='deimv2_vit_tiny', pretrained=True, in_chans=5, slots=SLOTS, layers=4,
                 radio_dim=512, reid_dim=256, teacher_dims=(3840, 1024), mean=None, std=None, masked_attention=True):
        super().__init__()
        if backbone == 'deimv2_vit_tiny':
            self.body = DEIMv2Tiny(in_chans, mean, std)
        else:
            self.body = TimmBackbone(backbone, pretrained, in_chans)
        c4, c8, c16, c32 = self.body.channels
        self.fpn = FPN([c8, c16, c32])
        # radio head: own blocks at stride 16
        self.radio = nn.Sequential(tower(P, 192, 3), nn.Conv2d(192, radio_dim, 1))
        # pixel head: the heavy blocks at strides 16 and 8; at stride 4 only the fusion with the fine
        # detail and one light convolution (as Mask2Former): stride 4 is 4x the pixels of stride 8
        self.pix16 = tower(P, P, 1)
        self.pix8 = tower(P, P, 2)
        self.lat4 = conv_bn(c4, E, 1)
        self.up8 = conv_bn(P, E, 1)
        self.pix4 = nn.Sequential(conv_bn(E, E, 3), nn.Conv2d(E, E, 1))
        # identity head: strides 16 + 8
        self.reid16 = tower(P, R, 2)
        self.reid8 = tower(P + R, R, 2)
        # place head: stride 8
        self.inout_map = tower(P, 64, 2)
        # decoder
        self.proj = nn.ModuleList([nn.Linear(P, D) for _ in range(3)])
        self.level = nn.Parameter(torch.zeros(3, D))
        self.query = nn.Parameter(torch.randn(slots, D) * 0.02)
        self.qpos = nn.Parameter(torch.randn(slots, D) * 0.02)
        self.box0 = nn.Parameter(torch.zeros(slots, 4))            # learned starting boxes, refined by every layer
        # denoising (DN-DETR / DINO), training only: a spoilt true box -> a hinted slot
        self.dn_content = mlp(4, D, D, depth=2)
        self.dn_pos = mlp(4, D, D, depth=2)
        self.slots = slots
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(layers)])
        self.norm = nn.LayerNorm(D)
        self.masked_attention = masked_attention
        # per-slot outputs (shared by all layers, as in Mask2Former / DINO)
        self.obj = mlp(D, D, 1, depth=2)
        self.box_delta = mlp(D, D, 4)
        self.mask_embed = mlp(D, D, E)
        self.reid_out = mlp(D + R, 512, 2 * reid_dim, depth=3, drop=0.1, norm=True)
        self.inout_out = mlp(D + 64 + 4, 128, INOUT, depth=3, drop=0.1, norm=True)
        # training only: the student's 256 -> the teachers' spaces
        self.to_teacher = nn.ModuleList([mlp(reid_dim, 512, t, depth=2) for t in teacher_dims])
        self.reid_dim = reid_dim
        with torch.no_grad():
            self.box0.copy_(inverse_sigmoid(torch.rand(slots, 4) * torch.tensor([1.0, 1.0, 0.1, 0.4]) + torch.tensor([0.0, 0.0, 0.02, 0.1])))
            self.obj[-1].bias.fill_(-4.6)                          # start at p = 0.01: focal-style prior

    def slot_outputs(self, q, box_logit):
        """Per-slot outputs without the full mask: its vector (mask_vec) is dotted with the pixel
        embedding only where it is needed -- sampled points while training, the whole map when predicting."""
        qn = self.norm(q)
        box_logit = box_logit + self.box_delta(qn)
        return {'obj': self.obj(qn).squeeze(-1), 'box': box_logit.sigmoid(), 'box_logit': box_logit, 'mask_vec': self.mask_embed(qn)}

    def attn_mask(self, mask_vec, pix_levels):
        """Mask2Former: a slot attends only where its current mask says its person is, the mask taken at
        each memory level's own size (strides 8, 16, 32 -- cheap); a slot with an empty mask attends
        everywhere."""
        m = torch.cat([torch.einsum('bsc,bchw->bshw', mask_vec.detach(), p).flatten(2) for p in pix_levels], 2) < 0
        m[m.all(-1)] = False
        # an additive mask with a large finite penalty: a boolean one gives NaN in PyTorch's fast attention path
        return (m.float() * -1e4).repeat_interleave(self.layers[0].heads, 0)

    @staticmethod
    def full_masks(out):
        """Mask logits of every slot on the whole stride-4 map (B x S x 152 x 272), for predicting."""
        return torch.einsum('bsc,bchw->bshw', out['mask_vec'], out['pix'])

    def forward(self, x, aux=True, slots_in=None, radio=True, dn=None):
        """x: B x 5 x H x W (normalised). Returns the final slot outputs (mask_vec, not the masks: see
        full_masks), the stride-4 pixel embedding, the stride-16 radio map, the stride-8 identity and
        place maps and, with aux, every earlier layer's slot outputs.
        dn (training only): {'box': B x Nd x 4 spoilt boxes (cx cy w h, 0..1), 'group': Nd group ids};
        the hinted slots run beside the 32 ordinary ones, which cannot see them, and come back per layer
        in out['dn']."""
        f4, f8, f16, f32 = self.body(x)
        p8, p16, p32 = self.fpn([f8, f16, f32])
        radio = self.radio(p16) if radio else None                 # a training target only: skipped when predicting
        pix8 = self.pix8(p8 + F.interpolate(self.pix16(p16), size=p8.shape[-2:], mode='bilinear', align_corners=False))
        pix = self.pix4(self.lat4(f4) + F.interpolate(self.up8(pix8), size=f4.shape[-2:], mode='bilinear', align_corners=False))
        r16 = self.reid16(p16)
        reid_map = self.reid8(torch.cat([p8, F.interpolate(r16, size=p8.shape[-2:], mode='bilinear', align_corners=False)], 1))
        mem, mpos, sizes = [], [], []
        for i, p in enumerate((p8, p16, p32)):
            b, c, h, w = p.shape
            mem.append(self.proj[i](p.flatten(2).transpose(1, 2)) + self.level[i])
            mpos.append(sine_pos(h, w, D, p.device).to(p.dtype))
            sizes.append((h, w))
        mem, mpos = torch.cat(mem, 1), torch.cat(mpos, 0)[None]
        pix_levels = [F.adaptive_avg_pool2d(pix.detach(), s) for s in sizes] if self.masked_attention else None
        B = x.shape[0]
        S = self.slots if slots_in is None else slots_in.shape[1]
        q = self.query[None].expand(B, -1, -1) if slots_in is None else slots_in
        qpos = self.qpos[None].expand(B, -1, -1)
        box_logit = self.box0[None].expand(B, -1, -1)
        self_mask = None
        if dn is not None:
            db = dn['box'].to(q.dtype)
            q = torch.cat([q, self.dn_content(db)], 1)
            qpos = torch.cat([qpos, self.dn_pos(db)], 1)
            box_logit = torch.cat([box_logit, inverse_sigmoid(dn['box'].float()).to(box_logit.dtype)], 1)
            T = q.shape[1]
            g = torch.full((T,), -1, device=x.device, dtype=torch.long)
            g[S:] = dn['group'].to(x.device)
            block = torch.zeros(T, T, dtype=torch.bool, device=x.device)
            block[:S, S:] = True                                    # the ordinary slots never see the hints
            block[S:, S:] = g[S:, None] != g[None, S:]              # hint groups do not see each other
            self_mask = (block.float() * -1e4).to(q.dtype)[None].expand(B * self.layers[0].heads, -1, -1)
        cur = self.slot_outputs(q, box_logit)
        outs = []
        for layer in self.layers:
            am = self.attn_mask(cur['mask_vec'], pix_levels).to(q.dtype) if self.masked_attention else None
            q = layer(q, qpos, mem, mpos, am, self_mask)
            cur = self.slot_outputs(q, cur['box_logit'].detach())
            outs.append(cur)
        split = lambda o, a, b: {k: v[:, a:b] for k, v in o.items()}
        main = [split(o, 0, S) for o in outs]
        out = dict(main[-1])
        out.update(slots=q[:, :S], pix=pix, radio=radio, aux=main[:-1] if aux else [], reid_map=reid_map, inout_map=self.inout_map(p8))
        if dn is not None:
            out['dn'] = [split(o, S, q.shape[1]) for o in outs]
        return out

    def pooled(self, out, weights):
        """Identity and place of each slot, pooled under `weights` (B x S x h x w, 0..1 at stride 4:
        the slot's own mask, or the draft's mask while training)."""
        q = self.norm(out['slots'])
        res = {}
        for name, m in (('reid', out['reid_map']), ('inout', out['inout_map'])):
            w = F.interpolate(weights.float(), size=m.shape[-2:], mode='bilinear', align_corners=False)
            w = w / w.flatten(2).sum(-1).clamp_min(1e-3)[..., None, None]
            res[name] = torch.einsum('bshw,bchw->bsc', w.to(m.dtype), m)
        v = self.reid_out(torch.cat([q, res['reid'].to(q.dtype)], -1))
        cloth, shape = v[..., :self.reid_dim], v[..., self.reid_dim:]
        place = self.inout_out(torch.cat([q, res['inout'].to(q.dtype), out['box'].to(q.dtype)], -1))
        return {'cloth': cloth, 'shape': shape, 'inout': place}
