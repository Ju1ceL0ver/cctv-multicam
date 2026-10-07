"""SAM-micro (07.10.2026): SAM 3.1's way of working -- a detector of people on every frame and a multiplexed memory
tracker -- built small for 1280 x 720, taught end to end on SAM 3.1's own results (masks, boxes and the people's
numbers over time). Nothing is copied from SAM's insides: only what SAM 3.1 gave on the frames is the target.

model (one frame 1280 x 736, stride 4/8/16):
  encoder   our MobileNetV4-M student (it learned SAM 3.1's features; its fusion of strides 8/16/32 -> 1024 at 16),
            run on the frame as it is -> f16 (256) + thin maps f8 (64), f4 (32); a 2-D sine position code
  detector  ENC layers over the f16 tokens with cross-attention to a learned 'person' prompt (no text), DEC layers of
            Q queries (DETR), heads: score, box (cxcywh), mask = query . pixel map (stride 4)
  tracker   up to SLOTS people at once (SAM 3.1's multiplex): memory of the last MEM frames = the frame's f16 fused
            with the slots' masks (one memory per frame for all people); memory attention (MEMA layers) on the
            current f16; slot tokens (an id embedding + the slot's pointer from its last frame) through a two-way
            decoder -> each slot's mask (stride 4) and 'is visible' logit, and its new pointer
training on clips of CLIP frames (one SAM 3.1 session, the same person = the same number): the detector on every
  frame (Hungarian matching: score, L1, GIoU; focal + L1 + GIoU + BCE + Dice); the tracker from the first frame on --
  a person enters a slot on the first frame SAM 3.1 shows them (prompted with SAM's mask, as SAM's detector adds a
  new object), every later frame: the slot's mask (BCE + Dice) and visibility (BCE); the memory of a frame holds the
  tracker's own masks (detached) -- as at work.

usage (venv_sam3, the card): micro_sam.py train OUT [STOP HH:MM] [--init CKPT]
                             micro_sam.py eval CKPT  (against SAM 3.1 on 23.09, as sam31_lite_eval; + ms a frame)"""
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
W, H, HP = 1280, 720, 736                  # the frame, padded to /32
D = 256
SLOTS, MEM = 16, 6
CLIP = int(os.environ.get('RA_MS_CLIP', '4'))
STRIDE_TICKS = 2
STUDENT = ROOT / 'runs' / 's31micro_door' / 'last.pt'
CHECK_MIN = 60


# ------------------------------------------------------------------ parts

class MHA(nn.Module):
    def __init__(self, d=D, h=8):
        super().__init__()
        self.h = h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, q, k, v):
        b, n, d = q.shape
        sp = lambda x: x.reshape(b, -1, self.h, d // self.h).transpose(1, 2)
        y = F.scaled_dot_product_attention(sp(self.q(q)), sp(self.k(k)), sp(self.v(v)))
        return self.o(y.transpose(1, 2).reshape(b, n, d))


class Layer(nn.Module):
    def __init__(self, cross=True, ffn=1024):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.sa = MHA()
        self.ca = MHA() if cross else None
        self.ff = nn.Sequential(nn.Linear(D, ffn), nn.GELU(), nn.Linear(ffn, D))

    def forward(self, x, pos=None, mem=None, mpos=None):
        y = self.n1(x)
        qk = y if pos is None else y + pos
        x = x + self.sa(qk, qk, y)
        if self.ca is not None and mem is not None:
            y = self.n2(x)
            x = x + self.ca(y if pos is None else y + pos, mem if mpos is None else mem + mpos, mem)
        return x + self.ff(self.n3(x))


class TwoWay(nn.Module):
    def __init__(self):
        super().__init__()
        self.sa, self.t2i, self.i2t = MHA(), MHA(), MHA()
        self.mlp = nn.Sequential(nn.Linear(D, 1024), nn.GELU(), nn.Linear(1024, D))
        self.ns = nn.ModuleList(nn.LayerNorm(D) for _ in range(4))

    def forward(self, t, img, pos):
        t = self.ns[0](t + self.sa(t, t, t))
        t = self.ns[1](t + self.t2i(t, img + pos, img))
        t = self.ns[2](t + self.mlp(t))
        img = self.ns[3](img + self.i2t(img + pos, t, t))
        return t, img


def sine_pos(h, w, d=D, device='cpu'):
    """2-D sine position code (any grid size)."""
    y, x = torch.meshgrid(torch.arange(h, device=device, dtype=torch.float32), torch.arange(w, device=device, dtype=torch.float32), indexing='ij')
    n = d // 4
    f = 1.0 / (10000 ** (torch.arange(n, device=device, dtype=torch.float32) / n))
    px, py = x.flatten()[:, None] * f, y.flatten()[:, None] * f
    return torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], 1)[None]          # 1 x hw x d


class MicroSAM(nn.Module):
    def __init__(self, enc=3, dec=4, queries=50, mema=2, student=STUDENT):
        super().__init__()
        import sam31_distill as SD
        ck = torch.load(student, map_location='cpu', weights_only=False) if student and Path(student).exists() else None
        arch = ck['arch'] if ck else {'name': 'mobilenetv4_conv_medium', 'inner': 384, 'blocks': 4, 'size': 1152}
        self.enc_net = SD.Student(**arch, pretrained=ck is None)
        if ck:
            self.enc_net.load_state_dict(ck['ema'] if ck.get('ema') else ck['model'])
        ch = self.enc_net.body.feature_info.channels()
        self.n16 = nn.Sequential(nn.Conv2d(1024, D, 1), nn.GroupNorm(32, D))
        self.n8 = nn.Sequential(nn.Conv2d(ch[0], 64, 1), nn.GroupNorm(8, 64))
        self.n4 = nn.Sequential(nn.Conv2d(ch[0], 32, 3, padding=1), nn.GroupNorm(8, 32))          # stride-8 map, upsampled
        # detector
        self.prompt = nn.Parameter(torch.randn(1, 4, D) * 0.02)
        self.denc = nn.ModuleList(Layer() for _ in range(enc))
        self.ddec = nn.ModuleList(Layer() for _ in range(dec))
        self.query = nn.Parameter(torch.randn(1, queries, D) * 0.02)
        self.qpos = nn.Parameter(torch.randn(1, queries, D) * 0.02)
        self.pix = nn.ModuleList([nn.Conv2d(D + 64, 64, 3, padding=1), nn.Conv2d(64 + 32, 64, 3, padding=1)])
        self.q_mask = nn.Linear(D, 64)
        self.q_box = nn.Sequential(nn.Linear(D, D), nn.ReLU(), nn.Linear(D, 4))
        self.q_score = nn.Linear(D, 1)
        # tracker
        self.mema = nn.ModuleList(Layer() for _ in range(mema))
        self.t_pos = nn.Parameter(torch.randn(MEM, 1, D) * 0.02)                                 # which past frame
        self.slot_id = nn.Parameter(torch.randn(1, SLOTS, D) * 0.02)
        self.ptr_in = nn.Linear(D, D)
        self.obj_in = nn.Sequential(nn.Linear(2 * D, D), nn.GELU(), nn.Linear(D, D))   # 08.10: the slot's own person
        self.twoway = nn.ModuleList(TwoWay() for _ in range(2))
        self.up = nn.Sequential(nn.ConvTranspose2d(D, 64, 2, 2), nn.GELU(), nn.ConvTranspose2d(64, 32, 2, 2))
        self.t_mask = nn.Linear(D, 64)          # 08.10: slot . the shared stride-4 pixel map (was a coarse x4 deconv)
        self.t_vis = nn.Linear(D, 1)
        self.mem_down = nn.Sequential(nn.Conv2d(SLOTS, 64, 3, 2, 1), nn.GELU(), nn.Conv2d(64, D, 3, 2, 1))
        self.mem_fuse = nn.Sequential(nn.Conv2d(D, D, 3, padding=1), nn.GELU(), nn.Conv2d(D, D, 1))

    # ---- encoder
    def features(self, x):
        """x: B x 3 x 736 x 1280, RGB 0..1 -> f16 (B, 256, 46, 80), f8 (B, 64, 92, 160), f4 (B, 32, 184, 320)."""
        e = self.enc_net
        x = (x - e.mean) / e.std
        f8, f16, f32 = e.body(x)
        y = e.l16(f16) + F.interpolate(e.l32(f32), size=f16.shape[-2:], mode='bilinear', align_corners=False) \
            + F.adaptive_avg_pool2d(e.l8(f8), f16.shape[-2:])
        y = e.out(e.blocks(y))
        g16 = self.n16(y)
        g8 = self.n8(f8)
        g4 = F.interpolate(self.n4(f8), scale_factor=2, mode='bilinear', align_corners=False)
        return g16, g8, g4

    # ---- detector
    def detect(self, g16, g8, g4):
        b, d, h, w = g16.shape
        pos = sine_pos(h, w, device=g16.device).to(g16.dtype)
        x = g16.flatten(2).transpose(1, 2)
        pr = self.prompt.expand(b, -1, -1)
        for L in self.denc:
            x = L(x, pos, pr)
        q = self.query.expand(b, -1, -1)
        qp = self.qpos.expand(b, -1, -1)
        for L in self.ddec:
            q = L(q, qp, x, pos)
        p = self.pixels(x, g8, g4, h, w)
        return {'pred_logits': self.q_score(q), 'pred_boxes': self.q_box(q).sigmoid(),
                'pred_masks': torch.einsum('bqc,bchw->bqhw', self.q_mask(q), p)}

    def pixels(self, x, g8, g4, h, w):
        """tokens (B x hw x D) + the thin stride-8/4 maps -> the stride-4 pixel map (B x 64 x 184 x 320)"""
        b, d = x.shape[0], x.shape[2]
        p = x.transpose(1, 2).reshape(b, d, h, w)
        p = F.gelu(self.pix[0](torch.cat([F.interpolate(p, size=g8.shape[-2:], mode='bilinear', align_corners=False), g8], 1)))
        return self.pix[1](torch.cat([F.interpolate(p, size=g4.shape[-2:], mode='bilinear', align_corners=False), g4], 1))

    # ---- tracker
    def memory(self, g16, slot_masks):
        """one frame's memory: f16 fused with the slots' masks (B x SLOTS x h4 x w4 probabilities) -> B x hw x D"""
        m = self.mem_down(slot_masks)
        m = F.interpolate(m, size=g16.shape[-2:], mode='bilinear', align_corners=False)
        return self.mem_fuse(m + g16).flatten(2).transpose(1, 2)

    def track(self, g16, g8, g4, mems, ptr, prev=None):
        """mems: [B x hw x D] of up to MEM past frames (oldest first); ptr: B x SLOTS x D (the slots' pointers)
        -> slot masks B x SLOTS x h4 x w4 (logits), visibility B x SLOTS, new pointers."""
        b, d, h, w = g16.shape
        pos = sine_pos(h, w, device=g16.device).to(g16.dtype)
        x = g16.flatten(2).transpose(1, 2)
        mem = torch.cat(mems[-MEM:], 1)
        tp = torch.cat([self.t_pos[MEM - len(mems[-MEM:]) + i].expand(b, h * w, d) for i in range(len(mems[-MEM:]))], 1)
        mpos = pos.repeat(1, len(mems[-MEM:]), 1) + tp
        for L in self.mema:
            x = L(x, pos, mem, mpos)
        t = self.slot_id.expand(b, -1, -1) + self.ptr_in(ptr)
        if prev is not None:      # each slot: the mean features and place under its last mask (SAM's object pointer role)
            wm = F.interpolate(prev.to(g16.dtype), size=(h, w), mode='area').flatten(2)            # B x S x hw
            wm = wm / wm.sum(-1, keepdim=True).clamp(min=1e-4)
            feat = torch.einsum('bsn,bnd->bsd', wm, g16.flatten(2).transpose(1, 2))
            where = torch.einsum('bsn,bnd->bsd', wm, pos.expand(b, -1, -1))
            t = t + self.obj_in(torch.cat([feat, where], -1))
        img = x
        for L in self.twoway:
            t, img = L(t, img, pos)
        up = self.pixels(img, g8, g4, h, w)
        return torch.einsum('bsc,bchw->bshw', self.t_mask(t), up), self.t_vis(t)[..., 0], t


# ------------------------------------------------------------------ data: clips of SAM 3.1 sessions at 1280 x 720

class Clips:
    def __init__(self, seed=0, test=False):
        import sam31_e2e_det as E
        import sam31_reid as R
        self.rng = random.Random(seed)
        self.items = []
        dirs = E.sources() if not test else sorted(p for p in (ROOT / 'data' / 'sam31_seg').glob('20260923*/cam*'))
        for d in dirs:
            if not (d / 'chunks.npz').exists():
                continue
            try:
                M = R.Masks(d / 'chunks.npz', mmap=True)
            except Exception:
                continue
            rows = M.rows
            sess = {}
            for i in range(len(rows)):
                sess.setdefault(int(rows[i, 0]), {}).setdefault(int(rows[i, 1]), []).append(i)
            starts = [(s, k) for s, by in sess.items() for k in sorted(by)
                      if all((k + j * STRIDE_TICKS) in by for j in range(CLIP))]
            if starts:
                self.items.append((d, M, sess, starts))
        self.caps = {}

    def frame(self, d, k):
        import cv2
        cap = self.caps.get(d)
        if cap is None:
            if len(self.caps) > 6:
                self.caps.pop(next(iter(self.caps))).release()
            cap = self.caps[d] = cv2.VideoCapture(str(d / 'video.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k)
        ok, f = cap.read()
        return f if ok else None

    def sample(self):
        """frames T x 736 x 1280 x 3 (RGB uint8, padded), masks T x N x 184 x 320 (stride 4, bool), boxes T x N x 4
        (cxcywh 0..1 of 1280 x 720), visible T x N."""
        import cv2
        while True:
            d, M, sess, starts = self.rng.choice(self.items)
            s, k0 = self.rng.choice(starts)
            ticks = [k0 + j * STRIDE_TICKS for j in range(CLIP)]
            fr = [self.frame(d, k) for k in ticks]
            if any(f is None for f in fr):
                continue
            Hs, Ws = fr[0].shape[:2]
            ids = sorted({int(M.rows[r, 2]) for k in ticks for r in sess[s][k]})[:SLOTS]
            pos = {o: i for i, o in enumerate(ids)}
            masks = np.zeros((CLIP, len(ids), 184, 320), bool)
            boxes = np.zeros((CLIP, len(ids), 4), np.float32)
            vis = np.zeros((CLIP, len(ids)), bool)
            for t, k in enumerate(ticks):
                for r in sess[s][k]:
                    o = int(M.rows[r, 2])
                    if o not in pos:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in M.rows[r, 4:8]]
                    m = np.zeros((Hs, Ws), np.uint8)
                    c = M.crop(r)
                    m[int(y1):int(y1) + c.shape[0], int(x1):int(x1) + c.shape[1]] = c[:Hs - int(y1), :Ws - int(x1)]
                    small = cv2.resize(m, (320, 180), interpolation=cv2.INTER_AREA) > 0
                    if not small.any():
                        continue
                    masks[t, pos[o], :180] = small
                    boxes[t, pos[o]] = [(x1 + x2) / 2 / Ws, (y1 + y2) / 2 / Hs, (x2 - x1) / Ws, (y2 - y1) / Hs]
                    vis[t, pos[o]] = True
            if not vis[0].any():
                continue
            imgs = np.zeros((CLIP, HP, W, 3), np.uint8)
            for t, f in enumerate(fr):
                imgs[t, :H] = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            return imgs, masks, boxes, vis


# ------------------------------------------------------------------ losses

def det_loss(out, boxes, masks):
    import sam31_e2e_det as E
    vis = boxes[:, 2] > 0
    b, m = boxes[vis], masks[vis]
    out1 = {'pred_logits': out['pred_logits'][:1], 'pred_boxes': out['pred_boxes'][:1], 'pred_masks': out['pred_masks'][:1]}
    loss, st = E.losses(out1, b.cpu().numpy(), m.cpu().numpy())
    return loss, st


def run_clip(model, imgs, masks, boxes, vis, dev, train=True):
    """the detector on every frame + the tracker through the clip -> (loss, stats)"""
    T, N = masks.shape[:2]
    x = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255
    mk = torch.from_numpy(masks).to(dev).float()
    bx = torch.from_numpy(boxes).to(dev)
    vs = torch.from_numpy(vis).to(dev)
    g16, g8, g4 = model.features(x)
    loss, st = 0.0, {}
    for t in range(T):
        out = model.detect(g16[t:t + 1], g8[t:t + 1], g4[t:t + 1])
        l, s = det_loss(out, bx[t], mk[t])
        loss = loss + l / T
        for k, v in s.items():
            st['det_' + k] = st.get('det_' + k, 0) + v / T
    # the tracker: slot i = person i of the clip; a person enters on the first frame it is visible
    slot_m = torch.zeros(1, SLOTS, 184, 320, device=dev)
    ptr = torch.zeros(1, SLOTS, D, device=dev)
    entered = torch.zeros(N, dtype=torch.bool, device=dev)
    mems = []
    lt, n_t = 0.0, 0
    for t in range(T):
        new = vs[t] & ~entered
        if t > 0:
            pm, pv, ptr_new = model.track(g16[t:t + 1], g8[t:t + 1], g4[t:t + 1], mems, ptr, slot_m)
            known = entered.clone()
            if known.any():
                idx = torch.nonzero(known)[:, 0]
                gm = mk[t, idx]
                pmk = pm[0, idx]
                bce = F.binary_cross_entropy_with_logits(pmk, gm)
                p = pmk.sigmoid().flatten(1); q = gm.flatten(1)
                dice = (1 - (2 * (p * q).sum(1) + 1) / (p.sum(1) + q.sum(1) + 1))
                vt = vs[t, idx].float()
                dice = (dice * vt).sum() / vt.sum().clamp(min=1)              # Dice only where the person is seen
                lv = F.binary_cross_entropy_with_logits(pv[0, idx], vt)
                lt = lt + 20 * bce + dice + lv
                n_t += 1
                st['trk_bce'] = st.get('trk_bce', 0) + float(bce.detach())
                st['trk_dice'] = st.get('trk_dice', 0) + float(dice.detach())
                st['trk_vis'] = st.get('trk_vis', 0) + float(lv.detach())
            ptr = ptr_new
            slot_m = torch.zeros_like(slot_m)
            slot_m[0, :N] = pm[0, :N].sigmoid().detach() * entered[:, None, None].float()
        if new.any():                                         # SAM's detector adds them: prompted with SAM's own mask
            slot_m = slot_m.clone()
            slot_m[0, :N][new] = mk[t][new]
            entered = entered | new
        mems.append(model.memory(g16[t:t + 1], slot_m))
    if n_t:
        loss = loss + lt / n_t
        for k in ('trk_bce', 'trk_dice', 'trk_vis'):
            st[k] = st[k] / n_t
    return loss, st


# ------------------------------------------------------------------ train

def train(out, stop='09:30', init=None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    dev = 'cuda'
    run = ROOT / 'runs' / out
    run.mkdir(parents=True, exist_ok=True)
    model = MicroSAM().to(dev)
    if init:
        sd = torch.load(init, map_location='cpu', weights_only=False)['model']
        own = model.state_dict()
        model.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
    enc = [p for n, p in model.named_parameters() if n.startswith('enc_net.')]
    rest = [p for n, p in model.named_parameters() if not n.startswith('enc_net.')]
    opt = torch.optim.AdamW([{'params': enc, 'lr': 2e-5, 'base': 2e-5}, {'params': rest, 'lr': 2e-4, 'base': 2e-4}], weight_decay=1e-4)
    data = Clips()
    say(run, 'старт: SAM-micro 1280x720 (кодировщик %.1f М, остальное %.1f М), клипы по %d кадра из %d окон, стоп %s'
        % (sum(p.numel() for p in enc) / 1e6, sum(p.numel() for p in rest) / 1e6, CLIP, len(data.items), stop))
    hh, mm = map(int, stop.split(':'))
    t_end = time.time() + ((hh * 60 + mm) - (time.localtime().tm_hour * 60 + time.localtime().tm_min)) % 1440 * 60
    t0, t_check, step, run_st, WARM = time.time(), time.time(), 0, {}, 300
    smoke = int(os.environ.get('RA_MS_SMOKE', '0'))
    while time.time() < t_end:
        imgs, masks, boxes, vis = data.sample()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, st = run_clip(model, imgs, masks, boxes, vis, dev)
        if not torch.isfinite(loss):
            continue
        frac = min(1.0, (time.time() - t0) / max(1.0, t_end - t0))
        f = min(1.0, (step + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g['lr'] = g['base'] * f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        for k, v in st.items():
            run_st[k] = v if k not in run_st else 0.98 * run_st[k] + 0.02 * v
        if smoke:
            print('smoke', step, json.dumps({k: round(v, 3) for k, v in st.items()}), 'gpu %.1f GB' % (torch.cuda.max_memory_allocated() / 1e9),
                  '%.1f s' % (time.time() - t0), flush=True)
            if step >= smoke:
                os._exit(0)
            continue
        if step % 50 == 0:
            with open(run / 'log.jsonl', 'a') as fo:
                fo.write(json.dumps({'step': step, 'updated': time.strftime('%H:%M:%S'), 'clips_per_s': round(step / (time.time() - t0), 3),
                                     **{k: round(v, 4) for k, v in run_st.items()}}) + '\n')
        if step % 500 == 0:
            torch.save({'model': model.state_dict(), 'step': step}, run / 'last.pt')
        if time.time() - t_check > CHECK_MIN * 60:
            torch.save({'model': model.state_dict(), 'step': step}, run / 'last.pt')
            say(run, 'шаг %d: потери %s; против SAM 3.1 на 23.09: %s' % (step, json.dumps({k: round(v, 3) for k, v in run_st.items()}),
                                                                       json.dumps(evaluate(run / 'last.pt', model=model))))
            model.train()
            t_check = time.time()
    torch.save({'model': model.state_dict(), 'step': step}, run / 'last.pt')
    say(run, 'конец, шаг %d: %s' % (step, json.dumps(evaluate(run / 'last.pt', model=model))))
    os._exit(0)


def say(run, text):
    line = time.strftime('%d.%m %H:%M  ') + text
    with open(run / 'progress.md', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode(), flush=True)


# ------------------------------------------------------------------ work and the check against SAM 3.1

class Runner:
    """At work: the detector every frame, the tracker on the slots; a detection not on a tracked person
    (mask IoU < 0.3) and scoring >= NEW opens a slot (prompted with its own mask, as SAM 3.1); a slot unseen for
    LOST frames is freed. -> per frame [(id, score, mask at the given size)]."""
    NEW = float(os.environ.get('RA_MS_NEW', '0.25'))     # 07.10: early in training the scores run 0.2-0.4
    SEEN, LOST = float(os.environ.get('RA_MS_SEEN', '0.5')), 25

    def __init__(self, model, dev):
        self.m, self.dev = model.eval(), dev
        self.mems, self.ptr = [], torch.zeros(1, SLOTS, D, device=dev)
        self.prev = torch.zeros(1, SLOTS, 184, 320, device=dev)
        self.ids, self.miss, self.next = [None] * SLOTS, [0] * SLOTS, 1

    @torch.no_grad()
    def step(self, rgb):
        import cv2
        x = np.zeros((HP, W, 3), np.uint8)
        x[:H] = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).to(self.dev).permute(2, 0, 1)[None].float() / 255
        with torch.autocast('cuda', dtype=torch.bfloat16):
            g16, g8, g4 = self.m.features(x)
            det = self.m.detect(g16, g8, g4)
            slot = torch.zeros(1, SLOTS, 184, 320, device=self.dev)
            out = []
            if self.mems and any(self.ids) and os.environ.get('RA_MS_DETONLY') != '1':
                pm, pv, ptr = self.m.track(g16, g8, g4, self.mems, self.ptr, self.prev)
                self.ptr = ptr.float()
                for i in range(SLOTS):
                    if self.ids[i] is None:
                        continue
                    if pv[0, i].sigmoid() >= self.SEEN:
                        mm = pm[0, i].sigmoid()
                        slot[0, i] = mm
                        self.miss[i] = 0
                        out.append((self.ids[i], float(pv[0, i].sigmoid()), (mm[:180] > 0.5).float().cpu().numpy()))
                    else:
                        self.miss[i] += 1
                        if self.miss[i] > self.LOST:
                            self.ids[i] = None
            sc = det['pred_logits'][0, :, 0].sigmoid()
            dm = det['pred_masks'][0].sigmoid()[:, :180] > 0.5
            for qi in torch.argsort(-sc).tolist():
                if sc[qi] < self.NEW:
                    break
                m = dm[qi].float().cpu().numpy()
                if m.sum() < 20 or any(_iou(m, o[2]) >= 0.3 for o in out):
                    continue
                if os.environ.get('RA_MS_DETONLY') == '1':          # the detector alone: one id per detection
                    out.append((self.next, float(sc[qi]), m)); self.next += 1
                    continue
                free = [i for i in range(SLOTS) if self.ids[i] is None]
                if not free:
                    break
                i = free[0]
                self.ids[i], self.miss[i] = self.next, 0
                self.next += 1
                slot[0, i, :180] = torch.from_numpy(m).to(self.dev)
                out.append((self.ids[i], float(sc[qi]), m))
            self.mems.append(self.m.memory(g16, slot))
            seen = slot.flatten(2).sum(-1) > 0                       # a slot unseen now keeps its last mask
            self.prev = torch.where(seen[..., None, None], slot, self.prev)
            self.mems = self.mems[-MEM:]
        return out


def _iou(a, b):
    i = (a * b).sum()
    return i / max(1e-6, a.sum() + b.sum() - i)


def evaluate(ckpt, model=None, n=2, ticks=96):
    """The same stretches, teacher cache and measures as sam31_lite_eval (23.09, never seen); + ms a frame."""
    import cv2
    import sam31_lite_eval as SL
    dev = 'cuda'
    if model is None:
        model = MicroSAM(student=None).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location='cpu', weights_only=False)['model'], strict=False)
    cache = ROOT / 'runs' / 's31micro_a' / 'evals' / 'teacher_cache'
    rep = []
    times = []
    for d, k0, dens in SL.stretches(n, ticks):
        key = cache / ('%s_%s_%d_%d.pkl' % (d.parent.name, d.name, k0, ticks))
        if not key.exists():
            continue
        T = SL.load_res(key)
        cap = cv2.VideoCapture(str(d / 'video.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES, k0)
        run = Runner(model, dev)
        S = {}
        for k in range(ticks):
            ok, f = cap.read()
            if not ok:
                break
            torch.cuda.synchronize(); t0 = time.time()
            res = run.step(f[:, :, ::-1].copy())
            torch.cuda.synchronize(); times.append(time.time() - t0)
            S[k] = [(i, p, cv2.resize(m.astype(np.uint8), (252, 252), interpolation=cv2.INTER_NEAREST).astype(bool)) for i, p, m in res]
        cap.release()
        r = SL.compare(T, S)
        r['teacher_people'] = r.get('teacher_people', sum(len(v) for v in T.values()))
        rep.append(r)
    keys = ('found', 'precision', 'mask_iou', 'idf1')
    w = np.array([r['teacher_people'] for r in rep], float)
    out = {k: round(float(np.average([r[k] for r in rep], weights=w)), 4) for k in keys}
    out['id_switches'] = int(sum(r['id_switches'] for r in rep))
    out['ms_per_frame'] = round(1000 * float(np.median(times[5:])), 1) if len(times) > 5 else None
    return out


if __name__ == '__main__':
    a = sys.argv[1:]
    if a[0] == 'train':
        kw = {}
        if '--init' in a:
            i = a.index('--init'); kw['init'] = a[i + 1]; del a[i:i + 2]
        train(*a[1:], **kw)
    elif a[0] == 'eval':
        print(json.dumps(evaluate(a[1])), flush=True)
