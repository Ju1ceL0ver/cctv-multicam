"""Training of the v2 slot model on the SAM 3.1 windows (v2_prep.py -> v2_data.py).

A step is a few clips; a clip is T moments of one window, both cameras. Frame by frame: the maps, the slots
(proposals + the previous frame's tracks), the cross-camera layer, the losses; the kept slots go on as tracks.
Who is whom: a track keeps the person it was matched to (if that person is hidden or gone now, the track must
say so and not claim anybody); proposals are matched (Hungarian, as v1) to the people no track holds.

Losses: person-ness (IoU-aware), box, mask (points), the same for denoising hints and every decoder layer;
centres and sizes (CenterNet focal), boundaries between touching people; state (seen / hidden / gone);
place (Gaussian NLL with the predicted uncertainty; velocity); zone; identity (the teachers' vectors of the
piece + supervised contrast within a camera: the same person at other moments against the others);
SameHead on pairs of slots of one camera, near and far in time.

Stages: --freeze-epochs with the backbone frozen (the new parts learn first), then everything, the ViT at a
lower rate; EMA of the weights. Clips: mostly T moments 1-3 ticks apart; --far of them two moments up to
5 minutes apart (tracks are not carried then -- they teach identity and SameHead).

usage: train_v2.py --out runs/v2_a [--tags TAG ...] [--heldout TAG ...] [--epochs N] ..."""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

W = {'obj': 2.0, 'l1': 5.0, 'giou': 2.0, 'bce': 5.0, 'dice': 5.0, 'cen': 1.0, 'size': 0.5, 'bnd': 1.0, 'state': 1.0,
     'place': 0.5, 'vel': 0.2, 'zone': 0.5, 'reid': 1.0, 'rel': 1.0, 'supcon': 0.5, 'same': 0.5}
ZONE_WEIGHT = (1.0, 0.6, 3.0)
AUX_POINTS = 3136                 # mask points per person on the helper layers (the final layer: 12544)
SCALER = None                     # torch.amp.GradScaler with --amp fp16 (a T4 has no bf16)


def backward(loss):
    (SCALER.scale(loss) if SCALER is not None else loss).backward()


def allreduce_grads(params, world):
    """Several GPUs (torchrun): the mean of every rank's gradients, one flat all-reduce. A parameter no rank used this
    step keeps no gradient (AdamW then leaves it alone, as on one card)."""
    import torch
    import torch.distributed as dist
    has = torch.tensor([p.grad is not None for p in params], device=params[0].device, dtype=torch.float32)
    dist.all_reduce(has)
    use = [p for p, h in zip(params, has.tolist()) if h > 0]
    if not use:
        return
    grads = [p.grad if p.grad is not None else torch.zeros_like(p) for p in use]
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat)
    flat /= world
    for p, g in zip(use, torch._utils._unflatten_dense_tensors(flat, grads)):
        if p.grad is None:
            p.grad = g
        else:
            p.grad.copy_(g)


def targets_tensor(tg, dev):
    import torch
    t = {k: torch.as_tensor(v).to(dev) for k, v in tg.items()}
    t['masks'] = t['masks'].float()
    return t


# ---------------------------------------------------------------- map losses
def centre_targets(tg, h, w, dev):
    """Gaussian peaks at mask centres (stride 8), size (w, h fractions) at the peak."""
    import torch
    heat = torch.zeros(h, w, device=dev)
    size = torch.zeros(2, h, w, device=dev)
    sw = torch.zeros(h, w, device=dev)
    m = tg['masks']
    if not len(m):
        return heat, size, sw
    small = torch.nn.functional.adaptive_avg_pool2d(m[None], (h, w))[0]
    ys = torch.arange(h, device=dev).float()[:, None]; xs = torch.arange(w, device=dev).float()[None]
    for i in range(len(m)):
        a = small[i]
        tot = a.sum()
        if tot < 1e-3:
            continue
        cy = (a.sum(1) * torch.arange(h, device=dev)).sum() / tot
        cx = (a.sum(0) * torch.arange(w, device=dev)).sum() / tot
        sig = max(1.0, float(math.sqrt(tot.item()) / 3))
        g = torch.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sig * sig))
        heat = torch.maximum(heat, g)
        iy, ix = int(cy.round().clamp(0, h - 1)), int(cx.round().clamp(0, w - 1))
        size[:, iy, ix] = tg['boxes'][i, 2:]
        sw[iy, ix] = 1.0
    return heat, size, sw


def focal_heat(logit, heat):
    """CenterNet's penalty-reduced focal loss."""
    import torch
    p = logit.float().sigmoid().clamp(1e-4, 1 - 1e-4)
    pos = heat.eq(1).float() + (heat > 0.98).float() * (1 - heat.eq(1).float())
    neg = 1 - pos
    lp = -(torch.log(p) * (1 - p) ** 2 * pos).sum()
    ln = -(torch.log(1 - p) * p ** 2 * (1 - heat) ** 4 * neg).sum()
    return (lp + ln) / pos.sum().clamp(min=1)


def boundary_target(masks):
    """Pixels (stride 4) where two different people touch: each mask dilated by 2, covered by two or more."""
    import torch
    import torch.nn.functional as F
    if len(masks) < 2:
        return torch.zeros(masks.shape[-2:], device=masks.device)
    d = F.max_pool2d(masks[:, None], 5, 1, 2)[:, 0]
    return (d.sum(0) >= 2).float()


# ---------------------------------------------------------------- matching
def match_frame(r, b, tg, assign):
    """Slots of image b <-> people. assign: {track slot: person or None (a false track)}. Returns (rows, cols) of
    matched slots/people for the final layer, per track slot its state target (0 seen, 1 hidden, 2 gone) and
    hidden place, and the auxiliary layers' matches: CO-MOT's -- the proposals are matched to every person,
    tracked ones included, so they keep learning to find people the tracks hold."""
    import torch
    import train_slots as TS
    person = tg['person'].tolist()
    col_of = {p: i for i, p in enumerate(person)}
    hidden = {int(p): i for i, p in enumerate(tg['hidden'].tolist())}
    rows, cols, states = [], [], {}
    taken = set()
    for s, p in assign.items():
        if p is not None and p in col_of:
            rows.append(s); cols.append(col_of[p]); states[s] = (0, None); taken.add(col_of[p])
        elif p is not None and p in hidden:
            states[s] = (1, hidden[p])
        else:
            states[s] = (2, None)
    T = r['tracks']
    S = r['obj'].shape[1]
    props = torch.arange(T, S, device=r['obj'].device)
    props = props[~r['pad'][b, T:]]
    sub = {k: v[b][props] for k, v in (('obj', r['obj']), ('box', r['box']), ('mask_vec', r['mask_vec']))}

    def hungarian(cols_):
        if not cols_:
            return [], []
        t = {'boxes': tg['boxes'][cols_], 'masks': tg['masks'][cols_]}
        pr, pc = TS.match(sub['obj'], sub['box'], sub['mask_vec'], r['pix'][b], t)
        return props[pr].tolist(), [cols_[i] for i in pc.tolist()]
    free = [i for i in range(len(person)) if i not in taken]
    fr, fc = hungarian(free)
    ar, ac = hungarian(list(range(len(person))))
    L = lambda x: torch.as_tensor(x, dtype=torch.long)
    return L(rows + fr), L(cols + fc), states, (L(rows + ar), L(cols + ac))


# ---------------------------------------------------------------- one frame
def frame_losses(model, m, r, tgs, matches, states, dn_matches, L, aux_matches=None):
    """Adds this frame's (both cameras') losses into L."""
    import torch
    import torch.nn.functional as F
    import train_slots as TS
    import slot_v2 as V
    dev = m['pix'].device
    B = m['pix'].shape[0]
    aux_matches = aux_matches or matches
    r = dict(r); r['pix'] = m['pix']
    pad = r['pad']
    silence = lambda o: dict(o, obj=o['obj'].masked_fill(pad, -20.0))       # padding slots are nobody: no loss
    r = silence(r)
    add = lambda k, v, n=1.0: L.__setitem__(k, L.get(k, 0.0) + v * n)
    for k, v in TS.set_losses(r, m['pix'], tgs, matches).items():
        add(k, v)
    # the helper layers (earlier decoder layers, denoising hints) only nudge learning: their masks are scored on
    # a half-size map with a quarter of the points -- the final layer keeps the full detail
    pix_half = F.avg_pool2d(m['pix'], 2)
    tg_half = [dict(t, masks=F.avg_pool2d(t['masks'][None], 2)[0] if len(t['masks']) else t['masks'].new_zeros(0, *pix_half.shape[-2:])) for t in tgs]
    points = TS.LOSS_POINTS
    TS.LOSS_POINTS = AUX_POINTS
    try:
        for a in r['aux']:
            for k, v in TS.set_losses(silence(a), pix_half, tg_half, aux_matches).items():
                add(k, v, 1.0 / len(r['aux']))
        if 'dn' in r:
            for a in r['dn']:
                for k, v in TS.set_losses(a, pix_half, tg_half, dn_matches).items():
                    add('dn_' + k, v, 1.0 / len(r['dn']))
    finally:
        TS.LOSS_POINTS = points
    # centres, sizes, boundaries
    h, w = m['cen'].shape[-2:]
    for b in range(B):
        heat, size, sw = centre_targets(tgs[b], h, w, dev)
        add('cen', focal_heat(m['cen'][b, 0], heat))
        if sw.sum() > 0:
            add('size', (F.l1_loss(m['cen'][b, 1:].float().sigmoid(), size, reduction='none') * sw).sum() / sw.sum())
        bt = boundary_target(tgs[b]['masks'])
        add('bnd', F.binary_cross_entropy_with_logits(m['bnd'][b, 0].float(), bt, pos_weight=torch.tensor(10.0, device=dev)))
    # state and place
    st_l, st_t, pl_l = [], [], []
    for b in range(B):
        rows, cols = matches[b]
        tg = tgs[b]
        seen = {int(s): int(c) for s, c in zip(rows.tolist(), cols.tolist())}
        for s, c in seen.items():
            st_l.append(r['state'][b, s]); st_t.append(0)
            pw = float(tg['place_w'][c])
            if pw > 0 and torch.isfinite(tg['xy'][c]).all():
                xy = (tg['xy'][c] - torch.tensor(V.XY_C, device=dev)) / V.XY_S
                z = tg['height'][c] / 2 / V.Z_S
                pl_l.append((r['place'][b, s], xy, z if torch.isfinite(z) else None, pw))
        for s, (k, hi) in states[b].items():
            if k == 0:
                continue
            st_l.append(r['state'][b, s]); st_t.append(k)
            if k == 1 and hi is not None and torch.isfinite(tg['hidden_xy'][hi]).all():
                xy = (tg['hidden_xy'][hi] - torch.tensor(V.XY_C, device=dev)) / V.XY_S
                pl_l.append((r['place'][b, s], xy, None, 0.5))
    if st_l:
        add('state', F.cross_entropy(torch.stack(st_l).float(), torch.as_tensor(st_t, device=dev), weight=torch.tensor([1.0, 2.0, 2.0], device=dev)))
    if pl_l:
        tot = 0.0
        for p, xy, z, wgt in pl_l:
            p = p.float()
            lv = p[3:6].clamp(-6, 3)
            nll = 0.5 * (((p[:2] - xy) ** 2) / lv[:2].exp() + lv[:2]).sum()
            if z is not None:
                nll = nll + 0.5 * ((p[2] - z) ** 2 / lv[2].exp() + lv[2])
            tot = tot + wgt * nll
        add('place', tot / len(pl_l))
    return r


def ident_losses(model, pooled_now, L, dev, bank=()):
    """Teachers' vectors + supervised contrast within a camera + SameHead pairs. pooled_all: list of
    (ident 512, cloth teacher, shape teacher, has, key (tag, cam, person), tick, xy, obj, zone)."""
    import torch
    import torch.nn.functional as F
    import slot_v2 as V
    add = lambda k, v: L.__setitem__(k, L.get(k, 0.0) + v)
    if not pooled_now:
        return
    pooled_all = list(pooled_now) + list(bank)
    cur = len(pooled_now)
    ident = torch.stack([p[0] for p in pooled_all]).float()
    has = torch.tensor([p[3] and i < cur for i, p in enumerate(pooled_all)], device=dev)
    if has.any():
        sc, ss = ident[has, :V.REID], ident[has, V.REID:]
        tc = torch.stack([p[1] for p, h in zip(pooled_all, has.tolist()) if h]).float()
        ts = torch.stack([p[2] for p, h in zip(pooled_all, has.tolist()) if h]).float()
        pc, ps = model.to_teacher[0](sc), model.to_teacher[1](ss)
        add('reid', (2 - F.cosine_similarity(pc, tc, dim=1) - F.cosine_similarity(ps, ts, dim=1)).mean())
        if len(sc) > 1:
            s = F.normalize(torch.cat([F.normalize(sc, dim=1), F.normalize(ss, dim=1)], 1), dim=1)
            t_ = F.normalize(torch.cat([F.normalize(tc, dim=1), F.normalize(ts, dim=1)], 1), dim=1)
            add('rel', F.mse_loss(s @ s.T, t_ @ t_.T))
    keys = [p[4] for p in pooled_all]
    cam = [(k[0], k[1]) for k in keys]
    n = len(keys)
    same = torch.tensor([[keys[i] == keys[j] for j in range(n)] for i in range(n)], device=dev)
    comparable = torch.tensor([[cam[i] == cam[j] for j in range(n)] for i in range(n)], device=dev)
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    z = F.normalize(ident, dim=1)
    sim = z @ z.T / 0.1
    pos = same & ~eye
    valid = comparable & ~eye
    if pos.any():
        logits = sim.masked_fill(~valid, -1e4)
        logp = logits - torch.logsumexp(logits, 1, keepdim=True)
        rows = pos.any(1) & (torch.arange(n, device=dev) < cur)
        if rows.any():
            add('supcon', -((logp * pos).sum(1)[rows] / pos.sum(1)[rows]).mean())
    # SameHead: all comparable pairs (balanced)
    ii, jj = torch.nonzero(valid & (torch.arange(n, device=dev)[:, None] < torch.arange(n, device=dev)[None])
                           & (torch.arange(n, device=dev)[:, None] < cur), as_tuple=True)
    if len(ii):
        lab = same[ii, jj].float()
        npos = int(lab.sum())
        if npos and npos < len(lab):
            neg_idx = torch.nonzero(lab == 0).flatten()
            keep = torch.cat([torch.nonzero(lab == 1).flatten(), neg_idx[torch.randperm(len(neg_idx), device=dev)[:max(npos * 3, 8)]]])
            ii, jj, lab = ii[keep], jj[keep], lab[keep]
        tick = torch.tensor([p[5] for p in pooled_all], device=dev, dtype=torch.float32)
        xy = torch.stack([p[6] for p in pooled_all]).float()
        obj = torch.tensor([p[7] for p in pooled_all], device=dev, dtype=torch.float32)
        zone = torch.stack([p[8] for p in pooled_all]).float()
        dt = (tick[ii] - tick[jj]).abs() * 0.08 / 60
        dist = (xy[ii, :2] - xy[jj, :2]).norm(dim=1) * V.XY_S
        var = (xy[ii, 3:5].exp().sum(1) + xy[jj, 3:5].exp().sum(1)).sqrt() * V.XY_S
        ctx = torch.stack([dt, dist, var, torch.ones_like(dt), obj[ii], obj[jj], zone[ii].argmax(1).float(), zone[jj].argmax(1).float()], 1)
        logit = model.same(ident[ii], ident[jj], ctx.detach())
        add('same', F.binary_cross_entropy_with_logits(logit.float(), lab))


# ---------------------------------------------------------------- a clip
P_DROP = 0.1          # MOTR: a track now and then lost -- a proposal must find the person again
P_FALSE = 0.3         # MOTR: now and then a false track (an empty proposal kept) -- the model must end it


def split_b(r, b):
    import torch
    out = {}
    for k, v in r.items():
        if k in ('aux', 'dn'):
            out[k] = [{kk: vv[b:b + 1] for kk, vv in a.items()} for a in v]
        elif torch.is_tensor(v) and v.dim() and v.shape[0] == 2:
            out[k] = v[b:b + 1]
        else:
            out[k] = v
    return out


def clip_step_old(model, asm, clip, dev, train, rng, far, use_dn, scale=1.0):
    """All frames of one clip, both cameras decoded as one batch. The backward pass runs after every frame
    (truncated in time: memory does not grow with the clip); a track's update is computed at the start of the
    next frame from the previous frame's detached slots, so it learns from that frame's losses. Identity
    losses compare this frame's slots with a detached bank of the clip's earlier frames. Returns the logs."""
    import torch
    import train_slots as TS
    logs = {}
    bank = []                   # detached (ident, key, tick, place, obj, zone) of earlier frames
    pending = [None, None]      # per camera: (detached slot outputs, keep, previous tracks, person_of)
    assign = [{}, {}]
    scene = [None, None]
    cams = ('cam1', 'cam2')
    n = len(clip['frames'])
    for fi, per in enumerate(clip['frames']):
        L = {}
        pooled_now = []
        tracks = [None, None]
        for b in range(2):
            if scene[b] is not None and isinstance(scene[b], tuple):    # the scene memory's update, in this frame's graph
                scene[b] = model.scene.update(*scene[b])
            if pending[b] is not None and not far:
                rq, keep, prev, person_of = pending[b]
                tr, idx = model.next_tracks(rq, keep, prev)
                tracks[b] = tr
                assign[b] = {j: person_of.get(int(s_)) for j, s_ in enumerate(idx[0].tolist())}
        rgb, bgv, sta, cam_ids = asm(per, cams, train, rng)
        with torch.no_grad():
            bg_sem = model.body.vit(bgv)[0]
        m = model.maps(rgb, sta, cam_ids, bg_sem, asm.last_world)
        tgs = [targets_tensor(f['tg'], dev) for f in per]
        dn, dn_matches = TS.make_dn(tgs, dev) if use_dn else (None, None)
        tb = None if far else model.stack_tracks(tracks, dev, m['mem'].dtype)
        sc = torch.cat([(scene[b] if scene[b] is not None else model.scene.start(cam_ids[b:b + 1])) for b in range(2)], 0)
        r = model.decode(m, cam_ids, tb, sc, dn)
        r1, r2 = model.cross_cameras(split_b(r, 0), split_b(r, 1))
        for b, rb in enumerate((r1, r2)):
            mb = split_b(m, b)
            rr, cc, st, aux = match_frame(dict(rb, pix=mb['pix']), 0, tgs[b], assign[b] if not far else {})
            frame_losses(model, mb, rb, [tgs[b]], [(rr, cc)], [st], None if dn_matches is None else [dn_matches[b]], L, [aux])
            if len(rr):
                wts = torch.zeros(1, rb['q'].shape[1], *mb['pix'].shape[-2:], device=dev, dtype=mb['pix'].dtype)
                wts[0, rr] = tgs[b]['masks'][cc].to(wts.dtype)
                po = model.pooled(mb, rb, wts)
                zl, zt = po['zone'][0, rr], tgs[b]['zone'][cc]
                k = zt >= 0
                if k.any():
                    L['zone'] = L.get('zone', 0.0) + torch.nn.functional.cross_entropy(zl[k].float(), zt[k], weight=torch.tensor(ZONE_WEIGHT, device=dev))
                for s_, c_ in zip(rr.tolist(), cc.tolist()):
                    tg = tgs[b]
                    pooled_now.append((po['ident'][0, s_], tg['cloth'][c_], tg['shape'][c_], bool(tg['has_reid'][c_]),
                                       (clip['tag'], cams[b], int(tg['person'][c_])), clip['ticks'][fi], rb['place'][0, s_].detach(),
                                       float(rb['obj'][0, s_].detach().sigmoid()), po['zone'][0, s_].detach().softmax(-1)))
            if far:
                continue
            keep = torch.zeros(1, rb['q'].shape[1], dtype=torch.bool, device=dev)
            person_of = {}
            for s_, c_ in zip(rr.tolist(), cc.tolist()):
                if rng.random() > P_DROP:
                    keep[0, s_] = True; person_of[s_] = int(tgs[b]['person'][c_])
            for s_, (k, _) in st.items():
                if k == 1 and not rb['pad'][0, s_]:
                    keep[0, s_] = True; person_of[s_] = assign[b].get(s_)
            if rng.random() < P_FALSE:
                T = rb['tracks']
                free = [s_ for s_ in range(T, rb['q'].shape[1]) if not keep[0, s_] and not rb['pad'][0, s_]]
                if free:
                    s_ = rng.choice(free)
                    keep[0, s_] = True; person_of[s_] = None
            det = {'q': rb['q'].detach(), 'box': rb['box'].detach(), 'tracks': rb['tracks']}
            prev = None if tb is None else {k_: v[b:b + 1].detach() for k_, v in tb.items()}
            pending[b] = (det, keep, prev, person_of)
            scene[b] = (sc[b:b + 1].detach(), model.norm(rb['q']).detach(), rb['pad'])
        ident_losses(model, pooled_now, L, dev, bank)
        bank += [(p[0].detach(),) + p[1:] for p in pooled_now]
        total = sum(W.get(k.replace('dn_', ''), 1.0) * v for k, v in L.items()) / n
        if train and torch.is_tensor(total):
            backward(total * scale)
        for k, v in L.items():
            logs[k] = logs.get(k, 0.0) + (float(v.detach()) if torch.is_tensor(v) else float(v)) / n
    return None, {k: round(v, 4) for k, v in logs.items()}


# ---------------------------------------------------------------- the fast step (see v2_fast.py for what changed and why)
def frame_losses_fast(model, m, r, tg, mt, dn_match, L, dev):
    """One camera's losses of a moment (B = 1): the same losses as frame_losses, without host stalls and per-person loops."""
    import torch
    import torch.nn.functional as F
    import train_slots as TS
    import slot_v2 as V
    import v2_fast as VF
    r = dict(r); r['pix'] = m['pix']
    pad = r['pad']
    silence = lambda o: dict(o, obj=o['obj'].masked_fill(pad, -20.0))
    r = silence(r)
    add = lambda k, v, n=1.0: L.__setitem__(k, L.get(k, 0.0) + v * n)
    for k, v in VF.set_losses_stack([r], m['pix'], tg, mt['rows_d'], mt['cols_d'], TS).items():
        add(k, v)
    pix_half = F.avg_pool2d(m['pix'], 2)
    tg_half = dict(tg, masks=F.avg_pool2d(tg['masks'][None], 2)[0] if len(tg['masks']) else tg['masks'].new_zeros(0, *pix_half.shape[-2:]))
    points = TS.LOSS_POINTS
    TS.LOSS_POINTS = AUX_POINTS
    try:
        if r['aux']:
            for k, v in VF.set_losses_stack([silence(a) for a in r['aux']], pix_half, tg_half, mt['arows_d'], mt['acols_d'], TS).items():
                add(k, v)
        if dn_match is not None and r.get('dn'):
            for k, v in VF.set_losses_stack(r['dn'], pix_half, tg_half, dn_match[0], dn_match[1], TS).items():
                add('dn_' + k, v)
    finally:
        TS.LOSS_POINTS = points
    h, w = m['cen'].shape[-2:]
    heat, size, sw = VF.centre_targets(tg, h, w, dev)
    add('cen', focal_heat(m['cen'][0, 0], heat))
    add('size', (F.l1_loss(m['cen'][0, 1:].float().sigmoid(), size, reduction='none') * sw).sum() / sw.sum().clamp(min=1.0))
    bt = boundary_target(tg['masks'])
    add('bnd', F.binary_cross_entropy_with_logits(m['bnd'][0, 0].float(), bt, pos_weight=VF.const('bnd_pw', 10.0, dev)))
    for k, v in VF.state_place_losses(r, tg, mt, dev, V).items():
        add(k, v)
    return r


def clip_step(model, asm, clip, dev, train, rng, far, use_dn, scale=1.0):
    """clip_step without the host stalls: everything that does not depend on the model (the frames' inputs, the
    targets, the denoising hints, the background features) is prepared for all moments up front, the cost of both
    cameras is copied to the host once per moment, the next frame's tracks are made before the backward pass is
    launched, and the logs stay on the card until the end of the clip. Returns (None, logs)."""
    import torch
    import train_slots as TS
    import slot_v2 as V
    import v2_fast as VF
    logs, logs_f = {}, {}
    bank = []
    scene = [None, None]
    assign = [{}, {}]
    nxt = [None, None]          # per camera: (the next frame's tracks, who each track is)
    cams = ('cam1', 'cam2')
    n = len(clip['frames'])
    prep = []
    for per in clip['frames']:
        rgb, bgv, sta, cam_ids = asm(per, cams, train, rng)
        world = asm.last_world
        with torch.no_grad():
            bg_sem = model.body.vit(bgv)[0]
        tgs = [VF.targets_tensor(f['tg'], dev) for f in per]
        dn, dn_matches = VF.make_dn(tgs, dev, TS.DN_MAX) if use_dn else (None, None)
        prep.append((rgb, sta, cam_ids, bg_sem, world, tgs, dn, dn_matches))
    for fi, (rgb, sta, cam_ids, bg_sem, world, tgs, dn, dn_matches) in enumerate(prep):
        L = {}
        pooled_now = []
        tracks = [None, None]
        for b in range(2):
            if scene[b] is not None and isinstance(scene[b], tuple):
                scene[b] = model.scene.update(*scene[b])
            if nxt[b] is not None and not far:
                tracks[b], assign[b] = nxt[b]
        m = model.maps(rgb, sta, cam_ids, bg_sem, world)
        tb = None if far else model.stack_tracks(tracks, dev, m['mem'].dtype)
        sc = torch.cat([(scene[b] if scene[b] is not None else model.scene.start(cam_ids[b:b + 1])) for b in range(2)], 0)
        r = model.decode(m, cam_ids, tb, sc, dn)
        r1, r2 = model.cross_cameras(split_b(r, 0), split_b(r, 1))
        rbs = (r1, r2)
        mbs = [split_b(m, b) for b in range(2)]
        launch = [VF.match_launch(dict(rbs[b], pix=mbs[b]['pix']), 0, tgs[b], TS) for b in range(2)]     # both cameras' costs on the card ...
        mts = [VF.match_finish(launch[b], tgs[b], assign[b] if not far else {}, dev) for b in range(2)]    # ... then one wait
        nxt = [None, None]
        for b, rb in enumerate(rbs):
            mb, tg, mt = mbs[b], tgs[b], mts[b]
            frame_losses_fast(model, mb, rb, tg, mt, None if dn_matches is None else dn_matches[b], L, dev)
            npy = tg['np']
            if mt['rows']:
                wts = torch.zeros(1, rb['q'].shape[1], *mb['pix'].shape[-2:], device=dev, dtype=mb['pix'].dtype)
                wts[0, mt['rows_d']] = tg['masks'][mt['cols_d']].to(wts.dtype)
                po = model.pooled(mb, rb, wts)
                zl, zt = po['zone'][0, mt['rows_d']], tg['zone'][mt['cols_d']]
                if (npy['zone'][mt['cols']] >= 0).any():
                    L['zone'] = L.get('zone', 0.0) + torch.nn.functional.cross_entropy(zl.float(), zt, weight=VF.const('zone_w', ZONE_WEIGHT, dev), ignore_index=-1)
                ident_b, zprob = po['ident'][0], po['zone'][0].detach().softmax(-1)
                objp, place_b = rb['obj'][0].detach().float().sigmoid(), rb['place'][0].detach()
                for s_, c_ in zip(mt['rows'], mt['cols']):
                    pooled_now.append((ident_b[s_], tg['cloth'][c_], tg['shape'][c_], bool(npy['has_reid'][c_]),
                                       (clip['tag'], cams[b], int(npy['person'][c_])), clip['ticks'][fi], place_b[s_], objp[s_], zprob[s_]))
            if far:
                continue
            S_ = rb['q'].shape[1]
            pad_cpu = mt['pad']
            keep = np.zeros(S_, bool)
            person_of = {}
            held = set()
            for s_, c_ in zip(mt['rows'], mt['cols']):          # tracks come first in the matches
                p_ = int(npy['person'][c_])
                if p_ in held:
                    continue                                    # a proposal on a person a track keeps: dropped, as the tracker does
                if rng.random() > P_DROP:
                    keep[s_] = True; person_of[s_] = p_; held.add(p_)
            for s_, (k, _) in mt['states'].items():
                if k == 1 and not pad_cpu[s_]:
                    keep[s_] = True; person_of[s_] = assign[b].get(s_)
            if rng.random() < P_FALSE:
                T_ = rb['tracks']
                free = [s_ for s_ in range(T_, S_) if not keep[s_] and not pad_cpu[s_]]
                if free:
                    s_ = rng.choice(free)
                    keep[s_] = True; person_of[s_] = None
            det = {'q': rb['q'].detach(), 'box': rb['box'].detach(), 'tracks': rb['tracks']}
            prev = None if tb is None else {k_: v[b:b + 1].detach() for k_, v in tb.items()}
            keep_cpu = torch.from_numpy(keep[None].copy())
            tr, idx = model.next_tracks(det, VF.to_dev(keep_cpu, dev), prev, keep_cpu=keep_cpu)
            nxt[b] = (tr, {j: person_of.get(int(s_)) for j, s_ in enumerate(idx[0].tolist())})
            scene[b] = (sc[b:b + 1].detach(), model.norm(rb['q']).detach(), rb['pad'])
        VF.ident_losses(model, pooled_now, L, dev, bank, V, rng)
        bank += [(p[0].detach(),) + p[1:] for p in pooled_now]
        total = sum(W.get(k.replace('dn_', ''), 1.0) * v for k, v in L.items()) / n
        if train and torch.is_tensor(total):
            backward(total * scale)
        for k, v in L.items():
            if torch.is_tensor(v):
                logs[k] = logs.get(k, 0.0) + v.detach().float() / n
            else:
                logs_f[k] = logs_f.get(k, 0.0) + float(v) / n
    keys = list(logs)
    vals = torch.stack([logs[k] for k in keys]).tolist() if keys else []
    out = {k: v for k, v in zip(keys, vals)}
    for k, v in logs_f.items():
        out[k] = out.get(k, 0.0) + v
    return None, {k: round(v, 4) for k, v in out.items()}


def draft_step(model, asm, item, dev, rng, use_dn, scale=1.0):
    """Two unrelated draft frames (v2_data.Drafts): finding people and their masks only -- a cold frame each, no
    tracks, the other camera empty. Returns (None, logs)."""
    import torch
    import train_slots as TS
    import v2_eval
    import v2_fast as VF
    frames, cams = item['frames'], item['cams']
    rgb, bgv, sta, cam_ids = asm(frames, cams, True, rng)
    world = asm.last_world
    with torch.no_grad():
        bg_sem = model.body.vit(bgv)[0]
    tgs = [VF.targets_tensor(f['tg'], dev) for f in frames]
    dn, dn_matches = VF.make_dn(tgs, dev, TS.DN_MAX) if use_dn else (None, None)
    m = model.maps(rgb, sta, cam_ids, bg_sem, world)
    sc = torch.cat([model.scene.start(cam_ids[b:b + 1]) for b in range(len(cams))], 0)
    r = model.decode(m, cam_ids, None, sc, dn)
    L = {}
    for b in range(len(cams)):
        rb = split_b(r, b)
        rb, _ = model.cross_cameras(rb, v2_eval.empty_like(rb))
        mb = split_b(m, b)
        mt = VF.match_finish(VF.match_launch(dict(rb, pix=mb['pix']), 0, tgs[b], TS), tgs[b], {}, dev)
        frame_losses_fast(model, mb, rb, tgs[b], mt, None if dn_matches is None else dn_matches[b], L, dev)
    total = sum(W.get(k.replace('dn_', ''), 1.0) * v for k, v in L.items())
    if torch.is_tensor(total):
        backward(total * scale)
    keys = [k for k, v in L.items() if torch.is_tensor(v)]
    vals = torch.stack([L[k].detach().float() for k in keys]).tolist() if keys else []
    return None, {k: round(v, 4) for k, v in zip(keys, vals)}


def init_from_v1(model, path):
    """Every v1 weight whose name and shape match (v1 learnt finding people on the rough drafts): the start the
    29.09 run took -- 26 -> 57 % found in one epoch."""
    import torch
    ck = torch.load(path, map_location='cpu')
    src = ck.get('ema', ck.get('model', ck))
    dst = model.state_dict()
    took = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(took)
    model.load_state_dict(dst)
    n = sum(v.numel() for v in took.values())
    return n, sum(v.numel() for v in dst.values())


# ---------------------------------------------------------------- main
def main():
    import multiprocessing as mp
    import torch
    import slot_v2 as V
    import v2_data as VD
    import train_slots as TS
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='runs/v2_a')
    ap.add_argument('--tags', nargs='*', default=None)
    ap.add_argument('--heldout', nargs='*', default=[])
    ap.add_argument('--weights', default='data/weights/deimv2_s/model.safetensors')
    ap.add_argument('--steps', type=int, default=20000)
    ap.add_argument('--freeze-steps', type=int, default=1500)
    ap.add_argument('--clips', type=int, default=1, help='clips per step')
    ap.add_argument('--T', type=int, default=4)
    ap.add_argument('--far', type=float, default=0.3, help='share of clips of two far-apart moments')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--vit-lr-mult', type=float, default=0.1)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--ema', type=float, default=0.9998)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--dn', type=int, default=1)
    ap.add_argument('--exam-every', type=int, default=0, help='(old) use --epoch-steps')
    ap.add_argument('--epoch-steps', type=int, default=1000, help='an epoch for the logs: this many steps, then the tests')
    ap.add_argument('--track-ticks', type=int, default=400, help='held-out window stretch for the tracking test')
    ap.add_argument('--resume', default='')
    ap.add_argument('--sched-from', type=int, default=-1, help='with --fresh-sched: the step where the cosine started (keeps the curve across restarts)')
    ap.add_argument('--no-cos', action='store_true', help='with --fresh-sched: no cosine either, the rates stay exactly as given')
    ap.add_argument('--freeze-vit-only', action='store_true', help='while frozen (--freeze-steps) only the ViT is frozen; the STA branch keeps learning')
    ap.add_argument('--fresh-sched', action='store_true', help='no warm-up; the cosine starts at --lr / --vit-lr-mult right at the resume step and falls to 0 at --steps')
    ap.add_argument('--test-every', type=int, default=1, help='run the exam and the tracking test after every N-th epoch (the checkpoint is saved every epoch)')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--ckpt', action='store_true', help='recompute ViT and decoder in backward (less memory, ~25 % slower)')
    ap.add_argument('--until', default='', help='HH:MM: save and stop then (the live counter takes the card at 10:00)')
    ap.add_argument('--drafts', type=float, default=0.0, help='share of the steps on rough draft frames (v2_data.Drafts)')
    ap.add_argument('--draft-only-steps', type=int, default=0, help='the first steps on drafts only (finding people first)')
    ap.add_argument('--draft-items', type=int, default=3, help='pairs of draft frames per draft step (a backward each)')
    ap.add_argument('--draft-workers', type=int, default=2)
    ap.add_argument('--far-workers', type=int, default=2)
    ap.add_argument('--init-v1', default='', help='v1 checkpoint: every weight whose name and shape match')
    ap.add_argument('--test', default='', help='v2_teacher_test list name: the student against SAM 3.1 after every epoch')
    ap.add_argument('--track-every', type=int, default=0, help='the tracking test every N epochs (0: with --test-every)')
    ap.add_argument('--owner-exam-every', type=int, default=0, help="the owner's 18.09 exam every N epochs (0: with --test-every)")
    ap.add_argument('--keep-every', type=int, default=1, help='keep epoch_NN.pt every N epochs (last.pt and best.pt always)')
    ap.add_argument('--lr-floor', type=float, default=0.0, help='the cosine ends at this share of the rate, not at 0')
    ap.add_argument('--backbone', default='', help='a timm name (e.g. repvit_m0_9.dist_450e_in1k): a light conv net instead of the ViT')
    ap.add_argument('--amp', default='bf16', choices=('bf16', 'fp16'), help='fp16 with loss scaling for cards without bf16 (T4)')
    ap.add_argument('--init', default='', help='any v2 checkpoint: every weight whose name and shape match (another variant as the start)')
    ap.add_argument('--decoder-layers', type=int, default=6)
    ap.add_argument('--slim', action='store_true', help='lighter identity heads (slot_v2.SlotV2 slim)')
    ap.add_argument('--rgb-size', default='', help='WxH of the RGB backbone input (default 1280x720; 2176x1224 = the full stored frame)')
    ap.add_argument('--det-only', action='store_true', help='finding people and their masks only: the other tasks weigh 0')
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    world, rank = int(os.environ.get('WORLD_SIZE', '1')), int(os.environ.get('RANK', '0'))
    main_rank = rank == 0
    if world > 1:                                    # torchrun --nproc_per_node N: one process per card, gradients averaged
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))
        import datetime
        dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=60))   # rank 0's tests: the others wait
    if a.amp == 'fp16' and dev == 'cuda':
        global SCALER
        SCALER = torch.amp.GradScaler('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True           # the input sizes never change: cuDNN picks its fastest kernels once
    all_tags = sorted({t for t, _ in VD.window_cams(exclude=('20260918',))})
    tags = [t for t in (a.tags or all_tags) if t not in a.heldout]
    if main_rank:
        json.dump(dict(vars(a), tags=tags, device=dev, world=world), open(out / 'args.json', 'w'), indent=1)
    print('windows:', tags, 'held out:', a.heldout, flush=True)
    rgb_size = tuple(int(x) for x in a.rgb_size.lower().split('x')) if a.rgb_size else None
    model = V.build(backbone=a.backbone or None, layers=a.decoder_layers, slim=a.slim, rgb_size=rgb_size)
    arch = {'layers': a.decoder_layers, 'slim': a.slim, 'rgb': list(model.rgb_size)}
    if a.det_only:
        for k in ('state', 'place', 'vel', 'zone', 'reid', 'rel', 'supcon', 'same'):
            W[k] = 0.0
    if a.weights and os.path.exists(ROOT / a.weights):
        from safetensors.torch import load_file
        missing, unexpected = model.body.load_deimv2(load_file(str(ROOT / a.weights)))
        print('DEIMv2: %d missing, %d unexpected' % (len(missing), len(unexpected)), flush=True)
    if a.init_v1:
        took, tot = init_from_v1(model, ROOT / a.init_v1)
        print('v1 weights: %.2f M of %.2f M' % (took / 1e6, tot / 1e6), flush=True)
    if a.init:
        took, tot = init_from_v1(model, ROOT / a.init)
        print('weights of %s: %.2f M of %.2f M' % (a.init, took / 1e6, tot / 1e6), flush=True)
    model.to(dev).train()
    model.to(memory_format=torch.channels_last)     # convolutions on Ampere tensor cores run faster this way
    model.set_checkpointing(a.ckpt)
    if world > 1:
        for v in model.state_dict().values():         # every rank starts from rank 0's weights
            dist.broadcast(v, 0)
    step0 = 0
    if a.resume:
        ck = torch.load(ROOT / a.resume, map_location='cpu')
        model.load_state_dict(ck['model']); step0 = ck.get('step', 0)
    ema = TS.EMA(model, a.ema) if main_rank else None
    if main_rank and a.resume and 'ema' in ck:
        ema.model.load_state_dict(ck['ema'])          # the tests score the average: carry it on, not start it again
        ema.updates = 20000                           # past the ramp
    groups = {}
    for n, p in model.named_parameters():
        key = ('vit' if n.startswith(('body.dinov3', 'body.cnn')) else 'body' if n.startswith('body.') else 'rest', p.dim() > 1)
        groups.setdefault(key, []).append(p)
    opt = torch.optim.AdamW([{'params': v, 'lr': a.lr * (a.vit_lr_mult if k[0] == 'vit' else 1.0), 'weight_decay': a.wd if k[1] else 0.0,
                              'name': k[0]} for k, v in groups.items()], fused=(dev == 'cuda'))
    if a.resume and 'opt' in ck:
        opt.load_state_dict(ck['opt'])            # Adam moments continue; the rates come from the arguments again
        for g in opt.param_groups:
            g['lr'] = a.lr * (a.vit_lr_mult if g['name'] == 'vit' else 1.0)
    base_lr = [g['lr'] for g in opt.param_groups]
    import sam31_reid
    for tag in (set(tags) | set(a.heldout)) if main_rank else ():   # the workers map the masks instead of each loading them
        for cam in ('cam1', 'cam2'):
            if (VD.SEG / tag / cam / 'chunks.npz').exists():
                sam31_reid.Masks.to_npy(VD.SEG / tag / cam / 'chunks.npz')
    if world > 1:
        dist.barrier()                                # the other ranks wait for rank 0's files
    q = mp.get_context('spawn').Queue(maxsize=8)
    procs = []
    for i in range(a.workers):
        T = a.T
        pr = mp.get_context('spawn').Process(target=VD.worker_stream, args=(tags, T, 3, 1000 + i + 100 * rank, q), daemon=True)
        pr.start(); procs.append(pr)
    far_q = mp.get_context('spawn').Queue(maxsize=8)
    if a.far > 0:
        for i in range(a.far_workers):
            pr = mp.get_context('spawn').Process(target=VD.worker_stream, args=(tags, 2, 3750, 2000 + i + 100 * rank, far_q), daemon=True)
            pr.start(); procs.append(pr)
    draft_q = mp.get_context('spawn').Queue(maxsize=8)
    if a.drafts > 0 or a.draft_only_steps > 0:
        for i in range(a.draft_workers):
            pr = mp.get_context('spawn').Process(target=VD.draft_stream, args=(3000 + i + 100 * rank, draft_q), daemon=True)
            pr.start(); procs.append(pr)
    test = None
    if a.test and main_rank:
        import v2_teacher_test
        test = v2_teacher_test.Test(a.test)
        print('teacher test: %d frames, %d people' % (len(test.items), test.people), flush=True)
    best_f1 = -1.0
    if a.resume and (out / 'epochs.jsonl').exists():           # a restart keeps the best so far, best.pt is not overwritten by a worse one
        for l in open(out / 'epochs.jsonl', encoding='utf-8'):
            try:
                best_f1 = max(best_f1, json.loads(l).get('test', {}).get('f1', -1.0))
            except ValueError:
                pass
    asm = VD.Assembler(dev, model.rgb_size)
    rng = random.Random(0)
    kind_rng = random.Random(7)                      # the kind of step: the same on every rank, so all use the same heads
    amp = dict(device_type='cuda', dtype=torch.bfloat16 if a.amp == 'bf16' else torch.float16, enabled=dev == 'cuda')
    log = open(out / 'log.jsonl', 'a') if main_rank else open(os.devnull, 'w')
    t0 = time.time()
    ep_sum, ep_cnt, ep_n, ep_t0 = {}, {}, 0, time.time()

    def status(step, extra=None):
        st = {'step': step, 'steps': steps, 'epoch': step // a.epoch_steps + 1, 'epoch_steps': a.epoch_steps,
              'elapsed_s': round(time.time() - t0), 'eta_s': round((time.time() - t0) / max(1, step - step0) * (steps - step)),
              'until': a.until, 'updated': time.strftime('%H:%M:%S')}
        st.update(extra or {})
        if main_rank:
            json.dump(st, open(out / 'status.json', 'w'), indent=1)
    steps = 3 if a.smoke else a.steps
    for step in range(step0, steps):
        stop = main_rank and (out / 'STOP').exists()
        if world > 1:
            flag = torch.tensor([float(stop)], device='cuda')
            dist.broadcast(flag, 0)
            stop = bool(flag.item())
        if stop:                                        # a graceful stop: save where we are, then end
            if not main_rank:
                break
            torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(), 'step': step, 'backbone': a.backbone or None, 'arch': arch}, out / 'last.pt')
            (out / 'STOP').unlink()
            status(step, {'phase': 'stopped'})
            print('stopped by STOP at step', step, flush=True)
            break
        if a.until and time.strftime('%H:%M') >= a.until and time.strftime('%H:%M') < '12:00' and world == 1:
            torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(), 'step': step, 'backbone': a.backbone or None, 'arch': arch}, out / 'last.pt')
            print('stopped at', a.until, 'step', step, flush=True)
            break
        frozen = step < a.freeze_steps
        for pn, p in model.body.named_parameters():
            p.requires_grad_(not (frozen and (not a.freeze_vit_only or pn.startswith(('dinov3', 'cnn')))))
        warm = min(1.0, (step - step0 + 1) / (500 if step0 == 0 else 200))   # after --resume the Adam moments are fresh: ramp again
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, steps))))
        if a.fresh_sched:
            warm = 1.0
            base = step0 if a.sched_from < 0 else a.sched_from
            cos = 0.5 * (1 + math.cos(math.pi * min(1.0, (step - base) / max(1, steps - base))))
            if a.no_cos:
                cos = 1.0
        cos = a.lr_floor + (1 - a.lr_floor) * cos
        for g, lr in zip(opt.param_groups, base_lr):
            g['lr'] = lr * warm * cos
        opt.zero_grad(set_to_none=True)
        logs = {}
        t_data = t_gpu = 0.0
        if dev == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        kind = 'draft' if step < a.draft_only_steps or (a.drafts > 0 and kind_rng.random() < a.drafts) else 'clip'
        if kind == 'draft':
            for c in range(a.draft_items):
                ts = time.time()
                item = draft_q.get()
                t_data += time.time() - ts
                ts = time.time()
                with torch.autocast(**amp):
                    _, lg = draft_step(model, asm, item, dev, rng, bool(a.dn), scale=1.0 / a.draft_items)
                if dev == 'cuda':
                    torch.cuda.synchronize()
                t_gpu += time.time() - ts
                for k, v in lg.items():
                    logs[k] = logs.get(k, 0) + v / a.draft_items
        for c in range(a.clips if kind == 'clip' else 0):
            far = a.far > 0 and kind_rng.random() < a.far
            kind = 'far' if far else 'clip'
            ts = time.time()
            clip = (far_q if far else q).get()
            t_data += time.time() - ts
            ts = time.time()
            with torch.autocast(**amp):
                _, lg = clip_step(model, asm, clip, dev, True, rng, far, bool(a.dn), scale=1.0 / a.clips)
            if dev == 'cuda':
                torch.cuda.synchronize()
            t_gpu += time.time() - ts
            for k, v in lg.items():
                logs[k] = logs.get(k, 0) + v / a.clips
        ts = time.time()
        if world > 1:
            allreduce_grads([p for p in model.parameters() if p.requires_grad], world)
        if SCALER is not None:
            SCALER.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if SCALER is not None:
            SCALER.step(opt)
            SCALER.update()
        else:
            opt.step()
        if main_rank:
            ema.update(model)
        t_gpu += time.time() - ts
        if not main_rank:
            continue
        if step % 20 == 0 or a.smoke:
            rec = dict(step=step, t=round(time.time() - t0), frozen=frozen, kind=kind, lr=opt.param_groups[0]['lr'], data_s=round(t_data, 1), gpu_s=round(t_gpu, 1),
                       mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if dev == 'cuda' else 0, **{k: round(v, 4) for k, v in logs.items()})
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + '\n'); log.flush()
        for k, v in logs.items():
            ep_sum[k] = ep_sum.get(k, 0.0) + v
            ep_cnt[k] = ep_cnt.get(k, 0) + 1
        ep_n += 1
        if step % 20 == 0:
            status(step + 1, {'phase': 'training', 'frozen': frozen, 'step_s': round(t_data + t_gpu, 2)})
        if (step + 1) % 500 == 0 or step == steps - 1:
            torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(), 'step': step + 1, 'backbone': a.backbone or None, 'arch': arch}, out / 'last.pt')
        if (step + 1) % a.epoch_steps == 0 or step == steps - 1:
            ep = (step + 1 + a.epoch_steps - 1) // a.epoch_steps
            rec = {'epoch': ep, 'step': step + 1, 'minutes': round((time.time() - ep_t0) / 60, 1), 'frozen': frozen,
                   'loss': {k: round(v / max(1, ep_cnt.get(k, 1)), 4) for k, v in ep_sum.items()}, 'time': time.strftime('%H:%M')}
            ck = {'model': model.state_dict(), 'ema': ema.model.state_dict(), 'opt': opt.state_dict(), 'step': step + 1, 'backbone': a.backbone or None, 'arch': arch}
            torch.save(ck, out / 'last.pt')
            if ep % a.keep_every == 0:
                torch.save(ck, out / ('epoch_%02d.pt' % ep))
            import v2_eval
            if test is not None:
                status(step + 1, {'phase': 'testing epoch %d' % ep})
                try:
                    rec['test'] = test.run(ema.model, dev)
                    if rec['test']['f1'] > best_f1:
                        best_f1 = rec['test']['f1']
                        torch.save(ck, out / 'best.pt')
                        rec['best'] = True
                except Exception as e:
                    rec['test'] = {'error': repr(e)[:300]}
            del ck
            if ep % (a.track_every or a.test_every) == 0 and a.heldout:
                status(step + 1, {'phase': 'tracking test, epoch %d' % ep})
                try:
                    rec['tracking'] = v2_eval.tracking(ema.model, dev, a.heldout[0], start=1500, ticks=a.track_ticks)
                except Exception as e:
                    rec['tracking'] = {'error': repr(e)[:300]}
            if ep % (a.owner_exam_every or a.test_every) == 0:
                status(step + 1, {'phase': "owner's exam, epoch %d" % ep})
                try:
                    rec['exam'] = v2_eval.exam(ema.model, dev)
                except Exception as e:
                    rec['exam'] = {'error': repr(e)[:300]}
            print('EPOCH', json.dumps(rec), flush=True)
            with open(out / 'epochs.jsonl', 'a') as f:
                f.write(json.dumps(rec) + '\n')
            ep_sum, ep_cnt, ep_n, ep_t0 = {}, {}, 0, time.time()
    for p in procs:
        p.terminate()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
