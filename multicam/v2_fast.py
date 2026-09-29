"""Faster pieces of the v2 training step (train_v2.py). Same maths as the originals, three things changed:

  * no host-device stalls: a small host array sent to the card from pageable memory makes CUDA drain the stream first
    (and .item(), .tolist(), boolean-mask indexing, a CPU index tensor on a CUDA tensor do the same). Measured before:
    ~500 such stalls per moment and 17 000 kernel launches per moment, the card mostly waiting for the Python thread.
    Here small host data goes through pinned memory with non_blocking copies, and everything the Python side needs to
    decide (who is matched, which slots are padding) is read once, after the one unavoidable sync (the Hungarian);
  * the per-person Python loops (centre targets, place / state losses, denoising hints, identity pairs) are tensor ops;
  * the helper layers' losses are computed in one call for all layers (layers are folded into the person dimension).

Every function has a reference in tests/ref_train_v2.py / ref_train_slots.py and a test in tests/test_v2_fast.py."""
import math

import numpy as np
import torch
import torch.nn.functional as F

_CONST = {}


def is_cuda(dev):
    return str(dev).startswith('cuda')


def to_dev(x, dev, dtype=None):
    """Small host data -> the card without a stream sync (pinned staging + non_blocking)."""
    t = x if torch.is_tensor(x) else torch.as_tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    if t.device.type != 'cpu':
        return t.to(dev)
    if is_cuda(dev):
        return t.contiguous().pin_memory().to(dev, non_blocking=True)
    return t.to(dev)


def const(name, values, dev, dtype=torch.float32):
    """A small constant tensor, made once per device."""
    k = (name, str(dev))
    if k not in _CONST:
        _CONST[k] = torch.tensor(values, dtype=dtype, device=dev)
    return _CONST[k]


# ------------------------------------------------------------------------------------------------ targets
FINAL_ALL = True    # 29.09: proposals learnt "not a person" on tracked people in 2/3 of the frames -> unsure on a cold frame
CPU_FIELDS = ('person', 'xy', 'height', 'place_w', 'zone', 'hidden', 'hidden_xy', 'has_reid')


def targets_tensor(tg, dev):
    """The frame's targets on the card, plus 'np': the small fields the Python side loops over, as numpy."""
    t = {k: to_dev(np.ascontiguousarray(v), dev) for k, v in tg.items()}
    t['masks'] = t['masks'].float()
    t['np'] = {k: np.asarray(tg[k]) for k in CPU_FIELDS}
    return t


def centre_targets(tg, h, w, dev):
    """Gaussian peaks at mask centres, sizes at the peaks -- all people at once (no .item(), no loop)."""
    heat = torch.zeros(h, w, device=dev)
    m = tg['masks']
    n = len(m)
    if not n:
        return heat, torch.zeros(2, h, w, device=dev), torch.zeros(h, w, device=dev)
    small = F.adaptive_avg_pool2d(m[None], (h, w))[0]                     # n x h x w
    tot = small.sum((1, 2))
    ok = (tot >= 1e-3).float()
    safe = tot.clamp(min=1e-3)
    ys = torch.arange(h, device=dev).float()
    xs = torch.arange(w, device=dev).float()
    cy = (small.sum(2) * ys).sum(1) / safe                                   # n
    cx = (small.sum(1) * xs).sum(1) / safe
    sig = (safe.sqrt() / 3).clamp(min=1.0)
    g = torch.exp(-((xs[None, None, :] - cx[:, None, None]) ** 2 + (ys[None, :, None] - cy[:, None, None]) ** 2) / (2 * sig ** 2)[:, None, None])
    heat = (g * ok[:, None, None]).amax(0)
    iy = cy.round().clamp(0, h - 1).long()
    ix = cx.round().clamp(0, w - 1).long()
    bad = ok == 0                                                            # people that are skipped go to a spare cell
    iy = torch.where(bad, torch.full_like(iy, h), iy)
    ix = torch.where(bad, torch.full_like(ix, w), ix)
    size = torch.zeros(2, h + 1, w + 1, device=dev)
    size[:, iy, ix] = tg['boxes'][:, 2:].T.to(size.dtype)
    sw = torch.zeros(h + 1, w + 1, device=dev)
    sw[iy, ix] = 1.0
    return heat, size[:, :h, :w], sw[:h, :w]


# ------------------------------------------------------------------------------------------------ denoising hints
def make_dn(targets, dev, dn_max=32):
    """DINO-style hints for a batch of frames (see train_slots.make_dn): the groups are made in one go."""
    n_max = max(len(t['boxes']) for t in targets)
    if n_max == 0:
        return None, None
    groups = max(1, dn_max // (2 * n_max))
    nd = groups * 2 * n_max
    B = len(targets)
    scale = const('dn_scale', [1.0, 1.0, 0.2, 0.5], dev)
    shift = const('dn_shift', [0.0, 0.0, 0.02, 0.05], dev)
    box = torch.rand(B, nd, 4, device=dev) * scale + shift
    group = torch.arange(groups, device=dev).repeat_interleave(2 * n_max)
    box4 = box.view(B, groups, 2 * n_max, 4)
    xy_only = const('dn_xy_only', [1.0, 1.0, 0.0, 0.0], dev)
    matches = []
    for b, t in enumerate(targets):
        tb = t['boxes'].float()
        n = len(tb)
        if n == 0:
            matches.append((to_dev(np.zeros(0, np.int64), dev), to_dev(np.zeros(0, np.int64), dev)))
            continue
        G = groups
        wh = tb[:, 2:].repeat(1, 2)[None]                                       # 1 x n x 4
        pos = tb[None] + (torch.rand(G, n, 4, device=dev) * 2 - 1) * 0.2 * wh * xy_only
        pos = torch.cat([pos[..., :2], tb[None, :, 2:] * (1 + (torch.rand(G, n, 2, device=dev) * 2 - 1) * 0.2)], -1)
        sign = torch.where(torch.rand(G, n, 2, device=dev) < 0.5, -1.0, 1.0)
        neg_xy = tb[None, :, :2] + sign * (0.4 + 0.6 * torch.rand(G, n, 2, device=dev)) * tb[None, :, 2:]
        neg = torch.cat([neg_xy, tb[None, :, 2:] * (0.6 + torch.rand(G, n, 2, device=dev))], -1)
        box4[b, :, :n] = pos
        box4[b, :, n_max:n_max + n] = neg
        r = (np.arange(G)[:, None] * 2 * n_max + np.arange(n)[None]).reshape(-1)
        c = np.tile(np.arange(n), G)
        matches.append((to_dev(r.astype(np.int64), dev), to_dev(c.astype(np.int64), dev)))
    box[..., :2] = box[..., :2].clamp(0.001, 0.999)
    box[..., 2:] = box[..., 2:].clamp(0.005, 1.0)
    return {'box': box, 'group': group}, matches


# ------------------------------------------------------------------------------------------------ matching
def match_costs(obj, box, mask_vec, pix, t, TS):
    """The Hungarian cost of train_slots.match for all people at once (a sub-matrix of it is the cost for fewer people)."""
    with torch.no_grad():
        pts = torch.rand(TS.MATCH_POINTS, 2, device=pix.device)
        n = len(t['boxes'])
        pl = mask_vec.float() @ TS.sample(pix, pts).T
        tm = TS.sample(t['masks'][:, None], pts[None].expand(n, -1, -1))[..., 0]
        p = obj.float().sigmoid()
        pos = F.softplus(-pl); neg = F.softplus(pl)
        bce = (pos @ tm.T + neg @ (1 - tm).T) / TS.MATCH_POINTS
        sp = pl.sigmoid()
        dice = 1 - (2 * sp @ tm.T + 1) / (sp.sum(-1)[:, None] + tm.sum(-1)[None] + 1)
        l1 = torch.cdist(box.float(), t['boxes'].float(), p=1)
        g = TS.giou(TS.box_xyxy(box.float()), TS.box_xyxy(t['boxes'].float()))
        return -TS.W_OBJ * p[:, None] + TS.W_L1 * l1 - TS.W_GIOU * g + TS.W_BCE * bce + TS.W_DICE * dice


def match_launch(r, b, tg, TS):
    """Start image b's matching on the card: the cost of every proposal against every person (no host copy yet)."""
    T = r['tracks']
    cost = None
    if len(tg['boxes']):
        cost = match_costs(r['obj'][b][T:], r['box'][b][T:], r['mask_vec'][b][T:], r['pix'][b], tg, TS)
    return {'cost': cost, 'pad': r['pad'][b], 'T': T}


def match_finish(h, tg, assign, dev):
    """The host half of train_v2.match_frame: one copy of the costs (and of the padding flags) for both variants of the
    matching. Launch both cameras first, finish both after: the two copies then wait for the forward once.

    Returns dict: rows/cols (final layer), arows/acols (helper layers), states {slot: (kind, hidden idx)}, *_d tensors,
    pad (numpy bool, all slots)."""
    from scipy.optimize import linear_sum_assignment
    npy = tg['np']
    person = npy['person'].tolist()
    col_of = {p: i for i, p in enumerate(person)}
    hidden = {int(p): i for i, p in enumerate(npy['hidden'].tolist())}
    rows, cols, states = [], [], {}
    taken = set()
    for s, p in assign.items():
        if p is not None and p in col_of:
            rows.append(s); cols.append(col_of[p]); states[s] = (0, None); taken.add(col_of[p])
        elif p is not None and p in hidden:
            states[s] = (1, hidden[p])
        else:
            states[s] = (2, None)
    T = h['T']
    pad_cpu = h['pad'].cpu().numpy()
    real = ~pad_cpu[T:]
    prop_idx = np.nonzero(real)[0] + T
    n = len(person)
    fr, fc, ar, ac = [], [], [], []
    if h['cost'] is not None and len(prop_idx):
        cost = h['cost'].cpu().numpy()[real]
        free = [i for i in range(n) if i not in taken]
        if free:
            rr, cc = linear_sum_assignment(cost[:, free])
            fr, fc = prop_idx[rr].tolist(), [free[i] for i in cc.tolist()]
        rr, cc = linear_sum_assignment(cost)
        ar, ac = prop_idx[rr].tolist(), cc.tolist()
    if FINAL_ALL:                  # the proposals find everybody on the final layer too (a detector); the track keeps who
        fr, fc = ar, ac            # it is, the tracker drops a proposal lying on a tracked person
    out = {'rows': rows + fr, 'cols': cols + fc, 'arows': rows + ar, 'acols': cols + ac, 'states': states, 'pad': pad_cpu}
    i64 = lambda x: to_dev(np.asarray(x, np.int64), dev)
    out['rows_d'], out['cols_d'] = i64(out['rows']), i64(out['cols'])
    out['arows_d'], out['acols_d'] = i64(out['arows']), i64(out['acols'])
    return out


# ------------------------------------------------------------------------------------------------ set losses
def giou_pair(a, b):
    """Elementwise generalised IoU of xyxy boxes (N x 4 with N x 4)."""
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)
    lt = torch.max(a[:, :2], b[:, :2]); rb = torch.min(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-6)
    lt2 = torch.min(a[:, :2], b[:, :2]); rb2 = torch.max(a[:, 2:], b[:, 2:])
    hull = (rb2 - lt2).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-6)


def set_losses_stack(outs, pix, tg, rows_d, cols_d, TS):
    """Mean over the layers `outs` of train_slots.set_losses (one image, B = 1): the layers are folded into the
    person dimension so the whole set is a handful of kernels. rows_d/cols_d: device long tensors of the matches."""
    Ly = len(outs)
    n = int(rows_d.shape[0])
    obj = torch.stack([o['obj'] for o in outs]).float()                            # Ly x 1 x S
    obj_t = torch.zeros_like(obj)
    zero = obj.new_zeros(())
    l1 = gi = bce = dc = zero
    npeople = max(1, n)
    if n:
        mv = torch.stack([o['mask_vec'][0, rows_d] for o in outs]).reshape(Ly * n, -1)
        tm = tg['masks'][cols_d].repeat(Ly, 1, 1)
        logits, target, k_imp = TS.mask_points(mv, pix[0], tm)
        with torch.no_grad():
            pu, tu = logits[:, k_imp:] > 0, target[:, k_imp:] > 0.5
            iou = (pu & tu).sum(1) / (pu | tu).sum(1).clamp(min=1)
        obj_t[:, 0, rows_d] = iou.clamp(min=0.05).view(Ly, n)
        pb = torch.stack([o['box'][0, rows_d] for o in outs]).float().reshape(Ly * n, 4)
        tb = tg['boxes'][cols_d].float().repeat(Ly, 1)
        l1 = F.l1_loss(pb, tb, reduction='sum') / Ly
        gi = (1 - giou_pair(TS.box_xyxy(pb), TS.box_xyxy(tb))).sum() / Ly
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(1).sum() / Ly
        pd = logits.sigmoid().flatten(1)
        t_ = target.flatten(1)
        dc = (1 - (2 * (pd * t_).sum(1) + 1) / (pd.sum(1) + t_.sum(1) + 1)).sum() / Ly
    p = obj.sigmoid()
    w = torch.where(obj_t > 0, obj_t, 0.75 * p.detach() ** 2)
    o = (F.binary_cross_entropy_with_logits(obj, obj_t, reduction='none') * w).sum() / npeople / Ly
    return {'obj': o, 'l1': l1 / npeople, 'giou': gi / npeople, 'bce': bce / npeople, 'dice': dc / npeople}


# ------------------------------------------------------------------------------------------------ state and place
def state_place_losses(r, tg, match, dev, V, ZONE_STATE_W=(1.0, 2.0, 2.0)):
    """State cross-entropy and the place NLL of one image (B = 1); the index lists come from the host (numpy), the maths is
    one gather per loss. Returns {'state': ..., 'place': ...} (missing when there is nothing to score)."""
    npy = tg['np']
    st_slot, st_tgt = [], []
    pl_slot, pl_xy, pl_z, pl_hz, pl_w = [], [], [], [], []
    xy_c = np.asarray(V.XY_C, np.float32)
    for s, c in zip(match['rows'], match['cols']):
        st_slot.append(s); st_tgt.append(0)
        pw = float(npy['place_w'][c])
        if pw > 0 and np.isfinite(npy['xy'][c]).all():
            z = float(npy['height'][c]) / 2 / V.Z_S
            ok = math.isfinite(z)
            pl_slot.append(s); pl_xy.append((npy['xy'][c] - xy_c) / V.XY_S); pl_z.append(z if ok else 0.0); pl_hz.append(1.0 if ok else 0.0); pl_w.append(pw)
    for s, (k, hi) in match['states'].items():
        if k == 0:
            continue
        st_slot.append(s); st_tgt.append(k)
        if k == 1 and hi is not None and np.isfinite(npy['hidden_xy'][hi]).all():
            pl_slot.append(s); pl_xy.append((npy['hidden_xy'][hi] - xy_c) / V.XY_S); pl_z.append(0.0); pl_hz.append(0.0); pl_w.append(0.5)
    out = {}
    if st_slot:
        pack = to_dev(np.asarray([st_slot, st_tgt], np.int64), dev)
        logits = r['state'][0][pack[0]].float()
        out['state'] = F.cross_entropy(logits, pack[1], weight=const('state_w', ZONE_STATE_W, dev))
    if pl_slot:
        idx = to_dev(np.asarray(pl_slot, np.int64), dev)
        num = to_dev(np.concatenate([np.asarray(pl_xy, np.float32), np.asarray(pl_z, np.float32)[:, None], np.asarray(pl_hz, np.float32)[:, None],
                                     np.asarray(pl_w, np.float32)[:, None]], 1), dev)
        p = r['place'][0][idx].float()
        lv = p[:, 3:6].clamp(-6, 3)
        nll = 0.5 * (((p[:, :2] - num[:, :2]) ** 2) / lv[:, :2].exp() + lv[:, :2]).sum(1)
        nll = nll + num[:, 3] * 0.5 * ((p[:, 2] - num[:, 2]) ** 2 / lv[:, 2].exp() + lv[:, 2])
        out['place'] = (num[:, 4] * nll).sum() / len(pl_slot)
    return out


# ------------------------------------------------------------------------------------------------ identity
def ident_losses(model, pooled_now, L, dev, bank, V, rng):
    """train_v2.ident_losses without per-pair Python->tensor conversions: the pair structure (who is the same person,
    which pairs are comparable, the negatives) is worked out in numpy, the maths runs on the gathered tensors.
    pooled: (ident, cloth, shape, has, key(tag, cam, person), tick, place(detached), obj(0-d tensor), zone probs)."""
    if not pooled_now:
        return
    dev_ = dev
    add = lambda k, v: L.__setitem__(k, L.get(k, 0.0) + v)
    pooled_all = list(pooled_now) + list(bank)
    cur = len(pooled_now)
    n = len(pooled_all)
    ident = torch.stack([p[0] for p in pooled_all]).float()
    has_idx = [i for i, p in enumerate(pooled_all) if p[3] and i < cur]
    if has_idx:
        hi = to_dev(np.asarray(has_idx, np.int64), dev_)
        sc, ss = ident[hi, :V.REID], ident[hi, V.REID:]
        tc = torch.stack([pooled_all[i][1] for i in has_idx]).float()
        ts = torch.stack([pooled_all[i][2] for i in has_idx]).float()
        pc, ps = model.to_teacher[0](sc), model.to_teacher[1](ss)
        add('reid', (2 - F.cosine_similarity(pc, tc, dim=1) - F.cosine_similarity(ps, ts, dim=1)).mean())
        if len(has_idx) > 1:
            s = F.normalize(torch.cat([F.normalize(sc, dim=1), F.normalize(ss, dim=1)], 1), dim=1)
            t_ = F.normalize(torch.cat([F.normalize(tc, dim=1), F.normalize(ts, dim=1)], 1), dim=1)
            add('rel', F.mse_loss(s @ s.T, t_ @ t_.T))
    keys = [p[4] for p in pooled_all]
    ids = {}
    kid = np.asarray([ids.setdefault(k, len(ids)) for k in keys])
    cams = {}
    cid = np.asarray([cams.setdefault((k[0], k[1]), len(cams)) for k in keys])
    same = kid[:, None] == kid[None]
    comparable = cid[:, None] == cid[None]
    eye = np.eye(n, dtype=bool)
    pos = same & ~eye
    valid = comparable & ~eye
    below = np.arange(n) < cur
    z = F.normalize(ident, dim=1)
    if pos.any():
        rows = pos.any(1) & below
        if rows.any():
            sim = z @ z.T / 0.1
            logits = sim.masked_fill(~to_dev(valid, dev_), -1e4)
            logp = logits - torch.logsumexp(logits, 1, keepdim=True)
            pos_t = to_dev(pos, dev_).float()
            per = (logp * pos_t).sum(1) / pos_t.sum(1).clamp(min=1)
            rw = to_dev(rows.astype(np.float32), dev_)
            add('supcon', -(per * rw).sum() / rw.sum())
    ii, jj = np.nonzero(valid & (np.arange(n)[:, None] < np.arange(n)[None]) & below[:, None])
    if len(ii):
        lab = same[ii, jj].astype(np.float32)
        npos = int(lab.sum())
        if npos and npos < len(lab):
            negs = np.nonzero(lab == 0)[0]
            pick = rng.sample(range(len(negs)), min(len(negs), max(npos * 3, 8)))
            keep = np.concatenate([np.nonzero(lab == 1)[0], negs[pick]])
            ii, jj, lab = ii[keep], jj[keep], lab[keep]
        tick = np.asarray([p[5] for p in pooled_all], np.float32)
        dt = np.abs(tick[ii] - tick[jj]) * 0.08 / 60
        pack = to_dev(np.stack([ii.astype(np.float32), jj.astype(np.float32), lab, dt]), dev_)        # ii, jj < 2^24: exact
        gi, gj = pack[0].long(), pack[1].long()
        xy = torch.stack([p[6] for p in pooled_all]).float()
        obj = torch.stack([p[7] for p in pooled_all]).float()
        zone = torch.stack([p[8] for p in pooled_all]).float()
        dist = (xy[gi, :2] - xy[gj, :2]).norm(dim=1) * V.XY_S
        var = (xy[gi, 3:5].exp().sum(1) + xy[gj, 3:5].exp().sum(1)).sqrt() * V.XY_S
        ctx = torch.stack([pack[3], dist, var, torch.ones_like(dist), obj[gi], obj[gj], zone[gi].argmax(1).float(), zone[gj].argmax(1).float()], 1)
        logit = model.same(ident[gi], ident[gj], ctx.detach())
        add('same', F.binary_cross_entropy_with_logits(logit.float(), pack[2]))
