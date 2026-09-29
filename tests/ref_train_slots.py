"""Train the slot model (slot_model.py) on the teacher+SAM drafts (slot_data.py).

Each slot is matched to one person of the draft (Hungarian: person-ness, box, mask); matched slots
copy the person's mask and box, the ReID teachers' two vectors and, where the owner answered /inout,
inside / outside / doorway. Unmatched slots learn "nobody". The stride-16 map copies C-RADIOv4-H.
Every decoder layer is supervised (Mask2Former). Identity is pooled under the draft's mask while
training (the model's own mask is still wrong early on), under its own mask when predicting.

The exam is the owner's 18.09 /paint frames: never trained on, scored like gold.evaluate (people
found at mask IoU >= 0.5, small ones, false ones, median mask IoU).

usage: train_slots.py OUT [--backbone NAME] [--epochs N] [--batch B] [--accum K] [--lr LR]
                          [--limit N] [--workers N] [--min-rating R] [--no-pretrained] [--cpu]
Resumes from OUT/last.pt. By day (09:40-21:02, camera 1 up) it saves and stops: the card belongs to the
live counter. Log: OUT/log.jsonl, exam: OUT/exam.json"""
import argparse
import datetime
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

W_OBJ, W_L1, W_GIOU, W_BCE, W_DICE = 2.0, 5.0, 2.0, 5.0, 5.0
W_RADIO, W_REID, W_REL, W_INOUT = 1.0, 1.0, 1.0, 0.5
INOUT_WEIGHT = (1.0, 0.6, 3.0)          # outside, inside, doorway: about the inverse of 337 / 650 / 65


class EMA:
    """Exponential moving average of the weights (and BN statistics); the exam scores the average.
    The decay ramps up over the first steps so the average does not remember the random start."""

    def __init__(self, model, decay):
        import copy
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.updates = decay, 0

    def update(self, model):
        import torch
        self.updates += 1
        d = self.decay * (1 - math.exp(-self.updates / 2000))
        with torch.no_grad():
            src = model.state_dict()
            for k, v in self.model.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(src[k].detach(), alpha=1 - d)
                else:
                    v.copy_(src[k])


def daytime():
    t = datetime.datetime.now().time()
    if not (datetime.time(9, 40) <= t < datetime.time(21, 2)):
        return False
    try:
        socket.create_connection((os.environ.get('RA_CAM_CAM1_IP', '192.168.99.241'), 554), 3).close()
        return True
    except OSError:
        return False


def box_xyxy(b):
    import torch
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou(a, b):
    """Pairwise generalised IoU of xyxy boxes, N x M."""
    import torch
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2]); rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    union = area(a)[:, None] + area(b)[None] - inter
    iou = inter / union.clamp(min=1e-6)
    lt2 = torch.min(a[:, None, :2], b[None, :, :2]); rb2 = torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb2 - lt2).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-6)


ONE_TO_MANY = 4              # slots per person on the auxiliary layers
DN_MAX = 32                  # hinted slots per frame at most (positives + negatives, all groups)


def make_dn(targets, device):
    """DINO-style hints: for every person a positive (true box moved and resized by up to 20 %: "mend
    it and outline the person") and a negative (moved 40-100 % off: "nobody here"), in as many groups as
    fit DN_MAX. Frames with fewer people pad with negatives. Returns the model's dn input and, per frame,
    (hint slot index, person index) of the positives."""
    import torch
    n_max = max(len(t['boxes']) for t in targets)
    if n_max == 0:
        return None, None
    groups = max(1, DN_MAX // (2 * n_max))
    nd = groups * 2 * n_max
    B = len(targets)
    box = torch.rand(B, nd, 4, device=device) * torch.tensor([1.0, 1.0, 0.2, 0.5], device=device) + torch.tensor([0.0, 0.0, 0.02, 0.05], device=device)
    group = torch.arange(groups, device=device).repeat_interleave(2 * n_max)
    matches = []
    for b, t in enumerate(targets):
        tb = t['boxes'].float().to(device)
        n = len(tb)
        r = []
        for g in range(groups):
            base = g * 2 * n_max
            if n == 0:
                continue
            wh = tb[:, 2:].repeat(1, 2)
            pos = tb + (torch.rand(n, 4, device=device) * 2 - 1) * 0.2 * wh * torch.tensor([1, 1, 0, 0], device=device)
            pos[:, 2:] = tb[:, 2:] * (1 + (torch.rand(n, 2, device=device) * 2 - 1) * 0.2)
            sign = torch.where(torch.rand(n, 2, device=device) < 0.5, -1.0, 1.0)
            neg = tb.clone()
            neg[:, :2] = tb[:, :2] + sign * (0.4 + 0.6 * torch.rand(n, 2, device=device)) * tb[:, 2:]
            neg[:, 2:] = tb[:, 2:] * (0.6 + torch.rand(n, 2, device=device))
            box[b, base:base + n] = pos
            box[b, base + n_max:base + n_max + n] = neg
            r += list(range(base, base + n))
        c = [i for g in range(groups) for i in range(n)] if n else []
        matches.append((torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c, dtype=torch.long)))
    box[..., :2] = box[..., :2].clamp(0.001, 0.999)
    box[..., 2:] = box[..., 2:].clamp(0.005, 1.0)
    return {'box': box, 'group': group}, matches


MATCH_POINTS = 4096           # uniform points per frame for the matching cost
LOSS_POINTS = 12544          # points per person for the mask loss (Mask2Former)
OVERSAMPLE, IMPORTANT = 3, 0.75


def sample(maps, pts):
    """maps: C x H x W or N x C x H x W; pts: K x 2 (shared) or N x K x 2, x y in 0..1 -> (N x) K x C."""
    import torch.nn.functional as F
    single = maps.dim() == 3
    m = maps[None] if single else maps
    g = pts[None] if pts.dim() == 2 else pts
    if g.shape[0] != m.shape[0]:
        g = g.expand(m.shape[0], -1, -1)
    v = F.grid_sample(m.float(), (g * 2 - 1)[:, :, None, :].float(), mode='bilinear', align_corners=False)[..., 0]  # N x C x K
    v = v.transpose(1, 2)
    return v[0] if single and pts.dim() == 2 else v


def match(obj, box, mask_vec, pix, t, k=1):
    """Slots <-> people of one frame; mask costs on MATCH_POINTS uniform points (Mask2Former).
    k > 1: one person may take up to k slots (one-to-many, for the auxiliary layers only: more
    "this is a person" lessons per step; the last layer stays one person - one slot)."""
    import torch
    import torch.nn.functional as F
    from scipy.optimize import linear_sum_assignment
    n = len(t['boxes'])
    if n == 0:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    with torch.no_grad():
        pts = torch.rand(MATCH_POINTS, 2, device=pix.device)
        pl = mask_vec.float() @ sample(pix, pts).T                          # S x K logits
        tm = sample(t['masks'][:, None], pts[None].expand(n, -1, -1))[..., 0]  # N x K
        p = obj.float().sigmoid()
        pos = F.softplus(-pl); neg = F.softplus(pl)                           # BCE against 1 and 0
        bce = (pos @ tm.T + neg @ (1 - tm).T) / MATCH_POINTS
        sp = pl.sigmoid()
        dice = 1 - (2 * sp @ tm.T + 1) / (sp.sum(-1)[:, None] + tm.sum(-1)[None] + 1)
        l1 = torch.cdist(box.float(), t['boxes'].float(), p=1)
        g = giou(box_xyxy(box.float()), box_xyxy(t['boxes'].float()))
        cost = -W_OBJ * p[:, None] + W_L1 * l1 - W_GIOU * g + W_BCE * bce + W_DICE * dice
    k = max(1, min(k, len(cost) // n))
    r, c = linear_sum_assignment(cost.repeat(1, k).cpu().numpy())
    return torch.as_tensor(r, dtype=torch.long), torch.as_tensor(c % n, dtype=torch.long)


def dice_loss(logits, target):
    p = logits.sigmoid().flatten(1); t = target.flatten(1)
    return (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()


def mask_points(mask_vec, pix, tmask):
    """Predicted logits and targets at LOSS_POINTS per person: 3/4 where the prediction is least sure
    (edges), 1/4 uniform (Mask2Former). The full mask is made only for the matched slots (a handful,
    not 32) and the points are read from that one-channel map: sampling the 128-channel embedding
    instead makes the backward pass scatter millions of values and is several times slower.
    mask_vec n x C, pix C x H x W, tmask n x H x W -> logits, targets (n x K), start of the uniform part."""
    import torch
    n = len(mask_vec)
    k_imp = int(LOSS_POINTS * IMPORTANT); k_uni = LOSS_POINTS - k_imp
    full = torch.einsum('nc,chw->nhw', mask_vec.float(), pix.float())[:, None]           # n x 1 x H x W
    with torch.no_grad():
        cand = torch.rand(n, LOSS_POINTS * OVERSAMPLE, 2, device=pix.device)
        cl = sample(full.detach(), cand)[..., 0]
        idx = (-cl.abs()).topk(k_imp, dim=1).indices
        imp = torch.gather(cand, 1, idx[..., None].expand(-1, -1, 2))
        pts = torch.cat([imp, torch.rand(n, k_uni, 2, device=pix.device)], 1)
    logits = sample(full, pts)[..., 0]
    target = sample(tmask[:, None].float(), pts)[..., 0]
    return logits, target, k_imp


def set_losses(out, pix, targets, matches):
    """Person-ness, box and mask losses of one set of slot outputs (the final one or an aux one)."""
    import torch
    import torch.nn.functional as F
    obj_t = torch.zeros_like(out['obj'], dtype=torch.float32)
    l1 = gi = bce = dc = out['obj'].new_zeros((), dtype=torch.float32)
    npeople = max(1, sum(len(r) for r, _ in matches))
    for b, (r, c) in enumerate(matches):
        if not len(r):
            continue
        logits, target, k_imp = mask_points(out['mask_vec'][b, r], pix[b], targets[b]['masks'][c])
        with torch.no_grad():                                     # IoU on the uniform points only: unbiased
            pu, tu = logits[:, k_imp:] > 0, target[:, k_imp:] > 0.5
            iou = (pu & tu).sum(1) / (pu | tu).sum(1).clamp(min=1)
        obj_t[b, r] = iou.clamp(min=0.05)
        pb, tb = out['box'][b, r].float(), targets[b]['boxes'][c].float()
        l1 = l1 + F.l1_loss(pb, tb, reduction='sum')
        gi = gi + (1 - torch.diag(giou(box_xyxy(pb), box_xyxy(tb)))).sum()
        bce = bce + F.binary_cross_entropy_with_logits(logits, target, reduction='none').mean(1).sum()
        dc = dc + dice_loss(logits, target) * len(r)
    # IoU-aware person-ness (varifocal, as DEIM's matchability-aware loss): a matched slot's target is how
    # well its mask fits, so the score ranks good masks first; nobody-slots are pushed down by p**2
    p = out['obj'].float().sigmoid()
    w = torch.where(obj_t > 0, obj_t, 0.75 * p.detach() ** 2)
    obj = (F.binary_cross_entropy_with_logits(out['obj'].float(), obj_t, reduction='none') * w).sum() / npeople
    return {'obj': obj, 'l1': l1 / npeople, 'giou': gi / npeople, 'bce': bce / npeople, 'dice': dc / npeople}


def step_losses(model, out, radio_t, radio_valid, targets):
    import torch
    import torch.nn.functional as F
    pix = out['pix']
    matches = [match(out['obj'][b], out['box'][b], out['mask_vec'][b], pix[b], t) for b, t in enumerate(targets)]
    L = set_losses(out, pix, targets, matches)
    for a in out['aux']:
        am = [match(a['obj'][b], a['box'][b], a['mask_vec'][b], pix[b], t, k=ONE_TO_MANY) for b, t in enumerate(targets)]
        for k, v in set_losses(a, pix, targets, am).items():
            L[k] = L[k] + v / len(out['aux'])
    if 'dn' in out:                                   # the hints: who is whom is known, no matching
        for a in out['dn']:
            for k, v in set_losses(a, pix, targets, out['dn_matches']).items():
                L['dn_' + k] = L.get('dn_' + k, 0.0) + v / len(out['dn'])
    # radio: cosine per valid cell
    v = radio_valid.to(out['radio'].device)
    if v.any():
        cos = F.cosine_similarity(out['radio'].float(), radio_t.float(), dim=1)
        L['radio'] = (1 - cos)[v].mean()
    # identity and place, pooled under the draft's masks for matched slots
    B, S = out['obj'].shape
    weights = torch.zeros(B, S, *pix.shape[-2:], device=pix.device, dtype=pix.dtype)
    for b, (r, c) in enumerate(matches):
        if len(r):
            weights[b, r] = targets[b]['masks'][c].to(weights.dtype)
    pooled = model.pooled(out, weights)
    sc, ss, tc, ts, io_p, io_t = [], [], [], [], [], []
    for b, (r, c) in enumerate(matches):
        t = targets[b]
        if not len(r):
            continue
        h = t['has_reid'][c]
        sc.append(pooled['cloth'][b, r][h]); ss.append(pooled['shape'][b, r][h])
        tc.append(t['cloth'][c][h]); ts.append(t['shape'][c][h])
        k = t['inout'][c] >= 0
        io_p.append(pooled['inout'][b, r][k]); io_t.append(t['inout'][c][k])
    if sc and sum(len(x) for x in sc):
        sc, ss, tc, ts = (torch.cat(x).float() for x in (sc, ss, tc, ts))
        pc, ps = model.to_teacher[0](sc), model.to_teacher[1](ss)
        L['reid'] = (2 - F.cosine_similarity(pc, tc, dim=1) - F.cosine_similarity(ps, ts, dim=1)).mean()
        if len(sc) > 1:                                     # relations: who is like whom, as the teachers see it
            s = F.normalize(torch.cat([F.normalize(sc, dim=1), F.normalize(ss, dim=1)], 1), dim=1)
            t_ = F.normalize(torch.cat([F.normalize(tc, dim=1), F.normalize(ts, dim=1)], 1), dim=1)
            L['rel'] = F.mse_loss(s @ s.T, t_ @ t_.T)
    if io_p and sum(len(x) for x in io_p):
        p, t = torch.cat(io_p).float(), torch.cat(io_t)
        L['inout'] = F.cross_entropy(p, t, weight=torch.tensor(INOUT_WEIGHT, device=p.device))
    W = {'obj': W_OBJ, 'l1': W_L1, 'giou': W_GIOU, 'bce': W_BCE, 'dice': W_DICE, 'radio': W_RADIO, 'reid': W_REID, 'rel': W_REL, 'inout': W_INOUT}
    W.update({'dn_' + k: W[k] for k in ('obj', 'l1', 'giou', 'bce', 'dice')})
    total = sum(W[k] * v for k, v in L.items())
    return total, {k: round(float(v.detach()), 4) for k, v in L.items()}


def predict(model, x, thr=0.5):
    """Slots -> people: [(score, mask logits at stride 4, box, cloth, shape, inout probs)] per frame.
    Masks and identity only for slots whose person-ness passes thr; no radio head (training only)."""
    import torch
    res = []
    with torch.no_grad():
        out = model(x, aux=False, radio=False)
        p = out['obj'].float().sigmoid()
        weights = torch.zeros(*p.shape, *out['pix'].shape[-2:], device=x.device, dtype=out['pix'].dtype)
        keep = [torch.nonzero(p[b] >= thr).flatten() for b in range(len(p))]
        masks = []
        for b, k in enumerate(keep):
            m = torch.einsum('sc,chw->shw', out['mask_vec'][b, k], out['pix'][b])
            weights[b, k] = m.sigmoid().to(weights.dtype)
            masks.append(m)
        pooled = model.pooled(out, weights)
    for b, k in enumerate(keep):
        res.append([(float(p[b, s]), masks[b][j].float(), out['box'][b, s].float(), pooled['cloth'][b, s].float(),
                     pooled['shape'][b, s].float(), pooled['inout'][b, s].float().softmax(-1)) for j, s in enumerate(k.tolist())])
    return res


def dedup(masks, scores, iou=0.5):
    """Indices kept after dropping a mask that covers, at IoU >= iou, a more confident one (a second
    slot on the same person). Masks: boolean arrays."""
    order = sorted(range(len(masks)), key=lambda j: -scores[j])
    kept = []
    for j in order:
        m = masks[j]
        if all((m & masks[i]).sum() < iou * (m | masks[i]).sum() for i in kept):
            kept.append(j)
    return kept


def exam(model, frames, dev, limit=None, thr=0.3):
    """The owner's 18.09 painted frames, scored as gold.evaluate does, at person-ness >= thr.
    `recall_any` ignores the score: could any of the 32 slots' masks find the person? It tells a bad
    mask from a mask the score does not trust yet."""
    import cv2
    import torch
    import torch.nn.functional as F
    import gold
    import slot_data
    split = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    where = json.load(open(ROOT / 'data' / 'seg_datasets' / 'backgrounds' / 'exam_frames.json'))
    ids = split[:limit] if limit else split
    found = total = false = small_found = small_total = any_found = found_d = false_d = 0
    ious, scores_hit = [], []
    model.eval()
    for ident in ids:
        w = where.get(ident, {})
        bgp = ROOT / 'data' / 'seg_datasets' / 'backgrounds' / ('%s_%s_%s.jpg' % (w.get('day'), w.get('cam'), str(w.get('segment', ''))[:-4])) if w else None
        s = slot_data.load(frames, 'paint_' + ident, False, None, image=gold.PAINT / ('%s.jpg' % ident),
                           label=gold.PAINT / ('%s_mask.png' % ident), background=bgp, cam_day=(w.get('day', '20260918'), w.get('cam', 'cam1')))
        x = torch.from_numpy(s['x'])[None].to(dev)
        with torch.autocast(dev.split(':')[0], torch.bfloat16, enabled=dev.startswith('cuda')):
            people = predict(model, x, thr=0.0)[0]
        truth = gold.gold(ident)
        pred, score = [], []
        for sc, m, *_ in people:
            full = F.interpolate(m[None, None], size=(slot_data.H, slot_data.W), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
            full = cv2.resize(full, (slot_data.SW, slot_data.SH), interpolation=cv2.INTER_LINEAR) > 0
            if full.sum() >= gold.MIN_PX:
                pred.append(full); score.append(sc)
        pairs_any, iou_any = gold.match(truth, pred)
        any_found += len(pairs_any)
        scores_hit += [score[j] for _, j in pairs_any]
        keep = [j for j, sc in enumerate(score) if sc >= thr]
        kd = [keep[i] for i in dedup([pred[j] for j in keep], [score[j] for j in keep])]
        pd_, _ = gold.match(truth, [pred[j] for j in kd])
        found_d += len(pd_); false_d += len(kd) - len(pd_)
        pairs, iou = gold.match(truth, [pred[j] for j in keep])
        total += len(truth); found += len(pairs); false += len(keep) - len(pairs)
        ious += [iou[i, j] for i, j in pairs]
        hit = {i for i, _ in pairs}
        for i, t in enumerate(truth):
            ys = np.nonzero(t.any(1))[0]
            if ys[-1] - ys[0] < gold.SMALL:
                small_total += 1; small_found += i in hit
    model.train()
    return {'frames': len(ids), 'people': total, 'thr': thr, 'recall': round(found / max(1, total), 4), 'false': false,
            'precision': round(found / max(1, found + false), 4),
            'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
            'small_recall': round(small_found / max(1, small_total), 4),
            'recall_any': round(any_found / max(1, total), 4),
            'dedup': {'recall': round(found_d / max(1, total), 4), 'false': false_d, 'precision': round(found_d / max(1, found_d + false_d), 4)},
            'score_of_hits_median': round(float(np.median(scores_hit)), 3) if scores_hit else None}


def main():
    import torch
    from torch.utils.data import DataLoader
    import slot_data
    from slot_model import SlotModel
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--backbone', default='deimv2_vit_tiny')
    ap.add_argument('--weights', default='', help='DEIMv2-S checkpoint (model.safetensors) for the deimv2 backbone')
    ap.add_argument('--vit-lr-mult', type=float, default=0.05)
    ap.add_argument('--wd', type=float, default=1e-4)
    ap.add_argument('--ema', type=float, default=0.9998)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--accum', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--min-rating', type=int, default=2)
    ap.add_argument('--exam-limit', type=int, default=0)
    ap.add_argument('--probe-every', type=int, default=250, help='optimizer steps between quick exams (0: off)')
    ap.add_argument('--dn', type=int, default=1, help='denoising hints (DN-DETR / DINO) while training')
    ap.add_argument('--init', default='', help='start from these weights (model + EMA), fresh optimizer: continue a run with a new recipe')
    ap.add_argument('--warmup', type=int, default=1000)
    ap.add_argument('--paint', action='store_true', help="fine-tune on the owner's /paint frames (not the exam day) mixed with drafts")
    ap.add_argument('--paint-mix', type=float, default=1.0, help='drafts per painted frame in the mix')
    ap.add_argument('--probe-frames', type=int, default=20)
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--cpu', action='store_true')
    ap.add_argument('--ignore-day', action='store_true')
    a = ap.parse_args()
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    dev = 'cpu' if a.cpu or not torch.cuda.is_available() else 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    frames = slot_data.Frames()
    ids = slot_data.train_ids(frames, a.min_rating)
    if a.limit:
        ids = ids[:: max(1, len(ids) // a.limit)][:a.limit]
    ds = slot_data.Dataset(frames, ids, train=True)
    if a.paint:                                   # the owner's masks, with as many drafts beside them so nothing is forgotten
        from torch.utils.data import ConcatDataset
        exam_ids = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
        paint = slot_data.Paint(frames, True, exclude_ids=exam_ids)
        rng_ = np.random.default_rng(0)
        mix = [ids[i] for i in rng_.choice(len(ids), min(len(ids), int(len(paint) * a.paint_mix)), replace=False)]
        ds = ConcatDataset([paint, slot_data.Dataset(frames, mix, train=True)])
        json.dump({'paint_frames': len(paint), 'draft_frames': len(mix), 'paint_days': sorted({it['day'] for it in paint.items})},
                  open(out_dir / 'paint_mix.json', 'w'), indent=1)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, collate_fn=slot_data.collate,
                    drop_last=True, persistent_workers=a.workers > 0, pin_memory=dev == 'cuda', prefetch_factor=4 if a.workers else None)
    model = SlotModel(a.backbone, pretrained=not a.no_pretrained, mean=frames.mean, std=frames.std, teacher_dims=frames.teacher_dims)
    if a.weights:
        from safetensors.torch import load_file
        missing, unexpected = model.body.load_deimv2(load_file(a.weights) if a.weights.endswith('.safetensors') else torch.load(a.weights, map_location='cpu'))
        loaded = {'missing': [k for k in missing if not k.startswith(('ours', 'imnet'))][:20], 'unexpected': unexpected[:20]}
        json.dump(loaded, open(out_dir / 'weights_loaded.json', 'w'), indent=1)
    model = model.to(dev).to(memory_format=torch.channels_last)
    # DEIMv2's recipe: the pretrained ViT learns 20x slower than the rest; no decay on norms and biases
    groups = {('vit', True): [], ('vit', False): [], ('rest', True): [], ('rest', False): []}
    for n, p in model.named_parameters():
        vit = n.startswith(('body.dinov3.', 'body.net.'))
        decay = p.ndim > 1 and 'norm' not in n and 'bn' not in n
        groups[('vit' if vit else 'rest', decay)].append(p)
    opt = torch.optim.AdamW([{'params': v, 'lr': a.lr * (a.vit_lr_mult if k[0] == 'vit' else 1.0), 'weight_decay': a.wd if k[1] else 0.0}
                             for k, v in groups.items() if v], betas=(0.9, 0.999))
    ema = EMA(model, a.ema)
    steps_total = a.epochs * max(1, len(dl) // a.accum)
    warm = min(a.warmup, steps_total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps_total))))
    start_epoch, step = 0, 0
    if a.init and not (out_dir / 'last.pt').exists():
        ck = torch.load(a.init, map_location='cpu')
        model.load_state_dict(ck['model'], strict=False)
        if 'ema' in ck:
            ema.model.load_state_dict(ck['ema'], strict=False); ema.updates = max(ck.get('ema_updates', 0), 20000)
        json.dump({'init': a.init, 'from_epoch': ck.get('epoch'), 'from_step': ck.get('step')}, open(out_dir / 'init.json', 'w'), indent=1)
    last = out_dir / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location='cpu')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        if 'ema' in ck:
            ema.model.load_state_dict(ck['ema']); ema.updates = ck.get('ema_updates', 0)
        start_epoch, step = ck['epoch'], ck['step']
    json.dump(vars(a) | {'frames': len(ids), 'device': dev, 'params_M': round(sum(p.numel() for p in model.parameters()) / 1e6, 2)},
              open(out_dir / 'args.json', 'w'), indent=1)
    log = open(out_dir / 'log.jsonl', 'a')
    amp = dict(device_type='cuda', dtype=torch.bfloat16) if dev == 'cuda' else dict(device_type='cpu', enabled=False)

    def save(epoch, name='last.pt'):
        torch.save({'model': model.state_dict(), 'ema': ema.model.state_dict(), 'ema_updates': ema.updates, 'opt': opt.state_dict(), 'sched': sched.state_dict(), 'epoch': epoch, 'step': step,
                    'backbone': a.backbone}, out_dir / (name + '.tmp'))
        os.replace(out_dir / (name + '.tmp'), out_dir / name)
    model.train()
    best = -1.0
    for epoch in range(start_epoch, a.epochs):
        t0 = time.time(); agg = {}; n = 0
        for k, (x, radio, valid, targets) in enumerate(dl):
            if not a.ignore_day and k % 50 == 0 and daytime():
                save(epoch); log.write(json.dumps({'stopped': time.strftime('%H:%M:%S'), 'why': 'day: the card is the live counter\'s'}) + '\n'); log.flush()
                return
            x = x.to(dev, non_blocking=True).to(memory_format=torch.channels_last)
            radio = radio.to(dev, non_blocking=True)
            targets = [{kk: v.to(dev) for kk, v in t.items()} for t in targets]
            dn, dn_matches = make_dn(targets, dev) if a.dn else (None, None)
            with torch.autocast(**amp):
                out = model(x, dn=dn)
            if dn is not None:
                out['dn_matches'] = dn_matches
            loss, parts = step_losses(model, out, radio, valid, targets)
            (loss / a.accum).backward()
            if (k + 1) % a.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); sched.step(); step += 1
                ema.update(model)
                if a.probe_every and step % a.probe_every == 0:       # a quick look: the first 20 exam frames
                    pr = exam(ema.model, frames, dev, a.probe_frames)
                    log.write(json.dumps({'probe': step, 'epoch': epoch, 'it': k, 'at': time.strftime('%H:%M'), **pr}) + '\n'); log.flush()
                    save(epoch, 'probe.pt')                            # for looking at it from outside while it trains
            n += 1
            for kk, v in parts.items():
                agg[kk] = agg.get(kk, 0.0) + v
            if k % 50 == 0:
                log.write(json.dumps({'epoch': epoch, 'it': k, 'of': len(dl), 'loss': round(float(loss), 4), **parts,
                                      's_per_it': round((time.time() - t0) / (k + 1), 3)}) + '\n'); log.flush()
        ex = exam(ema.model, frames, dev, a.exam_limit or None)
        rec = {'epoch': epoch, 'minutes': round((time.time() - t0) / 60, 1), 'mean': {kk: round(v / max(1, n), 4) for kk, v in agg.items()}, 'exam': ex}
        log.write(json.dumps(rec) + '\n'); log.flush()
        save(epoch + 1)
        if ex['recall'] * ex['precision'] > best:
            best = ex['recall'] * ex['precision']; save(epoch + 1, 'best.pt')
        json.dump(rec, open(out_dir / 'exam.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
