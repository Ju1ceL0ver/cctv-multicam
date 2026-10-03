"""Identity from a crop, after the segmentation: the slot model finds a person (box and mask), the
person is cut out of the full frame together with his mask, and a small ReID net gives the slot its
two vectors -- 256 clothes (copies TransReID MSMT17) and 256 body shape (copies CSCI LTCC).

Why a crop and not the slot model's own feature map: at the door the full 2560x1440 frame shows a
person 2.4x bigger than the model's 1088x608 input, and the teachers themselves look at crops. The
mask is a 4th input channel, so in a crowd the net looks at this person, not the neighbour in the crop.

Net: our own -- the ViT-Tiny of DEIMv2-S (see make()); the heavy teachers are what it learns from.
Training: the teacher+SAM drafts' people (crop from the 1280x720 frame, the draft's mask) and the
teachers' vectors of them (data/teacher_emb): cosine to each teacher through a projection, and the
relations between people of a batch as the teachers see them. Held out: the 401 held-out drafts.

usage: crop_reid.py train OUT.pt [--epochs N]     -> OUT.pt, OUT.log.jsonl
       crop_reid.py door OUT.pt SLOTCKPT            -> door pairs 17-19.09 with the slot model's boxes and masks"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CW, CH = 128, 256
MASK_DROP = 0.2                  # share of draft crops shown without their mask
W_ID = 0.5                       # weight of the identity lesson on track pieces
MARGIN = 0.05                    # as teacher_emb.py cut its crops


def crop(img, mask, box, jitter=None):
    """RGB-uint8 crop and 0/1 mask crop of one person, CW x CH. box x1 y1 x2 y2 in img pixels."""
    import cv2
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = x2 - x1, y2 - y1
    mx, my = MARGIN * w + 1, MARGIN * h + 1
    if jitter is not None:
        x1 += jitter[0] * w; x2 += jitter[1] * w; y1 += jitter[2] * h; y2 += jitter[3] * h
    X1, Y1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    X2, Y2 = min(img.shape[1], int(np.ceil(x2 + mx))), min(img.shape[0], int(np.ceil(y2 + my)))
    if X2 - X1 < 4 or Y2 - Y1 < 8:
        return None, None
    c = cv2.resize(img[Y1:Y2, X1:X2], (CW, CH), interpolation=cv2.INTER_LINEAR)
    m = cv2.resize(mask[Y1:Y2, X1:X2].astype(np.float32), (CW, CH), interpolation=cv2.INTER_LINEAR)
    return c, m


IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
DEIMV2 = ROOT / 'data' / 'weights' / 'deimv2_s' / 'model.safetensors'


def make(weights=DEIMV2, device='cpu', teacher_dims=(3840, 1024)):
    """Our own crop ReID net: the ViT-Tiny of DEIMv2-S (distilled from DINOv3-S, the slot model's own
    backbone; NON-COMMERCIAL, see deimv2_vit.py) on a 256x128 crop = 16x8 tokens. The person's mask is
    added to the patch embedding through a patch convolution of its own that starts at zero, and the
    tokens are pooled with the mask's weight next to the class token: the net looks at this person."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from deimv2_vit import VisionTransformer

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = VisionTransformer(embed_dim=192, num_heads=3, return_layers=[11], drop_path_rate=0.1)
            if weights and Path(weights).exists():
                from safetensors.torch import load_file
                sd = load_file(str(weights))
                pre = 'backbone.dinov3._model.'
                self.vit._model.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=True)
            self.register_buffer('mean', torch.tensor(IMAGENET[0]).reshape(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor(IMAGENET[1]).reshape(1, 3, 1, 1))
            self.mask_embed = nn.Conv2d(1, 192, 16, 16)
            nn.init.zeros_(self.mask_embed.weight); nn.init.zeros_(self.mask_embed.bias)
            self._mask = None
            self.vit._model.patch_embed.register_forward_hook(lambda mod, inp, out: out + self._mask if self._mask is not None else out)
            self.norm = nn.LayerNorm(384)
            self.cloth = nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Linear(384, 256), nn.BatchNorm1d(256))
            self.shape = nn.Sequential(nn.Linear(384, 384), nn.GELU(), nn.Linear(384, 256), nn.BatchNorm1d(256))
            self.to_teacher = nn.ModuleList([nn.Linear(256, teacher_dims[0]), nn.Linear(256, teacher_dims[1])])
            # the meta model: clothes and shape in, one identity vector out -- a learned, non-linear mix
            self.fuse = nn.Sequential(nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(), nn.Linear(512, 512), nn.LayerNorm(512), nn.GELU(),
                                      nn.Linear(512, 256))

        def fused(self, c, s):
            return self.fuse(torch.cat([F.normalize(c.float(), dim=1), F.normalize(s.float(), dim=1)], 1))

        def forward(self, rgb, mask):
            """rgb: N x 3 x CH x CW in 0..1, mask: N x 1 x CH x CW -> (clothes 256, shape 256), not normalised."""
            x = (rgb - self.mean) / self.std
            self._mask = self.mask_embed(mask).flatten(2).transpose(1, 2)
            try:
                tokens, cls = self.vit(x)[-1]
            finally:
                self._mask = None
            w = F.avg_pool2d(mask, 16).flatten(1)                     # the person's share of each patch
            w = (w + 0.05) / (w + 0.05).sum(1, keepdim=True)
            f = self.norm(torch.cat([cls, (tokens * w[..., None]).sum(1)], 1))
            return self.cloth(f), self.shape(f)

    return Net().to(device)


def frame_people(frames, ident, emb_dir, train, rng):
    """All people of one draft frame: crops, masks, teacher vectors."""
    import cv2
    import slot_data
    z = np.load(Path(emb_dir) / (ident + '.npz'))
    if not len(z['labels']):
        return None
    img = cv2.imread(str(frames.image(ident)))
    lab = cv2.imread(str(frames.drafts[ident]), cv2.IMREAD_UNCHANGED)
    lab = lab[:, :, 0] if lab.ndim == 3 else lab
    if lab.shape != img.shape[:2]:
        lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    rgb, msk, tc, ts = [], [], [], []
    for k, v in enumerate(z['labels']):
        jit = rng.uniform(-0.05, 0.05, 4) if train else None
        c, m = crop(img, lab == v, z['boxes'][k], jit)
        if c is None:
            continue
        if train:
            c = np.clip((c.astype(np.float32) - 128) * rng.uniform(0.85, 1.15) + 128 + rng.uniform(-12, 12), 0, 255).astype(np.uint8)
        rgb.append(c[:, :, ::-1]); msk.append(m); tc.append(z['cloth'][k]); ts.append(z['shape'][k])
    if not rgb:
        return None
    return (np.stack(rgb), np.stack(msk), np.stack(tc).astype(np.float32), np.stack(ts).astype(np.float32))


class People:
    def __init__(self, frames, ids, emb_dir, train):
        self.frames, self.ids, self.emb_dir, self.train = frames, ids, emb_dir, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, k):
        rng = np.random.default_rng() if self.train else np.random.default_rng(k)
        return frame_people(self.frames, self.ids[k], self.emb_dir, self.train, rng)


def collate(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    rgb = torch.from_numpy(np.concatenate([b[0] for b in batch])).permute(0, 3, 1, 2).float() / 255.0
    msk = torch.from_numpy(np.concatenate([b[1] for b in batch]))[:, None].float()
    tc = torch.from_numpy(np.concatenate([b[2] for b in batch]))
    ts = torch.from_numpy(np.concatenate([b[3] for b in batch]))
    return rgb, msk, tc, ts


TAU_S, TAU_T = 0.1, 0.05          # temperatures of the student's and the teachers' "who is like whom"


def rank_kl(student, teacher):
    """Every person's distribution over the others -- softmax of similarities -- as the teacher has it:
    the order of likeness, which is what picking the right one among the people inside needs."""
    import torch
    import torch.nn.functional as F
    s = F.normalize(student.float(), dim=1); t = F.normalize(teacher.float(), dim=1)
    n = len(s)
    eye = torch.eye(n, dtype=torch.bool, device=s.device)
    ls = (s @ s.T / TAU_S).masked_fill(eye, -1e4)
    lt = (t @ t.T / TAU_T).masked_fill(eye, -1e4)
    return F.kl_div(F.log_softmax(ls, 1), F.softmax(lt, 1), reduction='batchmean')


def losses(net, rgb, msk, tc, ts):
    import torch
    import torch.nn.functional as F
    c, s = net(rgb, msk)
    pc, ps = net.to_teacher[0](c), net.to_teacher[1](s)
    L = {'cloth': (1 - F.cosine_similarity(pc, tc, dim=1)).mean(), 'shape': (1 - F.cosine_similarity(ps, ts, dim=1)).mean()}
    if len(c) > 2:
        both = torch.cat([F.normalize(tc.float(), dim=1), F.normalize(ts.float(), dim=1)], 1)
        L['rank_cloth'] = rank_kl(c, tc)
        L['rank_shape'] = rank_kl(s, ts)
        L['rank_fused'] = rank_kl(net.fused(c, s), both)
    return L


def identity_loss(net, rgb, msk, piece, group):
    """InfoNCE on the fused vector over harvested track pieces: two pictures of one piece are one person,
    pieces the camera showed at the same time are different people; pieces apart in time are left alone
    (they may be one person broken by the tracker)."""
    import torch
    import torch.nn.functional as F
    c, s = net(rgb, msk)
    f = F.normalize(net.fused(c, s), dim=1)
    sim = f @ f.T / TAU_S
    same = piece[:, None] == piece[None, :]
    known = (group[:, None] == group[None, :]) | same        # same group: simultaneous, so certainly told apart; a piece drawn twice is still itself
    eye = torch.eye(len(f), dtype=torch.bool, device=f.device)
    pos = same & ~eye
    logits = sim.masked_fill(~known | eye, -1e4)
    has = pos.any(1)
    if not has.any():
        return sim.sum() * 0
    lp = F.log_softmax(logits, 1)
    return -(lp.masked_fill(~pos, 0).sum(1)[has] / pos.sum(1)[has]).mean()


class Pieces:
    """Harvested track pieces of the given days (reid_harvest), batched as train_reid does: groups of
    pieces the same camera showed at the same time; two pictures of each piece."""

    def __init__(self, days, seed=0):
        import random
        from train_reid import load_people, rivals, batches
        self.people = load_people(days, ROOT)
        self.gen = batches(self.people, rivals(self.people), random.Random(seed))
        self.rng = random.Random(seed + 1)

    def next(self):
        import cv2
        rgb, piece, group = [], [], []
        for g, members in enumerate(next(self.gen)):
            for i in members:
                for f in self.rng.sample(self.people[i]['files'], min(2, len(self.people[i]['files']))):
                    im = cv2.resize(cv2.imread(f), (CW, CH))
                    if self.rng.random() < 0.5:
                        im = im[:, ::-1]
                    rgb.append(np.ascontiguousarray(im[:, :, ::-1])); piece.append(i); group.append(g)
        return np.stack(rgb), np.array(piece), np.array(group)


def evaluate(net, dl, dev):
    """Held-out: cosine to the teachers, and how often a person's nearest neighbour by the student is the
    one the teachers call nearest (within the batch)."""
    import torch
    import torch.nn.functional as F
    net.eval()
    cc, cs, agree, n = [], [], 0, 0
    with torch.no_grad():
        for b in dl:
            if b is None:
                continue
            rgb, msk, tc, ts = (t.to(dev) for t in b)
            c, s = net(rgb, msk)
            cc.append(F.cosine_similarity(net.to_teacher[0](c), tc, dim=1).mean().item())
            cs.append(F.cosine_similarity(net.to_teacher[1](s), ts, dim=1).mean().item())
            if len(c) > 2:
                v = F.normalize(net.fused(c, s), dim=1)
                t = torch.cat([F.normalize(tc, dim=1), F.normalize(ts, dim=1)], 1)
                sv, st = v @ v.T, t @ t.T
                sv.fill_diagonal_(-9); st.fill_diagonal_(-9)
                agree += int((sv.argmax(1) == st.argmax(1)).sum()); n += len(c)
    net.train()
    return {'cos_clothes': round(float(np.mean(cc)), 4), 'cos_shape': round(float(np.mean(cs)), 4),
            'same_nearest_as_teachers': round(agree / max(1, n), 4)}


def train(out, epochs=8, batch=24, workers=8, lr=3e-4, piece_days=('20260917',), rank=True):
    import torch
    from torch.utils.data import DataLoader
    import slot_data
    import eval_slots
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    frames = slot_data.Frames()
    emb = ROOT / 'data' / 'teacher_emb'
    have = {p.stem for p in emb.glob('*.npz')}
    tr = [i for i in slot_data.train_ids(frames, 3) if i in have]
    va = [i for i in eval_slots.held_out_ids() if i in have]
    net = make(DEIMV2, dev, frames.teacher_dims)
    body = [p for n, p in net.named_parameters() if n.startswith('vit.')]
    rest = [p for n, p in net.named_parameters() if not n.startswith('vit.')]
    opt = torch.optim.AdamW([{'params': body, 'lr': lr / 6}, {'params': rest, 'lr': lr}], weight_decay=1e-4)
    dl = DataLoader(People(frames, tr, emb, True), batch_size=batch, shuffle=True, num_workers=workers, collate_fn=collate,
                    persistent_workers=workers > 0, drop_last=True)
    dv = DataLoader(People(frames, va, emb, False), batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate)
    total = epochs * len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + np.cos(np.pi * min(1.0, s / total))))
    pieces = Pieces(piece_days) if piece_days else None
    log = open(str(out) + '.log.jsonl', 'a')
    rec = {'epoch': -1, 'heldout': evaluate(net, dv, dev), 'frames_train': len(tr), 'frames_heldout': len(va)}
    log.write(json.dumps(rec) + '\n'); log.flush()
    for epoch in range(epochs):
        t0, agg, n = time.time(), {}, 0
        for b in dl:
            if b is None:
                continue
            rgb, msk, tc, ts = (t.to(dev) for t in b)
            drop = torch.rand(len(msk), 1, 1, 1, device=dev) < MASK_DROP        # sometimes no mask: the pieces have none
            msk = torch.where(drop, torch.ones_like(msk), msk)
            with torch.autocast('cuda', torch.bfloat16, enabled=dev.startswith('cuda')):
                L = losses(net, rgb, msk, tc, ts)
                if pieces is not None:
                    pr, pp, pg = pieces.next()
                    pr = torch.from_numpy(pr).permute(0, 3, 1, 2).float().to(dev) / 255.0
                    L['identity'] = identity_loss(net, pr, torch.ones(len(pr), 1, CH, CW, device=dev),
                                                  torch.from_numpy(pp).to(dev), torch.from_numpy(pg).to(dev))
            loss = L['cloth'] + L['shape'] + (L.get('rank_cloth', 0) + L.get('rank_shape', 0) if rank else 0) + L.get('rank_fused', 0) + W_ID * L.get('identity', 0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            for k, v in L.items():
                agg[k] = agg.get(k, 0.0) + float(v.detach())
            n += 1
        rec = {'epoch': epoch, 'minutes': round((time.time() - t0) / 60, 2), 'train': {k: round(v / max(1, n), 4) for k, v in agg.items()},
               'heldout': evaluate(net, dv, dev)}
        log.write(json.dumps(rec) + '\n'); log.flush()
        torch.save({'state': net.state_dict(), 'teacher_dims': frames.teacher_dims, 'epoch': epoch}, str(out))


def load(path, dev):
    import torch
    ck = torch.load(path, map_location='cpu')
    net = make(None, dev, tuple(ck['teacher_dims']))
    net.load_state_dict(ck['state'])
    return net.to(dev).eval()


def door(out_path, slot_ckpt, days=('20260917', '20260918', '20260919')):
    """The owner's door pairs: the slot model finds the event's person on the full snapshot; the crop and
    the mask come from the full 2560x1440 frame. Same scoring as reid_eval_shop.door_pairs."""
    import cv2
    import torch
    import torch.nn.functional as F
    import door_review
    import slot_data
    import eval_slot_heads as E
    from slot_model import SlotModel
    dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    frames = slot_data.Frames()
    ck = torch.load(slot_ckpt, map_location='cpu')
    model = SlotModel(ck.get('backbone', 'deimv2_vit_tiny'), pretrained=False, mean=frames.mean, std=frames.std, teacher_dims=frames.teacher_dims)
    model.load_state_dict(ck.get('ema', ck['model']), strict=False)
    model = model.to(dev).eval()
    net = load(out_path, dev)
    res = {}
    for day in days:
        clean = json.load(open(ROOT / 'data' / 'door_review' / ('%s_clean.json' % day), encoding='utf-8'))['visits']
        events = {e['event_id']: e for e in door_review.day_events(day, ROOT)}
        ids = sorted({v['entry'] for v in clean} | {v['exit'] for v in clean})
        bgs = E.backgrounds(day, 'cam1')
        vec, miss = {}, 0
        for i in ids:
            e = events[i]
            import datetime
            lt = datetime.datetime.fromisoformat(e['time_local'])
            sec = lt.hour * 3600 + lt.minute * 60 + lt.second
            bgp = min(bgs, key=lambda b: abs(b[0] + 450 - sec))[1] if bgs else None
            s = slot_data.load(frames, 'door_' + i, False, image=e['snapshot_full'], label=None, background=bgp, cam_day=(day, 'cam1'))
            out = E.run(model, s, dev)
            ppl = E.people(model, out)
            k = 0.5 * slot_data.W / slot_data.SW
            eb = [e['box_x1'] * k, e['box_y1'] * k, e['box_x2'] * k, e['box_y2'] * k]
            best, bi = None, 0.0
            for pp in ppl:
                cx, cy, w, h = pp[2]
                pb = [(cx - w / 2) * slot_data.W, (cy - h / 2) * slot_data.H, (cx + w / 2) * slot_data.W, (cy + h / 2) * slot_data.H]
                v = E.box_iou(eb, pb)
                if v > bi:
                    best, bi = pp, v
            if best is None or bi < 0.3:
                miss += 1; vec[i] = None; continue
            full = cv2.imread(e['snapshot_full'])
            # the model's mask (stride 4 of its 1088x608 frame) and box, back onto the 2560x1440 frame
            m = cv2.resize(best[1].astype(np.float32), (slot_data.W, slot_data.H), interpolation=cv2.INTER_LINEAR)
            m = cv2.resize(m, (full.shape[1], full.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
            cx, cy, w, h = best[2]
            sx, sy = full.shape[1], full.shape[0]
            box = [(cx - w / 2) * sx, (cy - h / 2) * sy, (cx + w / 2) * sx, (cy + h / 2) * sy]
            c, mm = crop(full, m, box)
            if c is None:
                miss += 1; vec[i] = None; continue
            rgb = torch.from_numpy(np.ascontiguousarray(c[:, :, ::-1]))[None].permute(0, 3, 1, 2).float().to(dev) / 255.0
            with torch.no_grad():
                a, b_ = net(rgb, torch.from_numpy(mm)[None, None].float().to(dev))
                fu = F.normalize(net.fused(a, b_), dim=1)[0].cpu().numpy()
            vec[i] = (F.normalize(a, dim=1)[0].cpu().numpy(), F.normalize(b_, dim=1)[0].cpu().numpy(), fu)
        t = {i: events[i]['unix_ms'] / 1000 for i in ids}
        day_res = {'exits': len(clean), 'person_not_found': miss}
        for name, wc in list(E.MIXES.items()) + [('fused', None)]:
            right = hard = hard_right = scored = 0
            for v in clean:
                if vec[v['exit']] is None or vec[v['entry']] is None:
                    continue
                inside = [w['entry'] for w in clean if t[w['entry']] < t[v['exit']] <= t[w['exit']] + 0.5 and vec[w['entry']] is not None]
                ex = vec[v['exit']]
                if wc is None:
                    d = {e_: 1 - float(ex[2] @ vec[e_][2]) for e_ in inside}
                else:
                    d = {e_: 1 - (wc * float(ex[0] @ vec[e_][0]) + (1 - wc) * float(ex[1] @ vec[e_][1])) for e_ in inside}
                best_e = min(d, key=d.get)
                scored += 1; right += best_e == v['entry']
                if len(inside) > 1:
                    hard += 1; hard_right += best_e == v['entry']
            day_res[name] = {'scored': scored, 'right_first': right, 'crowded': hard, 'crowded_right_first': hard_right}
        res[day] = day_res
    res['all'] = {name: {k: sum(res[d][name][k] for d in days) for k in ('scored', 'right_first', 'crowded', 'crowded_right_first')} for name in list(E.MIXES) + ['fused']}
    json.dump(res, open(ROOT / 'data' / 'logs' / ('crop_reid_door_%s.json' % Path(out_path).stem), 'w'), indent=1)
    return res


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['train', 'door'])
    ap.add_argument('out')
    ap.add_argument('slot', nargs='?')
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--rank', type=int, default=1, help='order-of-likeness losses for clothes and shape (0: cosine to the teachers only)')
    ap.add_argument('--identity', type=int, default=1, help='the same-person lesson on 17.09 track pieces')
    a = ap.parse_args()
    if a.cmd == 'train':
        train(Path(a.out), a.epochs, workers=a.workers, rank=bool(a.rank), piece_days=('20260917',) if a.identity else ())
    else:
        print(json.dumps(door(a.out, a.slot)['all'], indent=1))
