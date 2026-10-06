"""The small encoder taught by SAM 3.1's answers, not only by its features (06.10.2026; the owner: "the features are a
hint, the answers are the requirement").

Each frame goes through the teacher (ViT-L + SAM 3.1's detector) and through the student (the small encoder + the
very same detector, frozen); the gradient runs back through the detector into the encoder. The detector's 200 query
slots start from the same learned queries in both, so slot i of the student answers for slot i of the teacher:
- requirement: every slot's score (BCE to the teacher's probability), the presence logit; for the slots the teacher
  believes (p >= 0.3, weighted by p) the box (L1 + GIoU) and the mask (BCE to the teacher's mask probability + dice);
- hint (weight HINT): the trunk map and the necks' 1x/2x maps, as in sam31_distill.py.
Alignment is watched: of the teacher's people, how many have the student's box in the same slot at IoU >= 0.5.

usage (venv_sam3, the card): sam31_distill_det.py OUT INIT_CKPT [STOP HH:MM] [--batch B]  -> runs/OUT/..."""
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import sam31_distill as SD                                   # noqa: E402

HINT = float(os.environ.get('RA_DET_HINT', '0.2'))
W_SCORE, W_BOX, W_GIOU, W_MASK, W_DICE, W_PRES = 4.0, 5.0, 2.0, 4.0, 2.0, 1.0
P_MIN = 0.3
EVAL_EVERY = 10 ** 9           # the full-pipeline check only at the end (a second SAM in a subprocess crawled)


def box_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou(a, b):
    """a, b: (..., 4) xyxy; elementwise generalised IoU."""
    lt, rb = torch.max(a[..., :2], b[..., :2]), torch.min(a[..., 2:], b[..., 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[..., 2:] - x[..., :2]).clamp(min=0).prod(-1)
    union = area(a) + area(b) - inter
    iou = inter / union.clamp(min=1e-6)
    LT, RB = torch.min(a[..., :2], b[..., :2]), torch.max(a[..., 2:], b[..., 2:])
    hull = (RB - LT).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-6), iou


class Detector:
    """SAM 3.1's detector (from the predictor), its text prompt computed once."""

    def __init__(self):
        import sam31_segment as S
        pred = S.build()
        self.det = pred.model.detector
        self.det.eval()
        for p in self.det.parameters():
            p.requires_grad_(False)
        self.det.num_interactive_steps_val = 0
        self.seg_head = self.det.segmentation_head
        if os.environ.get('RA_DET_MASKS', '0') != '1':
            self.det.segmentation_head = None                   # no masks on the answer path: scores, boxes, presence
        self.tri = self.det.backbone.vision_backbone
        self.teacher_trunk = self.tri.trunk
        del pred
        with torch.no_grad():
            self.text = self.det.backbone.forward_text(['person'], device='cuda')

    def run(self, x, trunk):
        """x: (B,3,1008,1008) in [-1,1]; trunk: the encoder to use. Returns (out, trunk map)."""
        B = x.shape[0]
        self.tri.trunk = trunk
        bo = {'img_batch_all_stages': x}
        bo.update(self.det.backbone.forward_image(x, need_interactive_out=False, need_propagation_out=False))
        bo.update(self.text)
        self.tri.trunk = self.teacher_trunk
        fi = SimpleNamespace(img_ids=torch.arange(B, device=x.device), text_ids=torch.zeros(B, dtype=torch.long, device=x.device))
        out = self.det.forward_grounding(backbone_out=bo, find_input=fi, find_target=None,
                                         geometric_prompt=self.det._get_dummy_prompt(num_prompts=B))
        return out


def answer_losses(o_s, o_t):
    """The teacher's people (p >= P_MIN) paired one to one with the student's slots (Hungarian, cost = box L1 + GIoU
    + score, as DETR itself is trained); a paired slot must give the teacher's probability, its box and (when masks are
    on) its mask; every other slot must say nobody."""
    from scipy.optimize import linear_sum_assignment
    lt = o_t['pred_logits'].float().squeeze(-1)                  # (B, Q)
    ls = o_s['pred_logits'].float().squeeze(-1)
    pt = lt.sigmoid()
    B, Q = ls.shape
    target = torch.zeros_like(ls)
    parts, aligned, n_all = {}, [], 0
    box_l, giou_l, mask_l, dice_l, wsum = 0.0, 0.0, 0.0, 0.0, 0.0
    for b in range(B):
        ti = torch.nonzero(pt[b] >= P_MIN).squeeze(1)
        if not len(ti):
            continue
        bt = o_t['pred_boxes'].float()[b, ti]                      # (n, 4)
        bs = o_s['pred_boxes'].float()[b]                          # (Q, 4)
        with torch.no_grad():
            g, _ = giou(box_xyxy(bs)[:, None], box_xyxy(bt)[None])     # (Q, n)
            cost = (bs[:, None] - bt[None]).abs().sum(-1) * 5 + (1 - g) * 2 - ls[b].sigmoid()[:, None] * 2
            r, c = linear_sum_assignment(cost.cpu().numpy())
        r = torch.as_tensor(r, device=ls.device); c = torch.as_tensor(c, device=ls.device)
        w = pt[b, ti[c]]
        target[b, r] = w
        gg, iou = giou(box_xyxy(bs[r]), box_xyxy(bt[c]))
        box_l = box_l + ((bs[r] - bt[c]).abs().sum(-1) * w).sum()
        giou_l = giou_l + ((1 - gg) * w).sum()
        if 'pred_masks' in o_s and 'pred_masks' in o_t:
            ms = o_s['pred_masks'].float()[b, r]
            mt = o_t['pred_masks'].float()[b, ti[c]].sigmoid()
            bce = F.binary_cross_entropy_with_logits(ms, mt, reduction='none').flatten(1).mean(1)
            sp = ms.sigmoid().flatten(1)
            dice = 1 - (2 * (sp * mt.flatten(1)).sum(1) + 1) / (sp.sum(1) + mt.flatten(1).sum(1) + 1)
            mask_l = mask_l + (bce * w).sum(); dice_l = dice_l + (dice * w).sum()
        wsum = wsum + float(w.sum())
        aligned.append(float(((iou.detach() >= 0.5) & (ls[b, r].sigmoid().detach() >= 0.5)).float().sum()))
        n_all += len(ti)
    parts['score'] = F.binary_cross_entropy_with_logits(ls, target)
    loss = W_SCORE * parts['score']
    if 'presence_logit_dec' in o_t and 'presence_logit_dec' in o_s:
        a, b_ = o_s['presence_logit_dec'].float(), o_t['presence_logit_dec'].float()
        parts['presence'] = F.binary_cross_entropy_with_logits(a, b_.sigmoid())
        loss = loss + W_PRES * parts['presence']
    if wsum > 0:
        parts['box_l1'] = box_l / wsum
        parts['giou'] = giou_l / wsum
        loss = loss + W_BOX * parts['box_l1'] + W_GIOU * parts['giou']
        if torch.is_tensor(mask_l):
            parts['mask_bce'] = mask_l / wsum; parts['dice'] = dice_l / wsum
            loss = loss + W_MASK * parts['mask_bce'] + W_DICE * parts['dice']
    stats = {'teacher_people': float((pt >= 0.5).sum()) / B, 'student_people': float((ls.sigmoid() >= 0.5).sum()) / B,
             'found_train': (sum(aligned) / max(1, float((pt >= 0.5).sum()))) if n_all else None}
    return loss, parts, stats


def main(out, init, stop='12:00', batch=2):
    import faulthandler
    faulthandler.enable(open(ROOT / 'data' / 'logs' / 'fatal_s31det.log', 'a'))
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = 'cuda'
    run = ROOT / 'runs' / out
    run.mkdir(parents=True, exist_ok=True)
    D = Detector()
    ck = torch.load(str(ROOT / init) if not Path(init).is_absolute() else init, map_location='cpu', weights_only=False)
    s = SD.Student(**ck['arch'], pretrained=False)
    s.load_state_dict(ck['model'] if 'model' in ck else ck['ema'])
    s = s.to(dev).to(memory_format=torch.channels_last)
    ema = SD.Student(**ck['arch'], pretrained=False).to(dev).eval().requires_grad_(False)
    ema.load_state_dict(s.state_dict())
    params = [{'params': [p for n, p in s.named_parameters() if n.startswith('body.')], 'lr': 2e-4},
              {'params': [p for n, p in s.named_parameters() if not n.startswith('body.')], 'lr': 5e-4}]
    for g in params:
        g['base'] = g['lr']
    opt = torch.optim.AdamW(params, weight_decay=0.05, fused=True)
    ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    dl = torch.utils.data.DataLoader(SD.Frames(SD.sources()), batch_size=batch, num_workers=4, pin_memory=True,
                                     persistent_workers=True, prefetch_factor=4)
    neck = D.tri                                             # the teacher's necks, for the hint
    test = SD.test_frames(48)
    with torch.no_grad(), ac():
        xs = [SD.norm_input(test[i:i + 2], dev) for i in range(0, 8, 2)]
        tt = [D.teacher_trunk(x)[-1].float() for x in xs]
        tn = [SD.neck_maps(neck, t.to(torch.bfloat16)) for t in tt]
        sd = lambda maps: torch.cat([m.float().permute(1, 0, 2, 3).flatten(1) for m in maps], 1).std(1).clamp_min(1e-3).view(1, -1, 1, 1)
        scale = {'trunk': sd(tt), 'neck': [[sd([n[h][i] for n in tn]) for i in range(len(SD.SCALES))] for h in range(3)]}
    h, mi = map(int, stop.split(':'))
    lt = time.localtime()
    stop_at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, mi, 0, 0, 0, -1))
    if stop_at <= time.time():
        stop_at += 86400
    t_start = time.time()
    span = stop_at - t_start
    log = open(run / 'log.jsonl', 'a')
    SD.say(run, 'старт: кодировщик учится по ОТВЕТАМ детектора SAM 3.1 (уверенность каждого из 200 запросов, рамки, маски, '
                '«есть ли люди») — это требование; признаки — подсказка с весом %.1f. Начало — веса этапа 1 (%s). Стоп в %s; '
                'весь SAM с учеником против SAM 3.1 на 23.09 каждые %d шагов' % (HINT, init, stop, EVAL_EVERY))
    step, seen, t_last, t_save = 0, 0, time.time(), time.time()
    run_parts, run_stats = {}, {}
    best = None

    def save(tag='last'):
        torch.save({'model': s.state_dict(), 'ema': ema.state_dict(), 'arch': s.arch(), 'step': step}, run / (tag + '.pt'))

    hcache = {}
    h0 = heldout(D, ema, test, hcache, dev, ac)
    SD.say(run, 'проверка до обучения на 23.09 (48 кадров, ответы детектора): у учителя %d людей, ученик нашёл %.0f%%, '
                'точность %.0f%%' % (h0['teacher_people'], 100 * h0['found'], 100 * h0['precision']))
    s.train()
    for u8 in dl:
        now = time.time()
        if now >= stop_at or (run / 'STOP').exists():
            break
        prog = min(1.0, (now - t_start) / span)
        f = min(1.0, (step + 1) / 200) * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g['lr'] = g['base'] * f
        x = SD.norm_input(u8, dev)
        with torch.no_grad(), ac():
            t_list = D.teacher_trunk(x)                          # the teacher's map once: for its answers and the hint
            t_map = t_list[-1]
            o_t = D.run(x, _Fixed(t_list))
            t_necks = SD.neck_maps(neck, t_map)
        with ac():
            y = s(x.contiguous(memory_format=torch.channels_last))
            o_s = D.run(x, _Fixed(y))                            # the same detector on the student's map, with its graph
            loss, parts, stats = answer_losses(o_s, o_t)
            s_necks = SD.neck_maps(neck, y[-1])
            hint, hp = SD.losses(y[-1], t_map, s_necks, t_necks, scale)
            loss = loss + HINT * hint
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step) / (10 + step))
            torch._foreach_lerp_(list(ema.parameters()), list(s.parameters()), 1 - d)
            for be, bs_ in zip(ema.buffers(), s.buffers()):
                be.copy_(bs_)
        step += 1
        seen += len(u8)
        parts['hint'] = hint.detach()
        parts['hint_cos_trunk'] = hp['cos_trunk']
        for k, v in parts.items():
            v = float(v)
            run_parts[k] = run_parts[k] * 0.98 + v * 0.02 if k in run_parts else v
        for k, v in stats.items():
            if v is not None:
                run_stats[k] = run_stats[k] * 0.98 + v * 0.02 if k in run_stats else v
        if step % 25 == 0:
            st = {'step': step, 'frames_per_s': round(seen / (time.time() - t_last), 2), 'lr_factor': round(f, 3),
                  'loss': {k: round(v, 4) for k, v in run_parts.items()}, 'answers': {k: round(v, 3) for k, v in run_stats.items()},
                  'gpu_gb': round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), 'updated': time.strftime('%H:%M:%S'),
                  'eta_stop_min': round((stop_at - time.time()) / 60)}
            seen, t_last = 0, time.time()
            json.dump(st, open(run / 'status.json', 'w'), indent=1)
            if step % 500 == 0:
                h = heldout(D, ema, test, hcache, dev, ac)
                s.train()
                st['heldout'] = h
                SD.say(run, 'шаг %d, 23.09 (не видел): ученик нашёл %.0f%% людей учителя, точность %.0f%%' % (
                    step, 100 * h['found'], 100 * h['precision']))
                SD.say(run, 'шаг %d: на обучающих кадрах учитель видит %.1f чел./кадр, ученик %.1f; из людей учителя ученик нашёл '
                            '(своя рамка IoU>=0.5 и уверен) %.0f%%; рамки GIoU-потеря %.3f; подсказка: похожесть карты %.3f; %.2f кадра/с' % (
                                step, run_stats.get('teacher_people', 0), run_stats.get('student_people', 0),
                                100 * run_stats.get('found_train', 0), run_parts.get('giou', 0), run_parts.get('hint_cos_trunk', 0),
                                st['frames_per_s']))
                log.write(json.dumps(st) + '\n'); log.flush()
            if step % EVAL_EVERY == 0:
                pc = check(run, step, ema, s, D)
                if 'found' in pc:
                    SD.say(run, 'шаг %d: SAM 3.1 с учеником против SAM 3.1 на 23.09: найдено %.0f%% людей учителя, точность %.0f%%, '
                                'маски %.2f, IDF1 %.2f, смен номера %d' % (step, 100 * pc['found'], 100 * pc['precision'],
                                                                           pc['mask_iou'], pc['idf1'], pc['id_switches']))
                    score = pc['found'] + pc['idf1']
                    if best is None or score > best:
                        best = score
                        save('best')
                else:
                    SD.say(run, 'шаг %d: проверка упала: %s' % (step, pc.get('pipeline_error', '')[-300:]))
        if time.time() - t_save > 900:
            save(); t_save = time.time()
    save()
    pc = check(run, step, ema, s, D)
    SD.say(run, 'конец, шаг %d: %s' % (step, json.dumps(pc, ensure_ascii=False)))
    os._exit(0)


def heldout(D, net, test, cache, dev, ac):
    """The detector's answers on 48 frames of 23.09 (never trained on): of the teacher's people (p >= 0.5), how many
    the student also gives (a confident slot, Hungarian-paired box IoU >= 0.5), and how many of the student's confident
    slots are the teacher's people. In this process, no second SAM (the full-pipeline check is left to the end)."""
    from scipy.optimize import linear_sum_assignment
    found = tp_s = n_t = n_s = 0
    net.eval()
    with torch.no_grad(), ac():
        for i in range(len(test)):
            x = SD.norm_input(test[i:i + 1], dev)
            if i not in cache:
                o = D.run(x, _Fixed(D.teacher_trunk(x)))
                cache[i] = (o['pred_logits'].float().sigmoid()[0, :, 0].cpu(), o['pred_boxes'].float()[0].cpu())
            pt, bt = cache[i]
            o = D.run(x, _Fixed(net(x)))
            ps, bs = o['pred_logits'].float().sigmoid()[0, :, 0].cpu(), o['pred_boxes'].float()[0].cpu()
            T, S_ = bt[pt >= 0.5], bs[ps >= 0.5]
            n_t += len(T); n_s += len(S_)
            if len(T) and len(S_):
                _, iou = giou(box_xyxy(S_)[:, None], box_xyxy(T)[None])
                r, c = linear_sum_assignment(-iou.numpy())
                k = int((iou[r, c] >= 0.5).sum())
                found += k; tp_s += k
    return {'found': found / max(1, n_t), 'precision': tp_s / max(1, n_s), 'teacher_people': n_t, 'student_people': n_s}


def check(run, step, ema, s, D):
    """The full-pipeline check runs in its own process with its own SAM 3.1 (~6 GB): the teacher and the detector of
    this process leave the card meanwhile (on 06.10 the check crawled for 16 min with the card overflowing)."""
    D.det.cpu()
    torch.cuda.empty_cache()
    try:
        return SD.pipeline_check(run, step, ema.state_dict(), s.arch())
    finally:
        D.det.cuda()


class _Fixed(torch.nn.Module):
    """A 'trunk' that returns an already computed map (so the detector runs on the student's map with its graph)."""

    def __init__(self, y):
        super().__init__()
        self.y = y
        self.channel_list = [1024]

    def forward(self, x):
        return self.y


if __name__ == '__main__':
    a = sys.argv[1:]
    kw = {}
    if '--batch' in a:
        i = a.index('--batch'); kw['batch'] = int(a[i + 1]); del a[i:i + 2]
    main(*a, **kw)
