"""The small SAM's detector taught end to end on SAM 3.1's own results (07.10.2026).

Stage 2 of the small SAM, done right after the shadow copies failed (each short stack copied the teacher's inner output
to cosine 0.94-0.98 and the detector still found nobody: the average is ruled by the background and nothing watched
the score). Here the whole detector runs as SAM 3.1 runs it -- our encoder (the student trunk) -> SAM 3.1's detector
neck -> its encoder shortened 6 -> 4 layers -> its decoder 6 -> 4 -> its box, score (with presence) and mask heads --
on its own inputs, and only what comes out is compared with what SAM 3.1 gave on that frame: the people of the window
(data/sam31_seg, data/sam31_door chunks.npz -- SAM 3.1's final masks and boxes). DETR's recipe: Hungarian matching of
the queries to the teacher's people (score, box L1, GIoU), then a focal loss on every query's joint score, L1 + GIoU
on the boxes and BCE + Dice on the masks of the matched ones. Everything on the way learns: the trunk (lower rate),
the detector neck, the shortened stacks, the heads. The tracker is untouched (stage 3).

23.09 (the check day) and the door's 18.09 are never seen. Every CHECK_MIN minutes: the whole small SAM against SAM 3.1
on 23.09 (sam31_lite_eval.py with the trunk and heads saved here).

usage (venv_sam3, the card): sam31_e2e_det.py OUT STUDENT_CKPT [STOP HH:MM]  -> runs/OUT/{last.pt, heads.pt, progress.md}"""
import dataclasses
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
KEEP = {'det_enc': [0, 2, 3, 5], 'det_dec': [0, 2, 3, 5]}
SIZE = 1008
CHECK_MIN = 60
W_CLS, W_L1, W_GIOU, W_BCE, W_DICE = 2.0, 5.0, 2.0, 5.0, 5.0


def say(run, text):
    line = time.strftime('%d.%m %H:%M  ') + text
    with open(run / 'progress.md', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode(), flush=True)


# ------------------------------------------------------------------ boxes

def xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1)


def giou(a, b):
    """a: N x 4, b: M x 4 (xyxy) -> N x M."""
    lt, rb = torch.max(a[:, None, :2], b[None, :, :2]), torch.min(a[:, None, 2:], b[None, :, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2:] - x[:, :2]).clamp(min=0).prod(-1)
    union = area(a)[:, None] + area(b)[None] - inter
    iou = inter / union.clamp(min=1e-6)
    lt2, rb2 = torch.min(a[:, None, :2], b[None, :, :2]), torch.max(a[:, None, 2:], b[None, :, 2:])
    hull = (rb2 - lt2).clamp(min=0).prod(-1)
    return iou - (hull - union) / hull.clamp(min=1e-6)


# ------------------------------------------------------------------ frames and the teacher's people

def sources():
    import sam31_distill as SD
    out = []
    for kind in ('sam31_seg', 'sam31_door'):
        for d in sorted((ROOT / 'data' / kind).glob('*/cam*')):
            name = d.parent.name
            if name.startswith(SD.TEST_DAY) or (kind == 'sam31_door' and '20260918' in name):
                continue
            if (d / 'video.mp4').exists() and (d / 'chunks.npz').exists():
                out.append(d)
    return out


class Frames:
    """A random frame with the teacher's people on it: (RGB 1008 x 1008 uint8, boxes cxcywh 0..1, masks bool H x W)."""

    def __init__(self, dirs, seed=0):
        import sam31_reid as R
        self.rng = random.Random(seed)
        self.items = []
        for d in dirs:
            try:
                M = R.Masks(d / 'chunks.npz', mmap=True)
            except Exception:
                continue
            ticks = M.rows[:, 1].astype(int)
            by = {}
            for i, k in enumerate(ticks):
                by.setdefault(int(k), []).append(i)
            if by:                                                # a window with nobody in it gives nothing to learn
                self.items.append((d, M, by, sorted(by)))
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
        import cv2
        while True:
            d, M, by, ks = self.rng.choice(self.items)
            k = self.rng.choice(ks)
            f = self.frame(d, k)
            if f is None:
                continue
            H, W = f.shape[:2]
            boxes, masks = [], []
            for r in by[k]:
                x1, y1, x2, y2 = [float(v) for v in M.rows[r, 4:8]]
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                m = np.zeros((H, W), bool)
                c = M.crop(r)
                m[int(y1):int(y1) + c.shape[0], int(x1):int(x1) + c.shape[1]] = c[:H - int(y1), :W - int(x1)]
                boxes.append([(x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H])
                masks.append(m)
            img = cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1]
            return (np.ascontiguousarray(img), np.array(boxes, np.float32).reshape(-1, 4),
                    np.stack(masks) if masks else np.zeros((0, H, W), bool))


# ------------------------------------------------------------------ SAM 3.1's detector, end to end

def capture(pred):
    """One real call of the detector on a real frame: its prompt (the text 'person'), its find input and the tensor
    its backbone gets, so the training forward is the same call."""
    import cv2
    import sam31_lite_eval as SL
    det = pred.model.detector
    got = {}
    fg, fi = det.forward_grounding, det.backbone.forward_image

    def grab(backbone_out, find_input, find_target, geometric_prompt, **kw):
        if 'find_input' not in got:
            got.update(find_input=find_input, prompt=geometric_prompt,
                       text={k: v for k, v in backbone_out.items()
                             if k not in ('backbone_fpn', 'vision_pos_enc', 'id_mapping', 'img_batch_all_stages',
                                          'vision_features', 'sam2_backbone_out', 'interactive', 'propagation')})
        return fg(backbone_out=backbone_out, find_input=find_input, find_target=find_target, geometric_prompt=geometric_prompt, **kw)

    def grab_img(img, *a, **kw):
        got.setdefault('image', img.detach().float().cpu()[:1])
        return fi(img, *a, **kw)

    d = sources()[0]
    tmp = ROOT / 'data' / 'logs' / 'e2e_capture'
    tmp.mkdir(parents=True, exist_ok=True)
    for p in tmp.glob('*.jpg'):
        p.unlink()
    cap = cv2.VideoCapture(str(d / 'video.mp4'))
    for k in range(2):
        ok, f = cap.read()
        cv2.imwrite(str(tmp / ('%05d.jpg' % k)), cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 98])
    cap.release()
    det.forward_grounding, det.backbone.forward_image = grab, grab_img
    try:
        SL.run(pred, tmp)
    finally:
        det.forward_grounding, det.backbone.forward_image = fg, fi
    # how the frame becomes the backbone's tensor: try the usual normalisations, keep the closest
    rgb = torch.from_numpy(cv2.imread(str(tmp / '00000.jpg'))[:, :, ::-1].copy()).permute(2, 0, 1).float()[None] / 255
    cands = {'half': (rgb - 0.5) / 0.5,
             'imagenet': (rgb - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
             'unit': rgb}
    im = got['image']
    errs = {k: float((v - im).abs().mean()) for k, v in cands.items() if v.shape == im.shape}
    got['norm'] = min(errs, key=errs.get)
    got['norm_errs'] = errs
    return got


def normal(x):
    """Captured inside SAM's inference mode -> ordinary tensors (an inference tensor cannot be saved for backward:
    the learned prompt is indexed by the find input's text ids)."""
    if torch.is_tensor(x):
        return x.clone() if x.is_inference() else x
    if isinstance(x, dict):
        return {k: normal(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(normal(v) for v in x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.replace(x, **{f.name: normal(getattr(x, f.name)) for f in dataclasses.fields(x) if f.init})
    if hasattr(x, '__dict__'):
        for k, v in list(vars(x).items()):
            try:
                setattr(x, k, normal(v))
            except Exception:
                pass
    return x


def to_input(img, norm, dev):
    x = torch.from_numpy(img).to(dev).permute(2, 0, 1).float()[None] / 255
    if norm == 'half':
        return (x - 0.5) / 0.5
    if norm == 'imagenet':
        return (x - torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    return x


def forward(det, cap, x):
    fi = cap['find_input']
    ids = torch.zeros(1, dtype=torch.long, device=x.device)
    fi = dataclasses.replace(fi, img_ids=ids) if dataclasses.is_dataclass(fi) else fi
    if not dataclasses.is_dataclass(fi):
        fi.img_ids = ids
    backbone_out = {**cap['text'], 'img_batch_all_stages': x, **det.backbone.forward_image(x)}
    return det.forward_grounding(backbone_out=backbone_out, find_input=fi, find_target=None, geometric_prompt=cap['prompt'])


def losses(out, boxes, masks):
    """out: the detector's outputs for one frame; boxes: N x 4 cxcywh; masks: N x H x W (bool, full frame)."""
    from scipy.optimize import linear_sum_assignment
    logit = out['pred_logits'][0, :, 0].float()                       # Q joint score (with presence)
    pb = out['pred_boxes'][0].float()                                  # Q x 4 cxcywh
    pm = out['pred_masks'][0].float()                                  # Q x h x w logits
    Q = logit.shape[0]
    tgt = torch.zeros(Q, device=logit.device)
    st = {}
    if len(boxes):
        tb = torch.from_numpy(boxes).to(logit.device)
        with torch.no_grad():
            C = (-W_CLS * logit.sigmoid()[:, None] + W_L1 * torch.cdist(pb, tb, p=1) - W_GIOU * giou(xyxy(pb), xyxy(tb)))
            qi, ti = linear_sum_assignment(C.cpu().numpy())
        qi, ti = torch.as_tensor(qi, device=logit.device), torch.as_tensor(ti, device=logit.device)
        tgt[qi] = 1
        l_l1 = F.l1_loss(pb[qi], tb[ti], reduction='sum') / len(ti)
        l_giou = (1 - torch.diag(giou(xyxy(pb[qi]), xyxy(tb[ti])))).mean()
        tm = torch.from_numpy(masks).to(logit.device)[ti.cpu().numpy()].float()[:, None]
        tm = F.interpolate(tm, size=pm.shape[-2:], mode='area')[:, 0]
        sm = pm[qi]
        l_bce = F.binary_cross_entropy_with_logits(sm, (tm > 0.5).float())
        p = sm.sigmoid().flatten(1); t = (tm > 0.5).float().flatten(1)
        l_dice = (1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).mean()
        st.update(l1=float(l_l1.detach()), giou=float(l_giou.detach()), bce=float(l_bce.detach()), dice=float(l_dice.detach()))
        box_mask = W_L1 * l_l1 + W_GIOU * l_giou + W_BCE * l_bce + W_DICE * l_dice
    else:
        box_mask = logit.sum() * 0
    pr = logit.sigmoid()                                              # sigmoid focal loss, every query
    ce = F.binary_cross_entropy_with_logits(logit, tgt, reduction='none')
    pt = pr * tgt + (1 - pr) * (1 - tgt)
    l_cls = (ce * (1 - pt) ** 2 * (0.25 * tgt + 0.75 * (1 - tgt))).sum() / max(1, len(boxes))
    st['cls'] = float(l_cls.detach())
    st['n'] = len(boxes)
    return W_CLS * l_cls + box_mask, st


def save(run, student, det, step, arch_src, cap=None):
    import sam31_heads as SH
    stacks = SH.stacks_det(det) if hasattr(SH, 'stacks_det') else {'det_enc': det.transformer.encoder, 'det_dec': det.transformer.decoder}
    torch.save({'keep': KEEP, 'step': step, 'state': {k: v.state_dict() for k, v in stacks.items()},
                'detector_rest': {k: v for k, v in det.state_dict().items()
                                  if not k.startswith('transformer.encoder.') and not k.startswith('transformer.decoder.')
                                  and not k.startswith('backbone.vision_backbone.trunk.') and 'language' not in k and 'text' not in k},
                'prompt': ({cap['prompt_keys'][n]: cap['text'][cap['prompt_keys'][n]].detach().cpu() for n in cap['prompt_keys']}
                           if cap and cap.get('prompt_keys') else None)},
               run / 'heads.pt')
    ck = torch.load(arch_src, map_location='cpu', weights_only=False)
    ck = {'arch': ck['arch'], 'scale': ck.get('scale'), 'step': step, 'model': student.state_dict(), 'ema': student.state_dict()}
    torch.save(ck, run / 'last.pt')


def check(run, tag):
    py = ROOT.parent / 'venv_sam3' / 'Scripts' / 'python.exe'
    ev = run / 'evals'
    ev.mkdir(exist_ok=True)
    out = ev / ('%s.json' % tag)
    r = subprocess.run([str(py if py.exists() else sys.executable), str(ROOT / 'sam31_lite_eval.py'), str(run / 'last.pt'), '2', '96',
                        '--cache', str(ROOT / 'runs' / 's31micro_a' / 'evals' / 'teacher_cache'), '--out', str(out),
                        '--preview', str(ev / ('%s.jpg' % tag)), '--film', 'none', '--heads', str(run / 'heads.pt')],
                       capture_output=True, text=True, cwd=str(ROOT), env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    if r.returncode != 0 or not out.exists():
        return {'error': (r.stderr or r.stdout)[-400:]}
    return json.load(open(out))['mean']


def main(out, student_ckpt, stop='08:30'):
    import sam31_distill as SD
    import sam31_heads as SH
    import sam31_lite_eval as SL
    import sam31_segment as S
    run = ROOT / 'runs' / out
    run.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    torch.backends.cuda.matmul.allow_tf32 = True
    pred = S.build()
    SL.tune(pred)
    det = pred.model.detector
    tri = det.backbone.vision_backbone
    cap = capture(pred)                                     # with the teacher's trunk and stacks, as SAM 3.1 runs
    for k in ('find_input', 'prompt', 'text'):
        cap[k] = normal(cap[k])
    student = SD.load_student(student_ckpt).to(dev)
    tri.trunk = student
    full = {'det_enc': det.transformer.encoder, 'det_dec': det.transformer.decoder}
    det.transformer.encoder = SH.shorten(full['det_enc'], KEEP['det_enc']).to(dev)
    det.transformer.decoder = SH.shorten(full['det_dec'], KEEP['det_dec']).to(dev)
    del full
    if os.environ.get('RA_E2E_INIT_HEADS'):                      # 07.10: go on from a previous run's stacks, neck and heads
        hk = torch.load(os.environ['RA_E2E_INIT_HEADS'], map_location='cpu', weights_only=False)
        det.transformer.encoder.load_state_dict(hk['state']['det_enc'])
        det.transformer.decoder.load_state_dict(hk['state']['det_dec'])
        if hk.get('detector_rest'):
            det.load_state_dict(hk['detector_rest'], strict=False)
        if hk.get('prompt'):                                     # the learned 'person' prompt of that run
            for k, v in hk['prompt'].items():
                if k in cap['text'] and torch.is_tensor(cap['text'][k]) and cap['text'][k].shape == v.shape:
                    cap['text'][k] = v.to(dev, cap['text'][k].dtype)
    torch.cuda.empty_cache()
    det.eval()                                               # SAM 3.1's own inference path (joint score, no o2m)
    for p in det.parameters():
        p.requires_grad_(False)
    # 07.10, run b: run a (1e-4 on the stacks, 2e-5 on the trunk) went down until step ~500 and then up -- lower,
    # with a warm-up and a cosine to 10 % by the stop
    # 07.10: frame only -- the text 'person' becomes a learned prompt (starts at its text tensors, trained with the rest)
    prompt = torch.nn.ParameterDict()
    if os.environ.get('RA_E2E_PROMPT', '1') == '1':
        for k, v in list(cap['text'].items()):
            if torch.is_tensor(v) and v.is_floating_point():
                prompt[k.replace('.', '_')] = torch.nn.Parameter(v.detach().float().clone().to(dev))
                cap['text'][k] = prompt[k.replace('.', '_')]
    cap['prompt_keys'] = {k.replace('.', '_'): k for k in cap['text'] if k.replace('.', '_') in prompt}
    groups = [{'params': list(prompt.parameters()), 'lr': 5e-5},
              {'params': list(student.parameters()), 'lr': 1e-5},
              {'params': list(det.transformer.parameters()), 'lr': 5e-5},
              {'params': [p for n, p in tri.named_parameters() if n.startswith('convs.')], 'lr': 2e-5},
              {'params': [p for n, p in det.named_parameters() if n.startswith('segmentation_head.')], 'lr': 2e-5}]
    scale = float(os.environ.get('RA_E2E_LR_SCALE', '1'))      # 07.10: a resumed run goes on gentler
    for g in groups:
        for p in g['params']:
            p.requires_grad_(True)
        g['lr'] *= scale
        g['base'] = g['lr']
    groups = [g for g in groups if g['params']]
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    n_train = sum(p.numel() for g in groups for p in g['params'])
    data = Frames(sources())
    say(run, 'старт: детектор маленького SAM учится целиком на результатах SAM 3.1 (рамки, уверенности, маски людей '
             'окна), кодировщик + энкодер детектора %s + декодер %s; учится %.1f млн параметров; нормировка кадра %s %s; '
             'окон %d; стоп %s; проверка на 23.09 каждые %d мин'
        % (KEEP['det_enc'], KEEP['det_dec'], n_train / 1e6, cap['norm'], json.dumps({k: round(v, 4) for k, v in cap['norm_errs'].items()}),
           len(data.items), stop, CHECK_MIN))
    hh, mm = map(int, stop.split(':'))
    t_end = time.time() + ((hh * 60 + mm) - (time.localtime().tm_hour * 60 + time.localtime().tm_min)) % 1440 * 60
    step, accum, t_check, t0 = 0, 8, time.time(), time.time()
    t_start, best = time.time(), None
    WARM = 200
    smoke = int(os.environ.get('RA_E2E_SMOKE', '0'))           # N steps, print the losses, quit (a check of the wiring)
    run_st = {}
    while time.time() < t_end:
        opt.zero_grad(set_to_none=True)
        for _ in range(accum):
            img, boxes, masks = data.sample()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out_ = forward(det, cap, to_input(img, cap['norm'], dev))
            loss, st = losses(out_, boxes, masks)
            if not torch.isfinite(loss):
                run_st['skipped'] = run_st.get('skipped', 0) + 1
                continue
            (loss / accum).backward()
            for k, v in st.items():
                run_st[k] = v if k not in run_st else 0.98 * run_st[k] + 0.02 * v
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g['params']], 1.0)
        frac = min(1.0, (time.time() - t_start) / max(1.0, t_end - t_start))
        f = min(1.0, (step + 1) / WARM) * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * frac)))
        for g in groups:
            g['lr'] = g['base'] * f
        opt.step()
        step += 1
        if step % 250 == 0 and not smoke:                      # keep the latest and the best (by the running loss)
            save(run, student, det, step, student_ckpt, cap)
            score = run_st.get('cls', 9) * W_CLS + run_st.get('giou', 9) * W_GIOU + run_st.get('dice', 9) * W_DICE
            if best is None or score < best:
                best = score
                import shutil
                shutil.copy(run / 'last.pt', run / 'best_last.pt'); shutil.copy(run / 'heads.pt', run / 'best_heads.pt')
        if smoke:
            print('smoke step', step, json.dumps({k: round(v, 4) for k, v in run_st.items()}),
                  'gpu GB %.1f' % (torch.cuda.max_memory_allocated() / 1e9), '%.1f s' % (time.time() - t0), flush=True)
            if step >= smoke:
                os._exit(0)
            continue
        if step % 50 == 0:
            with open(run / 'log.jsonl', 'a') as f:
                f.write(json.dumps({'step': step, 'updated': time.strftime('%H:%M:%S'),
                                    'frames_per_s': round(step * accum / (time.time() - t0), 2),
                                    **{k: round(v, 4) for k, v in run_st.items()}}) + '\n')
        if step == 1 or time.time() - t_check > CHECK_MIN * 60:
            save(run, student, det, step, student_ckpt, cap)
            if step > 1:
                del out_
                torch.cuda.empty_cache()
                m = check(run, 'step%05d' % step)
                say(run, 'шаг %d: маленький SAM целиком против SAM 3.1 на 23.09: %s; потери %s'
                    % (step, json.dumps(m), json.dumps({k: round(v, 3) for k, v in run_st.items()})))
            t_check = time.time()
    save(run, student, det, step, student_ckpt, cap)
    m = check(run, 'final')
    say(run, 'конец, шаг %d: %s' % (step, json.dumps(m)))
    os._exit(0)


if __name__ == '__main__':
    main(*sys.argv[1:])
