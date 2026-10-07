"""The small SAM's tracker taught on clips of SAM 3.1's own results (07.10.2026), stage 3 of the small SAM.

SAM 3.1 keeps SAM 2's training path in its tracker (VideoTrackingDynamicMultiplex.forward_image ->
prepare_prompt_inputs -> forward_tracking): a clip, the people's masks on every frame and who is visible where; the
masks of the first frame (or a few conditioning frames) prompt it, it carries everybody through its memory over the
rest, new people come in at transition points, and every frame gives a mask and an "is there" score per person.
Here the clip and the masks are SAM 3.1's own: CLIP frames of one session (rows[:, 0]) of a window or a door stretch,
a person = its local id in that session (the same person on every frame by construction). The loss is the tracker's
answer against the teacher's on every non-conditioning frame: BCE + Dice on the masks, BCE on the object score.

The encoder is the night's student (frozen here: eight frames with gradients through it do not fit the 12 GB with
the tracker); what learns is the tracker -- memory attention, memory encoder, mask decoder, object pointers.
23.09 (the check day) and the door's 18.09 are never seen. Every CHECK_MIN minutes: the whole small SAM against
SAM 3.1 on 23.09 (sam31_lite_eval.py, the trunk and detector of DET_RUN, this tracker).

usage (venv_sam3, the card): sam31_e2e_trk.py OUT DET_RUN [STOP HH:MM]  -> runs/OUT/{tracker.pt, progress.md, log.jsonl}
RA_TRK_SMOKE=N: N steps, print, quit."""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
SIZE = 1008
CLIP = int(os.environ.get('RA_TRK_CLIP', '6'))
STEP_TICKS = int(os.environ.get('RA_TRK_STRIDE', '2'))      # 2 ticks = 0.16 s between clip frames
MAX_OBJ = 16
CHECK_MIN = 60


def say(run, text):
    line = time.strftime('%d.%m %H:%M  ') + text
    with open(run / 'progress.md', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        print(line.encode('ascii', 'replace').decode(), flush=True)


class Clips:
    """A random clip of one SAM 3.1 session: frames (T x 3 x 1008 x 1008, uint8 RGB) and the people's masks on each
    (T x N x 1008 x 1008 bool, the same person = the same index on every frame)."""

    def __init__(self, seed=0):
        import sam31_e2e_det as E
        import sam31_reid as R
        self.rng = random.Random(seed)
        self.items = []
        for d in E.sources():
            try:
                M = R.Masks(d / 'chunks.npz', mmap=True)
            except Exception:
                continue
            rows = M.rows
            sess = {}
            for i in range(len(rows)):
                sess.setdefault(int(rows[i, 0]), {}).setdefault(int(rows[i, 1]), []).append(i)
            starts = []
            for s, by in sess.items():
                ks = sorted(by)
                for k in ks:
                    if all((k + j * STEP_TICKS) in by for j in range(CLIP)):
                        starts.append((s, k))
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
        import cv2
        while True:
            d, M, sess, starts = self.rng.choice(self.items)
            s, k0 = self.rng.choice(starts)
            ticks = [k0 + j * STEP_TICKS for j in range(CLIP)]
            frames = [self.frame(d, k) for k in ticks]
            if any(f is None for f in frames):
                continue
            H, W = frames[0].shape[:2]
            ids = sorted({int(M.rows[r, 2]) for k in ticks for r in sess[s][k]})[:MAX_OBJ]
            pos = {o: i for i, o in enumerate(ids)}
            masks = np.zeros((CLIP, len(ids), SIZE, SIZE), bool)
            for t, k in enumerate(ticks):
                for r in sess[s][k]:
                    o = int(M.rows[r, 2])
                    if o not in pos:
                        continue
                    x1, y1 = int(M.rows[r, 4]), int(M.rows[r, 5])
                    c = M.crop(r)
                    m = np.zeros((H, W), np.uint8)
                    m[y1:y1 + c.shape[0], x1:x1 + c.shape[1]] = c[:H - y1, :W - x1]
                    masks[t, pos[o]] = cv2.resize(m, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST) > 0
            imgs = np.stack([np.ascontiguousarray(cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1]) for f in frames])
            if masks[0].any(axis=(1, 2)).sum() == 0:                       # nobody on the first frame: nothing to prompt
                continue
            return imgs, masks


def find_tracker(pred):
    """The real tracker (VideoTrackingMultiplex), wherever SAM 3.1's wrappers keep it: pred.model.tracker is a
    Sam3MultiplexPredictorWrapper on the build used here."""
    t = pred.model.tracker
    t = t.__dict__['_modules'].get('model', t)          # Sam3MultiplexPredictorWrapper.model (it proxies attributes)
    print('tracker class', type(t).__name__, flush=True)
    return t


def make_input(imgs, masks, dev):
    T, N = masks.shape[:2]
    x = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255
    x = (x - 0.5) / 0.5
    seg = torch.from_numpy(masks).to(dev)
    inp = SimpleNamespace(
        img_batch=SimpleNamespace(tensors=x, mask=None),
        find_inputs=[SimpleNamespace(img_ids=torch.tensor([t], device=dev)) for t in range(T)],
        find_targets=[SimpleNamespace(segments=seg[t].float(), num_boxes=torch.ones(N, dtype=torch.long, device=dev))
                      for t in range(T)],
        visible_objects_per_frame={t: {int(i) for i in np.nonzero(masks[t].any(axis=(1, 2)))[0]} for t in range(T)})
    return x, inp


def clip_loss(trk, x, inp):
    # 07.10: no precomputed features -- forward_tracking then computes each frame's own when it gets there (the frozen
    # encoder keeps no graph), instead of holding all frames x three necks at once (14 GB on 8 frames)
    bo = trk.prepare_prompt_inputs({}, inp)
    outs = trk.forward_tracking(bo, inp)
    cond = set(bo['init_cond_frames'])
    total, st, n = 0.0, {'bce': 0.0, 'dice': 0.0, 'obj': 0.0}, 0
    for t, out in enumerate(outs):
        if t in cond:
            continue
        gt = bo['gt_masks_per_frame'][t].float()            # N_t x 1 x H x W
        pm = out.get('pred_masks')                          # low-res logits, N_t x 1 x h x w
        if pm is None or gt.shape[0] == 0 or pm.shape[0] != gt.shape[0]:
            continue
        g = F.interpolate(gt, size=pm.shape[-2:], mode='area')
        tgt = (g > 0.5).float()
        l_bce = F.binary_cross_entropy_with_logits(pm.float(), tgt)
        p = pm.float().sigmoid().flatten(1); q = tgt.flatten(1)
        l_dice = (1 - (2 * (p * q).sum(1) + 1) / (p.sum(1) + q.sum(1) + 1)).mean()
        vis = (gt.flatten(1).sum(1) > 0).float()
        osl = out.get('object_score_logits')
        l_obj = F.binary_cross_entropy_with_logits(osl.float().view(-1)[:len(vis)], vis) if osl is not None else pm.sum() * 0
        total = total + 20 * l_bce + l_dice + l_obj
        st['bce'] += float(l_bce.detach()); st['dice'] += float(l_dice.detach()); st['obj'] += float(l_obj.detach()); n += 1
    if n == 0:
        return None, {}
    return total / n, {k: v / n for k, v in st.items()}


def check(run, det_run, tag):
    py = ROOT.parent / 'venv_sam3' / 'Scripts' / 'python.exe'
    ev = run / 'evals'
    ev.mkdir(exist_ok=True)
    out = ev / ('%s.json' % tag)
    det_run = ROOT / 'runs' / det_run
    r = subprocess.run([str(py if py.exists() else sys.executable), str(ROOT / 'sam31_lite_eval.py'), str(det_run / 'best_last.pt'), '2', '96',
                        '--cache', str(ROOT / 'runs' / 's31micro_a' / 'evals' / 'teacher_cache'), '--out', str(out),
                        '--preview', str(ev / ('%s.jpg' % tag)), '--film', 'none', '--heads', str(det_run / 'best_heads.pt')],
                       capture_output=True, text=True, cwd=str(ROOT),
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', RA_TRACKER_CKPT=str(run / 'tracker.pt')))
    if r.returncode != 0 or not out.exists():
        return {'error': (r.stderr or r.stdout)[-400:]}
    return json.load(open(out))['mean']


def main(out, det_run, stop='20:00'):
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
    dr = ROOT / 'runs' / det_run
    SH.install(pred, torch.load(dr / 'best_heads.pt', map_location='cpu', weights_only=False))
    student = SD.load_student(str(dr / 'best_last.pt')).to(dev).eval()
    det = pred.model.detector
    det.backbone.vision_backbone.trunk = student
    trk = find_tracker(pred)
    if getattr(trk, 'backbone', None) is None:               # SAM 3.1 feeds the tracker from the shared backbone
        if 'backbone' in trk._modules:
            trk._modules['backbone'] = det.backbone
        else:
            object.__setattr__(trk, 'backbone', det.backbone)
    print('tracker', type(trk).__name__, 'backbone', type(trk.backbone).__name__, flush=True)
    for p in pred.model.parameters():
        p.requires_grad_(False)
    names = [n for n, _ in trk.named_parameters() if not n.startswith('backbone.')]
    params = [p for n, p in trk.named_parameters() if not n.startswith('backbone.')]
    for p in params:
        p.requires_grad_(True)
    trk.train()
    trk.backbone.eval()
    for n, m in list(det.named_children()):               # the detector's own parts are not needed by the tracker
        if n != 'backbone':
            m.cpu()
    for n, m in list(det.backbone.named_children()):      # nor the text encoder
        if n != 'vision_backbone':
            m.cpu()
    torch.cuda.empty_cache()
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=1e-4)
    data = Clips()
    say(run, 'старт: трекер маленького SAM учится на клипах SAM 3.1 (%d кадров через %d такта, люди = номера внутри сеанса), '
             'кодировщик и детектор из %s заморожены; учится %.1f млн параметров трекера; окон/отрезков с клипами %d; стоп %s'
        % (CLIP, STEP_TICKS, det_run, sum(p.numel() for p in params) / 1e6, len(data.items), stop))
    hh, mm = map(int, stop.split(':'))
    t_end = time.time() + ((hh * 60 + mm) - (time.localtime().tm_hour * 60 + time.localtime().tm_min)) % 1440 * 60
    smoke = int(os.environ.get('RA_TRK_SMOKE', '0'))
    step, t0, t_check, run_st = 0, time.time(), time.time(), {}
    while time.time() < t_end:
        imgs, masks = data.sample()
        x, inp = make_input(imgs, masks, dev)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss, st = clip_loss(trk, x, inp)
        if loss is None or not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        for k, v in st.items():
            run_st[k] = v if k not in run_st else 0.98 * run_st[k] + 0.02 * v
        if smoke:
            print('smoke step', step, json.dumps({k: round(v, 4) for k, v in st.items()}), 'objects', masks.shape[1],
                  'gpu GB %.1f' % (torch.cuda.max_memory_allocated() / 1e9), '%.1f s' % (time.time() - t0), flush=True)
            if step >= smoke:
                os._exit(0)
            continue
        if step % 25 == 0:
            with open(run / 'log.jsonl', 'a') as f:
                f.write(json.dumps({'step': step, 'updated': time.strftime('%H:%M:%S'),
                                    'clips_per_min': round(step / (time.time() - t0) * 60, 2),
                                    **{k: round(v, 4) for k, v in run_st.items()}}) + '\n')
        if step % 200 == 0:
            torch.save({'step': step, 'names': names, 'state': {n: p.detach().cpu() for n, p in zip(names, params)}}, run / 'tracker.pt')
        if time.time() - t_check > CHECK_MIN * 60:
            torch.save({'step': step, 'names': names, 'state': {n: p.detach().cpu() for n, p in zip(names, params)}}, run / 'tracker.pt')
            torch.cuda.empty_cache()
            say(run, 'шаг %d: потери %s; весь маленький SAM против SAM 3.1 на 23.09: %s'
                % (step, json.dumps({k: round(v, 3) for k, v in run_st.items()}), json.dumps(check(run, det_run, 'step%05d' % step))))
            t_check = time.time()
    torch.save({'step': step, 'names': names, 'state': {n: p.detach().cpu() for n, p in zip(names, params)}}, run / 'tracker.pt')
    say(run, 'конец, шаг %d: %s' % (step, json.dumps(check(run, det_run, 'final'))))
    os._exit(0)


if __name__ == '__main__':
    main(*sys.argv[1:])
