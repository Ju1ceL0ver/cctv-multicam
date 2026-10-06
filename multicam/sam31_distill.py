"""SAM 3.1 with a small image encoder (05.10.2026). SAM 3.1 spends 0.40 of its 0.65 s a frame (3060, 7 people, bf16) in
its image encoder, a ViT-L/14 at 1008 px (32 blocks, 72x72x1024 out); detector heads 0.11, the tracker with its memory
0.10, association and memory encoding 0.03 (sam31_parts.py). EdgeTAM's recipe: a small CNN learns to give the same
72x72x1024 map, and everything after it -- the three necks, the detector, the memory tracker, the hot-start and
re-conditioning rules -- stays SAM 3.1's own. Two fixed cameras are a narrow world, so a small net can copy it closely.

Frames: every window of data/sam31_seg except 23.09 (kept for the tracking test) and every door stretch of
data/sam31_door, squeezed to 1008 x 1008 as SAM saw them; 90 % from ticks with people. The teacher runs on the fly
(no features on disk: 10 MB a frame). Loss: the trunk map and the three necks' maps at every scale (the teacher's own
frozen convs applied to the student's map), each standardised per channel, MSE + (1 - cosine).

usage (venv_sam3, the card): sam31_distill.py OUT [STOP HH:MM] [--init CKPT] [--student NAME]
  -> runs/OUT/{last.pt, status.json, log.jsonl}"""
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
CKPT = ROOT / 'data' / 'weights' / 'sam3' / 'sam3.1_multiplex.pt'
SIZE = 1008
TEST_DAY = '20260923'

import torch                                                    # noqa: E402
import torch.nn as nn                                           # noqa: E402
import torch.nn.functional as F                                 # noqa: E402


# ------------------------------------------------------------------ the student


def _safe_print(line):
    """Windows consoles of background jobs cannot print every character (a '>=' sign killed two runs on 06.10)."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
        print(line.encode(enc, 'replace').decode(enc, 'replace'), flush=True)

class Block(nn.Module):
    """ConvNeXt block: depthwise 7x7, norm, MLP x4."""

    def __init__(self, d):
        super().__init__()
        self.dw = nn.Conv2d(d, d, 7, padding=3, groups=d)
        self.norm = nn.GroupNorm(1, d)
        self.pw1 = nn.Conv2d(d, 4 * d, 1)
        self.pw2 = nn.Conv2d(4 * d, d, 1)
        self.gamma = nn.Parameter(torch.full((1, d, 1, 1), 1e-2))

    def forward(self, x):
        return x + self.gamma * self.pw2(F.gelu(self.pw1(self.norm(self.dw(x)))))


class Student(nn.Module):
    """A timm CNN at 1152 px (stride 16 -> 72 x 72), its strides 8/16/32 fused at 16, a few ConvNeXt blocks, 1x1 to the
    teacher's 1024 channels. Takes SAM's own input (1008, normalised to [-1, 1]) and returns [map] like SAM's trunk."""

    def __init__(self, name='repvit_m1_1', inner=384, blocks=4, size=1152, pretrained=True):
        super().__init__()
        import timm
        self.name, self.inner, self.nblocks, self.size = name, inner, blocks, size
        probe = timm.create_model(name, pretrained=False, features_only=True)
        red = probe.feature_info.reduction()
        idx = tuple(red.index(r) for r in (8, 16, 32))                 # strides 8/16/32, whatever the net's levels
        del probe
        self.body = timm.create_model(name, pretrained=pretrained, features_only=True, out_indices=idx)
        ch = self.body.feature_info.channels()
        self.l8 = nn.Conv2d(ch[0], inner, 1)
        self.l16 = nn.Conv2d(ch[1], inner, 1)
        self.l32 = nn.Conv2d(ch[2], inner, 1)
        self.blocks = nn.Sequential(*[Block(inner) for _ in range(blocks)])
        self.out = nn.Conv2d(inner, 1024, 1)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)
        self.channel_list = [1024]

    def forward(self, x):
        x = getattr(x, 'tensors', x)
        x = F.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        x = ((x * 0.5 + 0.5) - self.mean) / self.std
        f8, f16, f32 = self.body(x)
        y = self.l16(f16) + F.interpolate(self.l32(f32), size=f16.shape[-2:], mode='bilinear', align_corners=False) \
            + F.adaptive_avg_pool2d(self.l8(f8), f16.shape[-2:])
        y = self.out(self.blocks(y))
        if y.shape[-1] != SIZE // 14:
            y = F.interpolate(y, size=(SIZE // 14, SIZE // 14), mode='bilinear', align_corners=False)
        return [y]

    def arch(self):
        return {'name': self.name, 'inner': self.inner, 'blocks': self.nblocks, 'size': self.size}


def load_student(path, device='cuda'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    s = Student(**ck['arch'], pretrained=False)
    s.load_state_dict(ck['ema'] if ck.get('ema') else ck['model'])
    return s.to(device).eval()


# ------------------------------------------------------------------ the teacher

def teacher_neck(device='cuda'):
    """SAM 3.1's vision backbone (trunk + three necks) alone, with its weights."""
    from sam3.model_builder import _create_multiplex_tri_backbone
    neck = _create_multiplex_tri_backbone(use_fa3=False, use_rope_real=True)
    ck = torch.load(str(CKPT), map_location='cpu', weights_only=True)
    ck = ck.get('model', ck) if isinstance(ck.get('model', None), dict) else ck
    pre = 'detector.backbone.vision_backbone.'
    sd = {k[len(pre):]: v for k, v in ck.items() if k.startswith(pre)}
    missing, unexpected = neck.load_state_dict(sd, strict=False)
    missing = [k for k in missing if 'freqs_cis' not in k]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return neck.to(device).eval().requires_grad_(False)


SCALES = (1, 2)            # the necks' 2x and 1x maps; the 4x one (288 x 288, deconvolved from them) overflowed the card


def neck_maps(neck, x, scales=SCALES):
    """The necks' maps (3 heads x the given scales) on a trunk map x."""
    out = []
    for convs in (neck.convs, neck.interactive_convs, neck.propagation_convs):
        out.append([convs[i](x) for i in scales])
    return out


# ------------------------------------------------------------------ frames

def sources(test=False):
    """[(video, ticks with people, all ticks)]: windows (no 23.09 unless test) and door stretches (never in test)."""
    out = []
    kinds = ['sam31_seg'] if test else ['sam31_seg', 'sam31_door']
    for kind in kinds:
        for d in sorted((ROOT / 'data' / kind).glob('*/cam*')):
            v, c = d / 'video.mp4', d / 'chunks.npz'
            if not v.exists() or not c.exists():
                continue
            is_test = d.parent.name.startswith(TEST_DAY)
            if is_test != test:
                continue
            try:
                ticks = np.unique(np.load(c)['rows'][:, 1].astype(int))
                n = len(json.load(open(d / 'ticks.json')))
            except Exception:
                continue
            out.append((str(v), ticks, n))
    return out


def to_input(bgr):
    import cv2
    img = cv2.resize(bgr, (SIZE, SIZE), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)          # uint8 3xHxW


class Frames(torch.utils.data.IterableDataset):
    def __init__(self, srcs, seed=0):
        self.srcs, self.seed = srcs, seed

    def __iter__(self):
        import cv2
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi else 0) * 7919 + int(time.time()))
        caps = {}
        weights = np.array([max(1, len(t)) for _, t, _ in self.srcs], float)
        weights = (weights / weights.sum()).tolist()
        while True:
            v, ticks, n = rng.choices(self.srcs, weights)[0]
            k = int(rng.choice(ticks)) if (len(ticks) and rng.random() < 0.9) else rng.randrange(max(1, n))
            cap = caps.get(v)
            if cap is None:
                if len(caps) >= 6:
                    caps.pop(next(iter(caps))).release()
                cap = caps[v] = cv2.VideoCapture(v)
            cap.set(cv2.CAP_PROP_POS_FRAMES, k)
            ok, f = cap.read()
            if ok:
                yield to_input(f)


def test_frames(n=64):
    import cv2
    rng = random.Random(1)
    srcs = sources(test=True)
    out = []
    for i in range(n):
        v, ticks, _ = srcs[i % len(srcs)]
        cap = cv2.VideoCapture(v)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(rng.choice(ticks)))
        ok, f = cap.read()
        cap.release()
        if ok:
            out.append(to_input(f))
    return torch.stack(out)


# ------------------------------------------------------------------ training

def norm_input(u8, device):
    return (u8.to(device, non_blocking=True).float() / 255.0 - 0.5) / 0.5


WEIGHTS = [0.5, 1.0]                            # neck scales 2x, 1x


def losses(s_trunk, t_trunk, s_necks, t_necks, scale):
    """Standardised MSE + (1 - cosine) on the trunk map and on every neck map."""
    def one(s, t, sd):
        s, t = s.float(), t.float()
        mse = (((s - t) / sd) ** 2).mean()
        cos = 1 - F.cosine_similarity(s, t, dim=1).mean()
        return mse + cos, cos
    parts = {}
    tot, c_trunk = one(s_trunk, t_trunk, scale['trunk'])
    parts['trunk'] = tot.detach()
    parts['cos_trunk'] = 1 - c_trunk.detach()
    for h, (sn, tn) in enumerate(zip(s_necks, t_necks)):
        for i, (a, b) in enumerate(zip(sn, tn)):
            l, _ = one(a, b, scale['neck'][h][i])
            tot = tot + WEIGHTS[i] * l
            parts['neck%d_%d' % (h, SCALES[i])] = l.detach()
    return tot, parts


EVAL_EVERY = 2000          # steps between full-pipeline checks (SAM 3.1 heads on the student's map vs SAM 3.1)


def say(run, text):
    """One human line into progress.md (what is going on, in words) and the log."""
    line = '%s  %s' % (time.strftime('%d.%m %H:%M'), text)
    with open(run / 'progress.md', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    _safe_print(line)


def pipeline_check(run, step, ema_state, arch):
    """The whole SAM 3.1 with the student's encoder against SAM 3.1 on 23.09 (two crowded clips, 96 ticks each; the
    teacher's answers cached after the first time). Runs as a separate process while training waits."""
    import subprocess
    ev = run / 'evals'
    ev.mkdir(exist_ok=True)
    ck = run / 'eval.pt'
    torch.save({'ema': ema_state, 'arch': arch, 'step': step}, ck)
    torch.cuda.empty_cache()
    out = ev / ('step_%06d.json' % step)
    t0 = time.time()
    py = ROOT.parent / 'venv_sam3' / 'Scripts' / 'python.exe'
    r = subprocess.run([str(py if py.exists() else sys.executable), str(ROOT / 'sam31_lite_eval.py'), str(ck), '2', '96', '--cache', str(ev / 'teacher_cache'),
                        '--out', str(out), '--preview', str(ev / ('step_%06d.jpg' % step)), '--film', 'none'],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0 or not out.exists():
        return {'pipeline_error': (r.stderr or r.stdout)[-600:]}
    m = json.load(open(out))['mean']
    m['pipeline_s'] = round(time.time() - t0)
    return m


def main(out, stop='09:30', init=None, student='repvit_m1_1', batch=2):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = 'cuda'
    run = ROOT / 'runs' / out
    run.mkdir(parents=True, exist_ok=True)
    neck = teacher_neck(dev)
    s = Student(student).to(dev).to(memory_format=torch.channels_last)
    ema = Student(student, pretrained=False).to(dev).eval().requires_grad_(False)
    step0 = 0
    if init:
        ck = torch.load(init, map_location='cpu', weights_only=False)
        s.load_state_dict(ck['model']); step0 = ck.get('step', 0) if init.endswith('last.pt') and Path(init).parent == run else 0
    ema.load_state_dict(s.state_dict())
    params = [{'params': [p for n, p in s.named_parameters() if n.startswith('body.')], 'lr': 4e-4},
              {'params': [p for n, p in s.named_parameters() if not n.startswith('body.')], 'lr': 1e-3}]
    for g in params:
        g['base'] = g['lr']
    opt = torch.optim.AdamW(params, weight_decay=0.05, fused=True)
    ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    dl = torch.utils.data.DataLoader(Frames(sources()), batch_size=batch, num_workers=4, pin_memory=True,
                                     persistent_workers=True, prefetch_factor=4)
    test = test_frames(48)
    # per-channel scale of every target, from a few teacher batches
    with torch.no_grad(), ac():
        xs = [norm_input(test[i:i + 2], dev) for i in range(0, 8, 2)]
        tt = [neck.trunk(x)[-1].float() for x in xs]
        tn = [neck_maps(neck, t.to(torch.bfloat16)) for t in tt]
        sd = lambda maps: torch.cat([m.float().permute(1, 0, 2, 3).flatten(1) for m in maps], 1).std(1).clamp_min(1e-3).view(1, -1, 1, 1)
        scale = {'trunk': sd(tt), 'neck': [[sd([n[h][i] for n in tn]) for i in range(len(SCALES))] for h in range(3)]}
    t_start = time.time()
    h, m = map(int, stop.split(':'))
    lt = time.localtime()
    stop_at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))
    if stop_at <= t_start:
        stop_at += 86400
    span = stop_at - t_start
    log = open(run / 'log.jsonl', 'a')
    step, t_last, seen, t_save = step0, time.time(), 0, time.time()
    nparam = sum(p.numel() for p in s.parameters()) / 1e6
    say(run, 'старт: ученик %s (%.1f млн параметров) учится выдавать карту 72x72x1024 кодировщика SAM 3.1; '
             'стоп в %s; проверка на 23.09 (ученик его не видит): признаки каждые 1000 шагов, весь SAM 3.1 с учеником '
             'каждые %d шагов' % (student, nparam, stop, EVAL_EVERY))
    best_found = None
    run_parts = {}

    def evaluate():
        s.eval()
        cos, mse, nk = [], [], {}
        with torch.no_grad(), ac():
            for i in range(0, len(test), 4):
                x = norm_input(test[i:i + 4], dev)
                t = neck.trunk(x)[-1]
                y = ema(x)[-1]
                cos.append(F.cosine_similarity(y.float(), t.float(), dim=1).mean().item())
                mse.append((((y.float() - t.float()) / scale['trunk']) ** 2).mean().item())
                tn, yn = neck_maps(neck, t), neck_maps(neck, y)
                for h, name in ((0, 'detector'), (2, 'tracker')):
                    for j, sc in enumerate(SCALES):
                        nk.setdefault('test_cos_%s_neck_%dx' % (name, 2 ** (2 - sc)), []).append(
                            F.cosine_similarity(yn[h][j].float(), tn[h][j].float(), dim=1).mean().item())
        s.train()
        r = {'test_cos_trunk': round(float(np.mean(cos)), 4), 'test_mse_trunk': round(float(np.mean(mse)), 4)}
        r.update({k: round(float(np.mean(v)), 4) for k, v in nk.items()})
        return r

    def save(tag='last'):
        torch.save({'model': s.state_dict(), 'ema': ema.state_dict(), 'arch': s.arch(), 'step': step,
                    'scale': {'trunk': scale['trunk'].cpu()}}, run / (tag + '.pt'))

    s.train()
    for u8 in dl:
        now = time.time()
        if now >= stop_at or (run / 'STOP').exists():
            break
        prog = min(1.0, (now - t_start) / span)
        warm = min(1.0, (step - step0 + 1) / 300)
        f = warm * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * prog)))
        for g in opt.param_groups:
            g['lr'] = g['base'] * f
        x = norm_input(u8, dev)
        with torch.no_grad(), ac():
            t_trunk = neck.trunk(x)[-1]
            t_necks = neck_maps(neck, t_trunk)
        with ac():
            y = s(x.contiguous(memory_format=torch.channels_last))[-1]
            s_necks = neck_maps(neck, y)
            loss, parts = losses(y, t_trunk, s_necks, t_necks, scale)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            d = min(0.999, (1 + step - step0) / (10 + step - step0))
            torch._foreach_lerp_(list(ema.parameters()), list(s.parameters()), 1 - d)
            for be, bs in zip(ema.buffers(), s.buffers()):       # BatchNorm running statistics: copied, not averaged
                be.copy_(bs)
        step += 1
        seen += len(u8)
        for k, v in parts.items():
            run_parts[k] = run_parts[k] * 0.98 + v * 0.02 if k in run_parts else v
        if step % 50 == 0:
            st = {'step': step, 'frames_per_s': round(seen / (time.time() - t_last), 2), 'lr_factor': round(f, 3),
                  'loss': {k: round(float(v), 4) for k, v in run_parts.items()}, 'updated': time.strftime('%H:%M:%S'),
                  'stop': stop}
            seen, t_last = 0, time.time()
            st['gpu_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
            st['eta_stop_min'] = round((stop_at - time.time()) / 60)
            if step % 1000 == 0:
                ev = evaluate()
                st.update(ev)
                txt = ('шаг %d: похожесть признаков на учителя (косинус, 1 = одинаковые) — карта %.3f, шея детектора %.3f, '
                       'шея трекера %.3f; %.2f кадра/с' % (step, ev['test_cos_trunk'], ev.get('test_cos_detector_neck_1x', 0),
                                                       ev.get('test_cos_tracker_neck_1x', 0), st['frames_per_s']))
                if step % EVAL_EVERY == 0:
                    pc = pipeline_check(run, step, ema.state_dict(), s.arch())
                    st['pipeline'] = pc
                    if 'found' in pc:
                        txt += ('; SAM 3.1 с учеником против SAM 3.1: найдено %.0f%% людей учителя, точность %.0f%%, маски %.2f, '
                                'IDF1 %.2f, смен номера %d (%d с на проверку)' % (100 * pc['found'], 100 * pc['precision'],
                                pc['mask_iou'], pc['idf1'], pc['id_switches'], pc.get('pipeline_s', 0)))
                        if best_found is None or pc['found'] + pc['idf1'] > best_found:
                            best_found = pc['found'] + pc['idf1']
                            save('best')
                    else:
                        txt += '; проверка всего SAM упала: %s' % pc.get('pipeline_error', '')[-200:]
                say(run, txt)
                log.write(json.dumps(st) + '\n'); log.flush()
            json.dump(st, open(run / 'status.json', 'w'), indent=1)
        if time.time() - t_save > 900:
            save(); t_save = time.time()
    save()
    st = {'step': step, 'finished': time.strftime('%H:%M:%S')}
    st.update(evaluate())
    pc = pipeline_check(run, step, ema.state_dict(), s.arch())
    st['pipeline'] = pc
    say(run, 'конец, шаг %d: признаки %.3f; весь SAM: %s' % (step, st['test_cos_trunk'], json.dumps(pc, ensure_ascii=False)))
    log.write(json.dumps(st) + '\n'); log.close()
    json.dump(st, open(run / 'status.json', 'w'), indent=1)
    print(st, flush=True)
    os._exit(0)                                     # the persistent loader workers would keep it alive


if __name__ == '__main__':
    a = sys.argv[1:]
    kw = {}
    for flag in ('--init', '--student', '--batch'):
        if flag in a:
            i = a.index(flag)
            kw[flag[2:]] = a[i + 1] if flag != '--batch' else int(a[i + 1])
            del a[i:i + 2]
    main(*a, **kw)
