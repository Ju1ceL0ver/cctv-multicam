"""SAM 3.1's heads shortened and taught in its shadow (06.10.2026): stage 2 of the small SAM 3.1.

Three stacks of SAM 3.1 are cut: the detector's encoder 6 -> 2 layers, its decoder 6 -> 3, the tracker's memory
attention 4 -> 2. The shorter copies start from the teacher's own layers, spread over the stack (encoder 0 and 5,
decoder 0, 3 and 5, memory attention 0 and 3), and learn in its shadow: SAM 3.1 runs its sessions as always (with
the student encoder of stage 1), and every time one of the three stacks is called, its short copy gets the very same
inputs and learns to give the same output -- the encoder's memory, the decoder's query states, boxes and presence
logit of the last layer, the tracker's conditioned map. No labels, no matching: the queries are the same learned
slots in both, so slot i is compared with slot i. The teacher's answer is what SAM 3.1 uses downstream, so its
memory bank and its tracks stay the teacher's while the copies learn.

Frames: clips of CLIP ticks from every window of data/sam31_seg except 23.09 and every door stretch.
Every EVAL_EVERY minutes the whole small SAM (student encoder + short heads) is checked against SAM 3.1 on 23.09
(sam31_lite_eval.py --heads), and the speed is measured there too.

usage (venv_sam3, the card): sam31_heads.py OUT STUDENT_CKPT [STOP HH:MM]  -> runs/OUT/{heads.pt, progress.md, log.jsonl}"""
import copy
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
KEEP = {'det_enc': [0, 5], 'det_dec': [0, 3, 5], 'trk': [0, 3]}
CLIP = 64
EVAL_EVERY = 45                      # minutes



def _safe_print(line):
    """Windows consoles of background jobs cannot print every character (a '>=' sign killed two runs on 06.10)."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
        print(line.encode(enc, 'replace').decode(enc, 'replace'), flush=True)

def stacks(pred):
    t = pred.model.detector.transformer
    return {'det_enc': t.encoder, 'det_dec': t.decoder, 'trk': pred.model.tracker.model.transformer.encoder}


def shorten(module, keep):
    m = copy.deepcopy(module)
    m.layers = nn.ModuleList([m.layers[i] for i in keep])
    m.num_layers = len(keep)
    if hasattr(m, 'fine_layers'):
        m.fine_layers = [m.fine_layers[i] for i in keep]
    for p in m.parameters():
        p.requires_grad_(True)
    return m


def install(pred, ck):
    """Put the trained short stacks in place of SAM 3.1's own."""
    full = stacks(pred)
    keep = ck.get('keep', KEEP)
    for name, mod in full.items():
        s = shorten(mod, keep[name])
        s.load_state_dict(ck['state'][name])
        s.to(next(mod.parameters()).device).eval().requires_grad_(False)
        if name == 'trk':
            pred.model.tracker.model.transformer.encoder = s
        elif name == 'det_enc':
            pred.model.detector.transformer.encoder = s
        else:
            pred.model.detector.transformer.decoder = s


# ------------------------------------------------------------------ the shadow

def clone(x):
    if torch.is_tensor(x):
        return x.clone()
    if isinstance(x, list):
        return [clone(v) for v in x]
    if isinstance(x, tuple):
        return tuple(clone(v) for v in x)
    if isinstance(x, dict):
        return {k: clone(v) for k, v in x.items()}
    return x


def pairs(name, s, t, out):
    """What goes on downstream, student against teacher: the encoders' memory; the decoder's query states and presence
    logit of its last layer (its reference boxes are left out: nearly constant, they blew the loss up to nan on 06.10)."""
    if name in ('det_enc', 'trk'):
        out.append((s['memory'], t['memory']))
    elif name == 'det_dec':
        out.append((s[0][-1], t[0][-1]))                      # (Q, B, D) query states, last layer
        if s[2] is not None and t[2] is not None and torch.is_tensor(s[2]) and len(s[2]):
            out.append((s[2][-1], t[2][-1]))                  # presence logit, last layer


class Shadow:
    def __init__(self, name, teacher, student, opt, stats):
        self.name, self.teacher_forward, self.student, self.opt, self.stats = name, teacher.forward, student, opt, stats
        self.on = True

    def __call__(self, *a, **kw):
        if self.on:
            with torch.inference_mode(False):
                a_s, kw_s = clone(a), clone(kw)                       # the teacher's call may change its lists in place
        out_t = self.teacher_forward(*a, **kw)
        if not self.on:
            return out_t
        with torch.inference_mode(False), torch.enable_grad():
            tgt = clone(out_t)
            out_s = self.student(*a_s, **kw_s)
            pr = []
            pairs(self.name, out_s, tgt, pr)
            if pr:
                loss, cos = 0.0, []
                for s, t in pr:
                    s, t = s.float(), t.float().detach()
                    loss = loss + ((s - t) ** 2).mean() / (t.pow(2).mean() + 1e-2)
                    if s.dim() >= 2 and s.shape[-1] > 1:
                        c = F.cosine_similarity(s.reshape(-1, s.shape[-1]), t.reshape(-1, t.shape[-1]), dim=-1).mean()
                        loss = loss + (1 - c)
                        cos.append(float(c))
                if not torch.isfinite(loss):
                    st = self.stats.setdefault(self.name, {'n': 0, 'loss': None, 'cos': None})
                    st['skipped'] = st.get('skipped', 0) + 1
                    return out_t
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                self.opt.step()
                st = self.stats.setdefault(self.name, {'n': 0, 'loss': None, 'cos': None})
                st['n'] += 1
                lv, cv = float(loss), (float(np.mean(cos)) if cos else None)
                st['loss'] = lv if st['loss'] is None else 0.98 * st['loss'] + 0.02 * lv
                if cv is not None:
                    st['cos'] = cv if st['cos'] is None else 0.98 * st['cos'] + 0.02 * cv
        return out_t


# ------------------------------------------------------------------ clips

def sources():
    import sam31_distill as SD
    out = []
    for kind in ('sam31_seg', 'sam31_door'):
        for d in sorted((ROOT / 'data' / kind).glob('*/cam*')):
            if d.parent.name.startswith(SD.TEST_DAY) or not (d / 'video.mp4').exists() or not (d / 'chunks.npz').exists():
                continue
            try:
                ticks = np.unique(np.load(d / 'chunks.npz')['rows'][:, 1].astype(int))
            except Exception:
                continue
            if len(ticks) > CLIP:
                out.append((d, ticks))
    return out


def clip_frames(rng, srcs, folder):
    import cv2
    d, ticks = rng.choice(srcs)
    k0 = int(rng.choice(ticks[:-CLIP]))
    shutil.rmtree(folder, ignore_errors=True); folder.mkdir(parents=True)
    cap = cv2.VideoCapture(str(d / 'video.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES, k0)
    n = 0
    for k in range(CLIP):
        ok, f = cap.read()
        if not ok:
            break
        cv2.imwrite(str(folder / ('%05d.jpg' % k)), cv2.resize(f, (1008, 1008), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 93])
        n += 1
    cap.release()
    return '%s/%s@%d' % (d.parent.name, d.name, k0), n


# ------------------------------------------------------------------ training

def say(run, text):
    line = '%s  %s' % (time.strftime('%d.%m %H:%M'), text)
    with open(run / 'progress.md', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    _safe_print(line)


def save(run, students, step):
    torch.save({'keep': KEEP, 'step': step, 'state': {k: v.state_dict() for k, v in students.items()}}, run / 'heads.pt')


def check(run, student_ckpt, tag):
    py = ROOT.parent / 'venv_sam3' / 'Scripts' / 'python.exe'
    ev = run / 'evals'
    ev.mkdir(exist_ok=True)
    out = ev / ('%s.json' % tag)
    torch.cuda.empty_cache()
    r = subprocess.run([str(py if py.exists() else sys.executable), str(ROOT / 'sam31_lite_eval.py'), str(student_ckpt), '2', '96',
                        '--cache', str(ROOT / 'runs' / 's31micro_a' / 'evals' / 'teacher_cache'), '--out', str(out),
                        '--preview', str(ev / ('%s.jpg' % tag)), '--film', 'none', '--heads', str(run / 'heads.pt')],
                       capture_output=True, text=True, cwd=str(ROOT))
    if r.returncode != 0 or not out.exists():
        return {'error': (r.stderr or r.stdout)[-500:]}
    rep = json.load(open(out))
    m = rep['mean']
    m['lite_s_per_frame'] = round(float(np.mean([s['lite_s_per_frame'] for s in rep['stretches']])), 3)
    return m


def main(out, student_ckpt, stop='19:00'):
    import sam31_distill as SD
    import sam31_segment as S
    torch.backends.cuda.matmul.allow_tf32 = True
    run = ROOT / 'runs' / out
    run.mkdir(parents=True, exist_ok=True)
    pred = S.build()
    pred.model.detector.backbone.vision_backbone.trunk = SD.load_student(str(student_ckpt))
    full = stacks(pred)
    students = {k: shorten(m, KEEP[k]).train(False) for k, m in full.items()}        # eval mode: no dropout, grads still flow
    params = [p for s in students.values() for p in s.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    stats = {}
    shadows = {}
    for k, m in full.items():
        sh = Shadow(k, m, students[k], opt, stats)
        m.forward = sh
        shadows[k] = sh
    n_t = {k: sum(p.numel() for p in m.parameters()) / 1e6 for k, m in full.items()}
    n_s = {k: sum(p.numel() for p in m.parameters()) / 1e6 for k, m in students.items()}
    say(run, 'этап 2 старт: укороченные головы учатся в тени SAM 3.1 на одних и тех же входах — детектор: кодировщик %s, '
             'декодер %s; трекер: внимание к памяти %s (млн параметров: было %s, стало %s); стоп в %s; проверка всего '
             'маленького SAM против SAM 3.1 на 23.09 каждые %d мин' % (KEEP['det_enc'], KEEP['det_dec'], KEEP['trk'],
             {k: round(v, 1) for k, v in n_t.items()}, {k: round(v, 1) for k, v in n_s.items()}, stop, EVAL_EVERY))
    srcs = sources()
    rng = random.Random(0)
    h, mi = map(int, stop.split(':'))
    lt = time.localtime()
    stop_at = time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, mi, 0, 0, 0, -1))
    if stop_at <= time.time():
        stop_at += 86400
    t_start, span = time.time(), stop_at - time.time()
    t_eval, t_save, clips, frames = time.time(), time.time(), 0, 0
    log = open(run / 'log.jsonl', 'a')
    folder = run / 'clip'
    while time.time() < stop_at and not (run / 'STOP').exists():
        f = 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, (time.time() - t_start) / span)))
        for g in opt.param_groups:
            g['lr'] = 1e-4 * f
        name, n = clip_frames(rng, srcs, folder)
        if n < 8:
            continue
        SD_sessions = [(0, n, 0)]
        try:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                sid = pred.handle_request(dict(type='start_session', resource_path=str(folder), offload_video_to_cpu=True))['session_id']
                pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
                for _ in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
                    pass
                pred.handle_request(dict(type='close_session', session_id=sid))
        except Exception as e:
            say(run, 'клип %s упал: %s' % (name, repr(e)[:300]))
            torch.cuda.empty_cache()
            continue
        torch.cuda.empty_cache()
        clips += 1
        frames += n
        if clips % 10 == 0:
            st = {'clips': clips, 'frames': frames, 'lr': round(1e-4 * f, 7), 'updated': time.strftime('%H:%M:%S'),
                  'shadow': {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in stats.items()},
                  'frames_per_s': round(frames / (time.time() - t_start), 2)}
            json.dump(st, open(run / 'status.json', 'w'), indent=1)
            log.write(json.dumps(st) + '\n'); log.flush()
        if time.time() - t_save > 600:
            save(run, students, clips); t_save = time.time()
        if time.time() - t_eval > EVAL_EVERY * 60:
            save(run, students, clips)
            for sh in shadows.values():
                sh.on = False
            m = check(run, student_ckpt, 'clips_%05d' % clips)
            for sh in shadows.values():
                sh.on = True
            txt = 'клипов %d (%d кадров): похожесть выходов на учителя (косинус) — %s' % (
                clips, frames, ', '.join('%s %.3f' % (k, v['cos']) for k, v in stats.items() if v.get('cos') is not None))
            if 'found' in m:
                txt += ('; маленький SAM целиком против SAM 3.1 на 23.09: найдено %.0f%%, точность %.0f%%, маски %.2f, IDF1 %.2f, '
                        'смен номера %d, %.0f мс/кадр' % (100 * m['found'], 100 * m['precision'], m['mask_iou'], m['idf1'],
                                                          m['id_switches'], 1000 * m['lite_s_per_frame']))
            else:
                txt += '; проверка упала: %s' % m.get('error', '')[-300:]
            say(run, txt)
            t_eval = time.time()
    save(run, students, clips)
    m = check(run, student_ckpt, 'final')
    say(run, 'конец этапа 2: %d клипов, %d кадров; итог: %s' % (clips, frames, json.dumps(m, ensure_ascii=False)))
    os._exit(0)


if __name__ == '__main__':
    main(*sys.argv[1:])
