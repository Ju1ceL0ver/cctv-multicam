"""The small SAM 3.1 over the door stretches, exactly the way the teacher labelled them (06.10.2026), so that its
entries and exits can be put next to the teacher's and the owner's.

Per stretch of data/sam31_door (camera 1, +-25 s around each counter event, 12.5 ticks a second): the same frames
(the stretch's video.mp4 squeezed to 1008 x 1008, as SAM saw them), the same sessions (info.json: 240 ticks,
overlapping by 8), SAM 3.1's own predictor with the student's encoder in place of the ViT-L (and, when given, the
student's shortened heads), the same seam linking and ReID joining (sam31_reid.py, 0.35), the same conversion into
door_v2's run format (door_sam.convert). Then door_learn / door_fuse read data/door_v2/<day>_<NAME>.jsonl.gz like
any other run.

usage (venv_sam3, the card): door_micro.py CKPT NAME DAY [DAY ...] [--heads HEADS.pt] [--only N]
  -> data/micro_door/NAME/<stretch>/cam1/, data/door_v2/<day>_NAME.jsonl.gz, data/micro_door/NAME/progress.md"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
TEACHER = ROOT / 'data' / 'sam31_door'
MICRO = ROOT / 'data' / 'micro_door'
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'



def _safe_print(line):
    """Windows consoles of background jobs cannot print every character (a '>=' sign killed two runs on 06.10)."""
    try:
        print(line, flush=True)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, 'encoding', None) or 'ascii'
        print(line.encode(enc, 'replace').decode(enc, 'replace'), flush=True)

def say(base, text):
    line = '%s  %s' % (time.strftime('%d.%m %H:%M'), text)
    with open(base / 'progress.md', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    _safe_print(line)


POSTER_BOX = (514 / 1280, 35 / 720, 559 / 1280, 158 / 720)      # camera 1's advertising stand (door_exclusions.POSTER)


def no_poster(pred, box=POSTER_BOX, min_iou=0.5):
    """SAM never starts a track on the advertising stand: a detection on its box is dropped before the tracker sees
    it. Kept, the stand is an old track that wins the no-overlap rule against whoever stands in front of it (06.10:
    a man vanished for 15 s in front of it)."""
    import torch
    m = pred.model
    orig = m.run_backbone_and_detection

    def run(*a, **kw):
        det_out, pos = orig(*a, **kw)
        b = det_out['bbox']                                      # normalised xyxy, (1, Q, 4)
        P = torch.tensor(box, device=b.device, dtype=b.dtype)
        lt = torch.max(b[..., :2], P[:2]); rb = torch.min(b[..., 2:], P[2:])
        inter = (rb - lt).clamp(min=0).prod(-1)
        area = (b[..., 2:] - b[..., :2]).clamp(min=0).prod(-1)
        iou = inter / (area + (P[2] - P[0]) * (P[3] - P[1]) - inter + 1e-9)
        hit = iou >= min_iou
        if hit.any():
            pos = pos & ~hit
            det_out['scores'] = torch.where(hit, torch.zeros_like(det_out['scores']), det_out['scores'])
        return det_out, pos
    m.run_backbone_and_detection = run
    return pred


def build(ckpt, heads=None):
    import torch
    import sam31_distill as SD
    import sam31_segment as S
    pred = S.build()
    pred.model.detector.backbone.vision_backbone.trunk = SD.load_student(ckpt)
    if os.environ.get('RA_KEEP_POSTER') != '1':
        no_poster(pred)
    import sam31_lite_eval as SL                      # 07.10: RA_S31_NEWDET / RA_S31_SCORE thresholds
    SL.tune(pred)
    if heads:
        import sam31_heads as SH
        SH.install(pred, torch.load(heads, map_location='cpu', weights_only=False))
    return pred


def frames(src, dst, stride=1):
    """Every stride-th tick of the stretch: sam_in/ (1008 x 1008, SAM's) and frames/ (2176 x 1224, the ReID crops;
    numbered like the kept ticks, so sam31_reid reads them instead of the stretch's video)."""
    import cv2
    (dst / 'sam_in').mkdir(parents=True, exist_ok=True)
    if stride > 1:
        (dst / 'frames').mkdir(exist_ok=True)
    cap = cv2.VideoCapture(str(src / 'video.mp4'))
    k = i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i % stride == 0:
            cv2.imwrite(str(dst / 'sam_in' / ('%05d.jpg' % k)), cv2.resize(f, (1008, 1008), interpolation=cv2.INTER_AREA),
                        [cv2.IMWRITE_JPEG_QUALITY, 93])
            if stride > 1:
                cv2.imwrite(str(dst / 'frames' / ('%05d.jpg' % k)), f, [cv2.IMWRITE_JPEG_QUALITY, 90])
            k += 1
        i += 1
    cap.release()
    return k


def sessions_of(n, size=240, overlap=8):
    out, s = [], 0
    while s < n:
        e = min(n, s + size)
        out.append((s, e, 0 if not out else overlap))
        if e == n:
            break
        s = e - overlap
    return out


def run(ckpt, name, days, heads=None, only=None, stride=1, tags=None):
    import torch
    import sam31_segment as SG
    import door_sam as DS
    import door_v2 as D
    base = MICRO / name
    base.mkdir(parents=True, exist_ok=True)
    say(base, 'старт: %s (%s), дни %s; те же отрезки, сеансы и сшивка, что у учителя' % (name, ckpt, ' '.join(days)))
    pred = build(ckpt, heads)
    for day in days:
        spans = D.stretches(day)
        if only:
            spans = spans[:int(only)]
        t_day, n_day = time.time(), 0
        for a, b in spans:
            tag = DS.tag_of(day, a)
            if tags and tag not in tags.split(','):
                continue
            src = TEACHER / tag / 'cam1'
            out = base / tag / 'cam1'
            if (out / 'report.json').exists() or not (src / 'info.json').exists():
                continue
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True)
            info = json.load(open(src / 'info.json'))
            stride = int(stride)
            n = frames(src, out, stride)
            sess = [tuple(s) for s in info['sessions']] if stride == 1 else sessions_of(n, int(os.environ.get('RA_LIVE_SESSION', max(60, 720 // stride))))   # 08.10: as door_live
            t0 = time.time()
            SG.label(pred, out, sess, None)
            sec = time.time() - t0
            shutil.rmtree(out / 'sam_in', ignore_errors=True)
            shutil.copy(src / 'ticks.json', out / 'ticks.json')
            if stride == 1:
                try:
                    os.link(src / 'video.mp4', out / 'video.mp4')
                except OSError:
                    shutil.copy(src / 'video.mp4', out / 'video.mp4')
            info.update({'model': name, 'ckpt': str(ckpt), 'heads': str(heads) if heads else None, 'stride': stride,
                         'ticks': n, 'sessions': [list(x) for x in sess],
                         's_per_live_tick': round(sec / max(1, sum(e - s for s, e, _ in sess)), 3)})
            json.dump(info, open(out / 'info.json', 'w'), indent=1)
            r = subprocess.run([RF, 'sam31_reid.py', '%s/cam1' % tag, '0.35'], cwd=str(ROOT), capture_output=True, text=True,
                               env=dict(os.environ, RA_S31_ROOT=str(base), RA_S31_VIDEO='0'))
            if r.returncode:
                say(base, 'ReID упал на %s: %s' % (tag, r.stderr[-300:]))
            shutil.rmtree(out / 'frames', ignore_errors=True)
            n_day += n
            torch.cuda.empty_cache()
        if tags:
            continue
        done = DS.convert([day], door=base, name=name, tick=D.TICK * int(stride))
        say(base, '%s: %d кадров за %.0f мин; треки -> data/door_v2/%s_%s.jsonl.gz (%s)' %
            (day, n_day, (time.time() - t_day) / 60, day, name, done))


if __name__ == '__main__':
    a = sys.argv[1:]
    kw = {}
    for flag in ('--heads', '--only', '--stride', '--tags'):
        if flag in a:
            i = a.index(flag)
            kw[flag[2:]] = a[i + 1]
            del a[i:i + 2]
    run(a[0], a[1], a[2:], **kw)
