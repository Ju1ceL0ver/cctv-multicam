"""How far from camera 1 every tracked person at the door is (03.10.2026), for the door rule.

Camera 1 looks at the door from inside, so a person in the shop is nearer than the door and one in the gallery
farther; depth_probe showed this tells the sides apart where the feet alone do not (AUC 0.74 -> 0.91 there).
Per tick of a door_v2 run (every EVERY-th): Depth Anything V2 Small on the whole frame (1088 x 612), put on the
empty hall's scale by an affine fit over the pixels outside every person; the student's (yolo26n-seg) masks matched
to the run's boxes; per person the median and the near quarter of the depth over the mask, the hall's depth behind
the mask, and the hall's depth under the feet.

usage (venv_rfdetr: transformers): door_depth.py RUN.jsonl.gz [EVERY]
  -> RUN.depth.json.gz {"<stretch>|<t>|<world>": [mask_depth, mask_near, mask_behind, foot_static]}"""
import gzip
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
STUDENT = ROOT / 'runs' / 'student_seg_all_n' / 'weights' / 'best.pt'


def main(path, every=2):
    import cv2
    import torch
    from PIL import Image
    from transformers import pipeline
    from ultralytics import YOLO
    import door_v2 as D
    import sam31_segment as SS
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
    rows = []
    try:
        for l in gzip.open(path, 'rt'):
            rows.append(json.loads(l))
    except (EOFError, json.JSONDecodeError):
        pass
    head, ticks = rows[0], rows[1:]
    day, cam = head['day'], head.get('cam', 'cam1')
    static = np.load(ROOT / 'data' / 'scene' / ('depth_%s_da2_large.npy' % cam)).astype(np.float32)
    pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=0 if torch.cuda.is_available() else -1)
    seg = YOLO(str(STUDENT))
    raw = D.Raw(day, cam)
    by_span = {}
    for r in ticks:
        by_span.setdefault(r['s'], []).append(r)
    out, t0, done = {}, time.time(), 0
    status = Path(str(path).replace('.jsonl.gz', '.depth.status.json'))
    for si, rs in sorted(by_span.items()):
        a, b = head['spans'][si]
        n = int(round((rs[-1]['t'] - a) / D.TICK)) + 1
        tks = SS.tick_frames(day, cam, a, n)
        for r in rs:
            i = int(round((r['t'] - a) / D.TICK))
            done += 1
            if i % every or not r['p']:
                continue
            img = raw.get(tks[i])
            if img is None:
                continue
            small = cv2.resize(img, (1088, 612), interpolation=cv2.INTER_AREA)
            dep = np.asarray(pipe(Image.fromarray(small[:, :, ::-1]))['predicted_depth'].squeeze(), np.float32)
            dep = cv2.resize(dep, (1088, 612), interpolation=cv2.INTER_CUBIC)
            boxes = []
            bgm = np.ones((612, 1088), bool)
            for q in r['p']:
                cx, cy, w, h = q['box']
                bx = np.array([(cx - w / 2) * 1088, (cy - h / 2) * 624, (cx + w / 2) * 1088, (cy + h / 2) * 624])
                boxes.append(bx)
                bgm[max(0, int(bx[1]) - 4):int(bx[3]) + 4, max(0, int(bx[0]) - 4):int(bx[2]) + 4] = False
            if bgm.sum() < 1000:
                continue
            coef = np.linalg.lstsq(np.stack([dep[bgm], np.ones(bgm.sum())], 1), static[bgm], rcond=None)[0]
            fit = dep * coef[0] + coef[1]
            res = seg.predict(small, imgsz=1088, conf=0.25, verbose=False, retina_masks=True)[0]
            B = res.boxes.xyxy.cpu().numpy() if res.boxes is not None and len(res.boxes) else np.zeros((0, 4))
            M = res.masks.data.cpu().numpy() > 0.5 if res.masks is not None and len(B) else None
            for q, bx in zip(r['p'], boxes):
                key = '%d|%.2f|%d' % (r['s'], r['t'], q['w'])
                fx, fy = (int(np.clip(q['foot'][0] / 2, 0, 1087)), int(np.clip(q['foot'][1] / 2, 0, 611))) if q.get('foot') else (None, None)
                fs = float(static[fy, fx]) if fx is not None else None
                md = mn = mb = None
                if len(B):
                    ix = np.clip(np.minimum(B[:, 2], bx[2]) - np.maximum(B[:, 0], bx[0]), 0, None) * np.clip(np.minimum(B[:, 3], bx[3]) - np.maximum(B[:, 1], bx[1]), 0, None)
                    iou = ix / ((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - ix + 1e-6)
                    j = int(np.argmax(iou))
                    if iou[j] >= 0.4:
                        m = cv2.resize(M[j].astype(np.uint8), (1088, 612), interpolation=cv2.INTER_NEAREST) > 0
                        if m.sum() >= 30:
                            v = fit[m]
                            md, mn, mb = float(np.median(v)), float(np.percentile(v, 75)), float(np.median(static[m]))
                if md is None:                                     # no mask: the middle of the box
                    x1, x2 = int((bx[0] + bx[2]) / 2 - (bx[2] - bx[0]) * 0.2), int((bx[0] + bx[2]) / 2 + (bx[2] - bx[0]) * 0.2)
                    y1, y2 = int(bx[1] + (bx[3] - bx[1]) * 0.3), int(bx[1] + (bx[3] - bx[1]) * 0.9)
                    reg = fit[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
                    if reg.size:
                        md = float(np.median(reg)); mn = float(np.percentile(reg, 75))
                        mb = float(np.median(static[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]))
                out[key] = [md, mn, mb, fs]
        json.dump({'span': si, 'spans': len(by_span), 'ticks': done, 'of': len(ticks), 'elapsed_s': round(time.time() - t0),
                   'updated': time.strftime('%H:%M:%S')}, open(status, 'w'))
        if si % 5 == 0:                                            # keep what is done if the night ends early
            with gzip.open(str(path).replace('.jsonl.gz', '.depth.json.gz'), 'wt') as f:
                json.dump(out, f)
    with gzip.open(str(path).replace('.jsonl.gz', '.depth.json.gz'), 'wt') as f:
        json.dump(out, f)
    json.dump({'finished': time.strftime('%H:%M:%S'), 'people_ticks': len(out), 'elapsed_s': round(time.time() - t0)}, open(status, 'w'))


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2)
