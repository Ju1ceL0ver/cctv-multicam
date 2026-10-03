"""Does the person's own depth tell inside from outside at the door? (03.10.2026)

Camera 1 looks at the door from inside: people in the shop are nearer than the door, people in the gallery farther.
For the feet of the people who really crossed (door_state.side_samples: 1.5-6 s before an entry outside, after it
inside; exits the other way), in the door zone of the frame, the frame goes through Depth Anything V2; its relative
depth is put on the empty hall's scale (an affine fit over the pixels outside every person's box) and the person's
value is the median over the middle of their box. Compared, by ROC AUC and by a cross-validated logistic fit:
the person's depth, the box height, the foot position, and their mixes.

usage: depth_probe.py [N] DAY=RUN.jsonl.gz ...  -> data/door_v2/depth_probe.json"""
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ZONE = (300, 350, 1300, 1050)      # x1 y1 x2 y2 (2176 frame) of the feet: the door and the vestibule
LO_S = float(os.environ.get('RA_PROBE_LO', '1.5'))   # 3.0: away from the counter's +-2 s jitter, cleaner sides
STUDENT = ROOT / 'runs' / 'student_seg_all_n' / 'weights' / 'best.pt'


def main(n, args):
    import cv2
    import torch
    from PIL import Image
    from transformers import pipeline
    import door_learn as L
    import door_state as S
    import door_v2 as D
    import sam31_segment as SS
    rng = random.Random(0)
    static = np.load(ROOT / 'data' / 'scene' / 'depth_cam1_da2_large.npy').astype(np.float32)      # 612 x 1088
    samples = []
    for a in args:
        day, path = a.split('=', 1)
        ticks, _ = D._read(path)
        at = {round(r['t'], 2): r for r in ticks}
        d = {'learn': L.load_day(day, path), 'by': S.tracks(ticks)}
        for (f, kind, t, w), lab in zip(d['learn']['cands'], d['learn']['y']):
            if lab != 1:
                continue
            tc = t - L.SHIFT
            for s, q in d['by'].get(w, []):
                dt = s - tc
                fx, fy = q['foot']
                if LO_S <= abs(dt) <= 6 and ZONE[0] <= fx <= ZONE[2] and ZONE[1] <= fy <= ZONE[3]:
                    inside = (dt > 0) if kind == 'in' else (dt < 0)
                    samples.append((day, round(s, 2), q, int(inside), at[round(s, 2)]['p']))
    rng.shuffle(samples)
    pos = [x for x in samples if x[3] == 1][:n // 2]
    neg = [x for x in samples if x[3] == 0][:n // 2]
    picked = sorted(pos + neg, key=lambda x: (x[0], x[1]))
    dev = 0 if torch.cuda.is_available() else -1
    pipe = pipeline('depth-estimation', model='depth-anything/Depth-Anything-V2-Small-hf', device=dev)
    raws = {}
    rows = []
    from ultralytics import YOLO
    seg = YOLO(str(STUDENT))
    for day, s, q, y, people in picked:
        if day not in raws:
            raws[day] = D.Raw(day)
        img = raws[day].get(SS.tick_frames(day, 'cam1', s, 1)[0])
        if img is None:
            continue
        small = cv2.resize(img, (1088, 612), interpolation=cv2.INTER_AREA)
        dep = np.asarray(pipe(Image.fromarray(small[:, :, ::-1]))['predicted_depth'].squeeze(), np.float32)
        dep = cv2.resize(dep, (1088, 612), interpolation=cv2.INTER_CUBIC)
        bgm = np.ones((612, 1088), bool)
        for p in people:
            cx, cy, w, h = p['box']
            x1, y1, x2, y2 = (cx - w / 2) * 1088, (cy - h / 2) * 1248 / 2, (cx + w / 2) * 1088, (cy + h / 2) * 1248 / 2
            bgm[max(0, int(y1) - 4):int(y2) + 4, max(0, int(x1) - 4):int(x2) + 4] = False
        A = np.stack([dep[bgm], np.ones(bgm.sum())], 1)
        coef = np.linalg.lstsq(A, static[bgm], rcond=None)[0]
        fit = dep * coef[0] + coef[1]
        # the person's own pixels: the student's mask that fits the run's box best, laid on the frame's depth
        cx, cy, w, h = q['box']
        qb = np.array([(cx - w / 2) * 1088, (cy - h / 2) * 624, (cx + w / 2) * 1088, (cy + h / 2) * 624])
        res = seg.predict(small, imgsz=1088, conf=0.25, verbose=False, retina_masks=True)[0]
        mask_depth = mask_near = mask_bg = np.nan
        if res.masks is not None and len(res.boxes):
            B = res.boxes.xyxy.cpu().numpy()
            ix = np.clip(np.minimum(B[:, 2], qb[2]) - np.maximum(B[:, 0], qb[0]), 0, None) * np.clip(np.minimum(B[:, 3], qb[3]) - np.maximum(B[:, 1], qb[1]), 0, None)
            iou = ix / ((B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) + (qb[2] - qb[0]) * (qb[3] - qb[1]) - ix + 1e-6)
            j = int(np.argmax(iou))
            if iou[j] >= 0.4:
                m = res.masks.data[j].cpu().numpy() > 0.5
                m = cv2.resize(m.astype(np.uint8), (1088, 612), interpolation=cv2.INTER_NEAREST) > 0
                if m.sum() >= 30:
                    v = fit[m]
                    mask_depth, mask_near, mask_bg = float(np.median(v)), float(np.percentile(v, 75)), float(np.median(static[m]))
        x1, x2 = int((cx - w * 0.2) * 1088), int((cx + w * 0.2) * 1088)
        y1, y2 = int((cy - h * 0.2) * 1248 / 2), int((cy + h * 0.4) * 1248 / 2)
        reg = fit[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        bgr = static[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
        ffx, ffy = int(np.clip(q['foot'][0] / 2, 0, 1087)), int(np.clip(q['foot'][1] / 2, 0, 611))
        rows.append({'day': day, 't': s, 'inside': y, 'person_depth': float(np.median(reg)), 'behind_depth': float(np.median(bgr)),
                     'mask_depth': mask_depth, 'mask_near': mask_near, 'mask_behind': mask_bg, 'foot_static': float(static[ffy, ffx]),
                     'height': h * 1248, 'fx': q['foot'][0], 'fy': q['foot'][1]})
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_predict
    Y = np.array([r['inside'] for r in rows])
    feats = {'person_depth': ['person_depth'], 'person_minus_behind': ['person_depth', 'behind_depth'], 'height': ['height'],
             'foot_xy': ['fx', 'fy'], 'foot_xy_height': ['fx', 'fy', 'height'], 'foot_xy_height_depth': ['fx', 'fy', 'height', 'person_depth', 'behind_depth']}
    rep = {'samples': len(rows), 'inside': int(Y.sum())}
    for name, cols in feats.items():
        X = np.array([[r[c] for c in cols] for r in rows], np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        P = cross_val_predict(LogisticRegression(max_iter=1000), X, Y, cv=5, method='predict_proba')[:, 1]
        rep[name] = {'auc': round(float(roc_auc_score(Y, P)), 4), 'accuracy': round(float(((P > 0.5) == Y).mean()), 4)}
    for c in ('person_depth', 'height', 'fy'):
        v = np.array([r[c] for r in rows])
        rep['raw_auc_' + c] = round(float(max(roc_auc_score(Y, v), 1 - roc_auc_score(Y, v))), 4)
    rep['rows'] = rows
    json.dump(rep, open(ROOT / 'data' / 'door_v2' / ('depth_probe%s.json' % os.environ.get('RA_PROBE_TAG', '')), 'w'), indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != 'rows'}, indent=1))


if __name__ == '__main__':
    main(int(sys.argv[1]), sys.argv[2:])
