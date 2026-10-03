"""SAM 3 on the full 2560x1440 exam frames of 18.09, for the owner's eye: every "person" it finds outlined
with its confidence; red when it matches none of the owner's people (IoU < 0.5), a white dashed box for a
person of his it missed.

usage: sam3_full.py   -> data/logs/sam3_full/NNN.jpg (1920x1080), data/logs/sam3_full_sheet.jpg (all frames)"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONF = 0.25


def main():
    import cv2
    import gold
    import sam3_crop as C
    import sam3_sheet as S
    from ultralytics.models.sam import SAM3SemanticPredictor
    man = {it['id']: it for it in json.load(open(gold.PAINT / 'manifest.json'))['items']}
    ids = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    pred = SAM3SemanticPredictor(overrides=dict(conf=CONF, task='segment', mode='predict', model=str(C.W3), half=True, save=False, verbose=False, imgsz=1008))
    out = ROOT / 'data' / 'logs' / 'sam3_full'
    out.mkdir(parents=True, exist_ok=True)
    thumbs = []
    for ident in ids:
        full = C.full_frame(man[ident]) if ident in man else None
        if full is None:
            continue
        H, W = full.shape[:2]
        pred.set_image(full); C.patch(pred)
        r = pred(text=['person'])[0]
        ms, conf = [], []
        if r.masks is not None:
            for m, c in zip(r.masks.data.cpu().numpy() > 0.5, r.boxes.conf.cpu().numpy()):
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                ms.append(m); conf.append(float(c))
        keep = C.dedup(ms, conf)
        ms = [ms[k] for k in keep]; conf = [conf[k] for k in keep]
        truth = gold.gold(ident)
        small = [C.to_small(m) for m in ms]
        pairs, iou = gold.match(truth, [m for m in small])
        hit_m = {j for _, j in pairs}; hit_t = {i for i, _ in pairs}
        false = [j for j in range(len(ms)) if j not in hit_m]
        big = [cv2.resize(t.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool) for t in truth]
        img = S.draw(full, ms, [big[i] for i in range(len(big)) if i not in hit_t], false)
        for m, c in zip(ms, conf):                               # the confidence at the top of each mask
            ys, xs = np.nonzero(m)
            cv2.putText(img, '%.2f' % c, (int(xs.mean()) - 30, max(30, int(ys.min()) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5)
            cv2.putText(img, '%.2f' % c, (int(xs.mean()) - 30, max(30, int(ys.min()) - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        S.label(img, '18.09 exam %s: owner %d, SAM 3 found %d, false %d (red), missed %d (dashed)' % (ident, len(truth), len(pairs), len(false), len(truth) - len(pairs)))
        cv2.imwrite(str(out / ('%s.jpg' % ident)), cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 88])
        thumbs.append(cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA))
    rows = [np.hstack(thumbs[k:k + 4] + [np.zeros_like(thumbs[0])] * (4 - len(thumbs[k:k + 4]))) for k in range(0, len(thumbs), 4)]
    cv2.imwrite(str(ROOT / 'data' / 'logs' / 'sam3_full_sheet.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print('frames', len(thumbs))


if __name__ == '__main__':
    main()
