"""A sheet for the eye: the owner's masks, the current teacher's draft (yolo26x-seg + SAM 2.1 L, the /paint
<id>_init.png he started from) and SAM 3 (text "person", conf >= 0.5) on the same exam frames.
Each person in a colour of his own; a machine mask nobody of the owner's matches (IoU < 0.5) in red, and the
owner's people the machine missed get a white dashed box.

usage: sam3_sheet.py [N]   -> data/logs/sam3_sheet.jpg"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
COL = [(66, 197, 245), (80, 220, 100), (245, 160, 66), (220, 90, 220), (240, 220, 70), (160, 100, 255), (60, 200, 200),
       (200, 160, 90), (120, 230, 180), (250, 120, 160)]


def draw(img, masks, missed=(), false=()):
    import cv2
    out = img.copy()
    for k, m in enumerate(masks):
        colour = (40, 40, 255) if k in false else COL[k % len(COL)]
        cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cs, -1, (0, 0, 0), 4)
        cv2.drawContours(out, cs, -1, colour, 2)
    for m in missed:
        ys, xs = np.nonzero(m)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        for x in range(x1, x2, 12):
            cv2.line(out, (x, y1), (min(x + 6, x2), y1), (255, 255, 255), 2); cv2.line(out, (x, y2), (min(x + 6, x2), y2), (255, 255, 255), 2)
        for y in range(y1, y2, 12):
            cv2.line(out, (x1, y), (x1, min(y + 6, y2)), (255, 255, 255), 2); cv2.line(out, (x2, y), (x2, min(y + 6, y2)), (255, 255, 255), 2)
    return out


def label(img, text):
    import cv2
    cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img, text, (10, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return img


def main():
    import cv2
    import gold
    from ultralytics.models.sam import SAM3SemanticPredictor
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ids = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    # the most crowded frames first: that is where the teachers differ
    ids = sorted(ids, key=lambda i: -len(gold.gold(i)))[:n]
    pred = SAM3SemanticPredictor(overrides=dict(conf=0.5, task='segment', mode='predict', model=str(ROOT / 'data' / 'weights' / 'sam3' / 'sam3.pt'),
                                                half=True, save=False, verbose=False, imgsz=1008))
    rows = []
    for ident in ids:
        img = cv2.imread(str(gold.PAINT / ('%s.jpg' % ident)))
        truth = gold.gold(ident)
        init = cv2.imread(str(gold.PAINT / ('%s_init.png' % ident)), cv2.IMREAD_UNCHANGED)
        init = init[:, :, 0] if init is not None and init.ndim == 3 else init
        teacher = [init == v for v in np.unique(init) if v and (init == v).sum() >= gold.MIN_PX] if init is not None else []
        pred.set_image(str(gold.PAINT / ('%s.jpg' % ident)))
        if not hasattr(pred.model, 'mask_threshold'):
            pred.model.mask_threshold = 0.0
        r = pred(text=['person'])[0]
        sam = []
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy() > 0.5:
                if m.shape != truth[0].shape:
                    m = cv2.resize(m.astype(np.uint8), (truth[0].shape[1], truth[0].shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
                if m.sum() >= gold.MIN_PX:
                    sam.append(m)
        panels = [label(draw(img, truth), 'owner: %d people' % len(truth))]
        for name, ms in (('teacher (yolo + SAM 2.1)', teacher), ('SAM 3', sam)):
            pairs, iou = gold.match(truth, ms)
            hit_t = {i for i, _ in pairs}; hit_m = {j for _, j in pairs}
            false = [j for j in range(len(ms)) if j not in hit_m]
            missed = [truth[i] for i in range(len(truth)) if i not in hit_t]
            med = np.median([iou[i, j] for i, j in pairs]) if pairs else 0
            panels.append(label(draw(img, ms, missed, false), '%s: found %d/%d, false %d, contour %.2f' % (name, len(pairs), len(truth), len(false), med)))
        rows.append(np.hstack([cv2.resize(p, (800, 450), interpolation=cv2.INTER_AREA) for p in panels]))
    cv2.imwrite(str(ROOT / 'data' / 'logs' / 'sam3_sheet_1008.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('ok', ids)


if __name__ == '__main__':
    main()
