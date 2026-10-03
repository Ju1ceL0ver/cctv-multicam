"""The owner's idea (27.09): SAM 3 finds the people, then every person is outlined again close up.

  find     SAM 3 on the full 2560x1440 frame, text "person" (imgsz 1008, conf >= CONF); duplicates (mask IoU
           >= 0.5 with a surer one) dropped
  closer   per person: the box + MARGIN px on the full frame is cut out and the person outlined again inside
           it from his box -- by SAM 2.1 Large, and by SAM 3 itself (box prompt)
  together the close-up masks are put back on the full frame; where two overlap the nearer person (lower
           feet) wins, as in the drafts; nothing is drawn outside a person's cut-out
  measure  on the owner's 18.09 exam frames against his masks (at 1280x720): found, false, contour, small --
           SAM 3 as it comes, and the two close-up variants

usage: sam3_crop.py [N_SHEET]   -> data/logs/sam3_crop.json, data/logs/sam3_crop.jpg"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CONF = 0.4
MARGIN = 50
W3 = ROOT / 'data' / 'weights' / 'sam3' / 'sam3.pt'


def full_frame(it):
    """The exam frame at full size from the raw recording."""
    import seg_compare
    if it.get('film') is not None:
        return seg_compare.raw_at(it['day'], it['cam'], it['film'], str(ROOT))[0]
    import paint
    from rawsource import segments
    path = next((p for p, _ in segments(it['cam'], it['day']) if os.path.basename(p) == it['segment']), None)
    return paint._raw_frame(path, it['second'])[0] if path else None


def patch(pred):
    if getattr(pred, 'model', None) is not None and not hasattr(pred.model, 'mask_threshold'):
        pred.model.mask_threshold = 0.0      # Ultralytics 8.4.108 asks the SAM 3 model for it and it has none


def dedup(masks, scores, iou=0.5):
    order = sorted(range(len(masks)), key=lambda j: -scores[j])
    kept = []
    for j in order:
        if all((masks[j] & masks[i]).sum() < iou * max(1, (masks[j] | masks[i]).sum()) for i in kept):
            kept.append(j)
    return kept


def put_together(shape, pieces):
    """pieces: [(box full px, mask of the cut-out, (X1, Y1))] -> list of full-frame masks, the nearer on top."""
    lab = np.zeros(shape, np.int32)
    order = sorted(range(len(pieces)), key=lambda k: pieces[k][0][3])       # farther (higher feet) first
    for k in order:
        box, m, (X1, Y1) = pieces[k]
        h, w = m.shape
        sub = lab[Y1:Y1 + h, X1:X1 + w]
        sub[m] = k + 1
    return [lab == k + 1 for k in range(len(pieces))]


def to_small(m, size=(1280, 720)):
    import cv2
    return cv2.resize(m.astype(np.float32), size, interpolation=cv2.INTER_AREA) > 0.5


def score(truth_all, pred_all):
    import gold
    found = total = false = sf = st = 0
    ious = []
    for truth, pred in zip(truth_all, pred_all):
        pred = [m for m in pred if m.sum() >= gold.MIN_PX]
        pairs, iou = gold.match(truth, pred)
        total += len(truth); found += len(pairs); false += len(pred) - len(pairs)
        ious += [iou[i, j] for i, j in pairs]
        hit = {i for i, _ in pairs}
        for i, t in enumerate(truth):
            ys = np.nonzero(t.any(1))[0]
            if ys[-1] - ys[0] < gold.SMALL:
                st += 1; sf += i in hit
    return {'people': total, 'recall': round(found / max(1, total), 4), 'false': false, 'precision': round(found / max(1, found + false), 4),
            'mask_iou_median': round(float(np.median(ious)), 4) if ious else None, 'mask_iou_mean': round(float(np.mean(ious)), 4) if ious else None,
            'small_recall': round(sf / max(1, st), 4)}


def main():
    import cv2
    import torch
    import gold
    import seg_compare
    from ultralytics import SAM
    from ultralytics.models.sam import SAM3Predictor, SAM3SemanticPredictor
    n_sheet = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    man = {it['id']: it for it in json.load(open(gold.PAINT / 'manifest.json'))['items']}
    ids = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    finder = SAM3SemanticPredictor(overrides=dict(conf=CONF, task='segment', mode='predict', model=str(W3), half=True, save=False, verbose=False, imgsz=1008))
    boxer3 = SAM3Predictor(overrides=dict(conf=0.25, task='segment', mode='predict', model=str(W3), half=True, save=False, verbose=False, imgsz=1008))
    sam21 = SAM(seg_compare.SAM_WEIGHTS)
    rep = {'conf': CONF, 'margin': MARGIN}
    truth_all, direct_all, c21_all, c3_all, frames_kept = [], [], [], [], []
    t = {'find': 0.0, 'sam21': 0.0, 'sam3box': 0.0}
    for ident in ids:
        it = man.get(ident)
        if it is None:
            continue
        full = full_frame(it)
        if full is None:
            continue
        H, W = full.shape[:2]
        t0 = time.time()
        finder.set_image(full); patch(finder)
        r = finder(text=['person'])[0]
        t['find'] += time.time() - t0
        ms, boxes, conf = [], [], []
        if r.masks is not None:
            mm = r.masks.data.cpu().numpy() > 0.5
            bb = r.boxes.xyxy.cpu().numpy(); cc = r.boxes.conf.cpu().numpy()
            for m, b, c in zip(mm, bb, cc):
                if m.shape != (H, W):
                    m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                ms.append(m); boxes.append(b); conf.append(float(c))
        keep = dedup(ms, conf)
        ms = [ms[k] for k in keep]; boxes = [boxes[k] for k in keep]
        p21, p3 = [], []
        for b in boxes:
            x1, y1, x2, y2 = [float(v) for v in b]
            X1, Y1 = max(0, int(x1 - MARGIN)), max(0, int(y1 - MARGIN))
            X2, Y2 = min(W, int(x2 + MARGIN)), min(H, int(y2 + MARGIN))
            crop = np.ascontiguousarray(full[Y1:Y2, X1:X2])
            cb = [x1 - X1, y1 - Y1, x2 - X1, y2 - Y1]
            t0 = time.time()
            with torch.autocast('cuda', dtype=torch.float16):
                rr = sam21.predict(crop, bboxes=[cb], imgsz=1024, verbose=False)[0]
            m21 = (rr.masks.data[0].float().cpu().numpy() > 0.5) if rr.masks is not None else np.zeros(crop.shape[:2], bool)
            if m21.shape != crop.shape[:2]:
                m21 = cv2.resize(m21.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            t['sam21'] += time.time() - t0
            p21.append((b, m21, (X1, Y1)))
            t0 = time.time()
            try:
                boxer3.set_image(crop); patch(boxer3)
                r3 = boxer3(bboxes=[cb])[0]
                m3 = (r3.masks.data[0].float().cpu().numpy() > 0.5) if r3.masks is not None else np.zeros(crop.shape[:2], bool)
                if m3.shape != crop.shape[:2]:
                    m3 = cv2.resize(m3.astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            except Exception as e:
                rep['sam3_box_error'] = repr(e)[:300]
                m3 = np.zeros(crop.shape[:2], bool)
            t['sam3box'] += time.time() - t0
            p3.append((b, m3, (X1, Y1)))
        truth = gold.gold(ident)
        truth_all.append(truth)
        direct_all.append([to_small(m) for m in ms])
        c21_all.append([to_small(m) for m in put_together((H, W), p21)])
        c3_all.append([to_small(m) for m in put_together((H, W), p3)])
        frames_kept.append((ident, cv2.resize(full, (1280, 720), interpolation=cv2.INTER_AREA)))
    n = max(1, len(truth_all))
    rep['frames'] = len(truth_all)
    rep['seconds_per_frame'] = {k: round(v / n, 3) for k, v in t.items()}
    rep['sam3_as_is'] = score(truth_all, direct_all)
    rep['sam3_then_sam21_closeup'] = score(truth_all, c21_all)
    rep['sam3_then_sam3_closeup'] = score(truth_all, c3_all)
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'sam3_crop.json', 'w'), indent=1)
    # the sheet: the most crowded frames
    import sam3_sheet as S
    order = sorted(range(len(truth_all)), key=lambda k: -len(truth_all[k]))[:n_sheet]
    rows = []
    for k in order:
        ident, img = frames_kept[k]
        truth = truth_all[k]
        panels = [S.label(S.draw(img, truth), 'owner: %d people' % len(truth))]
        for name, ms in (('SAM 3 as is', direct_all[k]), ('SAM 3 + SAM 2.1 close-up', c21_all[k]), ('SAM 3 + SAM 3 close-up', c3_all[k])):
            ms = [m for m in ms if m.sum() >= gold.MIN_PX]
            pairs, iou = gold.match(truth, ms)
            hit_t = {i for i, _ in pairs}; hit_m = {j for _, j in pairs}
            false = [j for j in range(len(ms)) if j not in hit_m]
            missed = [truth[i] for i in range(len(truth)) if i not in hit_t]
            med = np.median([iou[i, j] for i, j in pairs]) if pairs else 0
            panels.append(S.label(S.draw(img, ms, missed, false), '%s: %d/%d, false %d, contour %.3f' % (name, len(pairs), len(truth), len(false), med)))
        rows.append(np.hstack([cv2.resize(p, (800, 450), interpolation=cv2.INTER_AREA) for p in panels]))
    cv2.imwrite(str(ROOT / 'data' / 'logs' / 'sam3_crop.jpg'), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
