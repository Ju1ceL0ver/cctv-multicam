"""SAM 3 as the teacher, measured before it replaces yolo26x-seg + SAM 2.1 (27.09).

1. exam: the owner's 18.09 /paint frames (52) -- SAM 3 finds and outlines every "person" by itself (text
   prompt, no detector) at a few confidence thresholds; scored as gold.evaluate (found at mask IoU >= 0.5,
   false, contour, small people). The current teacher scored 96.6 % / precision 84.0 % / contour 0.966.
2. speed per frame on the 3060.
3. video: one of our clips (24 frames, 12.5 fps) through SAM 3's video predictor with the same prompt --
   how many identities it keeps, against our mask-overlap linking.

usage: sam3_try.py [WEIGHTS]   -> data/logs/sam3_try.json"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def masks_of(result, shape):
    import cv2
    if result.masks is None:
        return [], []
    m = result.masks.data.cpu().numpy() > 0.5
    conf = result.boxes.conf.cpu().numpy() if result.boxes is not None else np.ones(len(m))
    out = []
    for x in m:
        if x.shape != shape:
            x = cv2.resize(x.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        out.append(x)
    return out, conf


def main():
    import cv2
    import gold
    from ultralytics.models.sam import SAM3SemanticPredictor
    weights = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / 'data' / 'weights' / 'sam3' / 'sam3.pt')
    rep = {'weights': weights}
    out_path = ROOT / 'data' / 'logs' / 'sam3_try_1008.json'
    save = lambda: json.dump(rep, open(out_path, 'w'), indent=1)
    pred = SAM3SemanticPredictor(overrides=dict(conf=0.1, task='segment', mode='predict', model=weights, half=True, save=False, verbose=False, imgsz=1008))
    ids = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    per = []
    t_total = 0.0
    for ident in ids:
        img = str(gold.PAINT / ('%s.jpg' % ident))
        truth = gold.gold(ident)
        t0 = time.time()
        pred.set_image(img)
        if not hasattr(pred.model, 'mask_threshold'):       # Ultralytics 8.4.108 asks the SAM 3 model for it and it has none
            pred.model.mask_threshold = 0.0
        r = pred(text=['person'])[0]
        t_total += time.time() - t0
        ms, conf = masks_of(r, truth[0].shape if truth else (720, 1280))
        per.append((truth, ms, conf))
    rep['s_per_frame'] = round(t_total / max(1, len(ids)), 3)
    for thr in (0.2, 0.3, 0.4, 0.5):
        found = total = false = sf = st = 0
        ious = []
        for truth, ms, conf in per:
            pred_m = [m for m, c in zip(ms, conf) if c >= thr and m.sum() >= gold.MIN_PX]
            pairs, iou = gold.match(truth, pred_m)
            total += len(truth); found += len(pairs); false += len(pred_m) - len(pairs)
            ious += [iou[i, j] for i, j in pairs]
            hit = {i for i, _ in pairs}
            for i, t in enumerate(truth):
                ys = np.nonzero(t.any(1))[0]
                if ys[-1] - ys[0] < gold.SMALL:
                    st += 1; sf += i in hit
        rep['exam_conf_%.1f' % thr] = {'people': total, 'recall': round(found / max(1, total), 4), 'false': false,
                                        'precision': round(found / max(1, found + false), 4),
                                        'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
                                        'small_recall': round(sf / max(1, st), 4)}
        save()
    # video: one clip of ours
    try:
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        clips = sorted((ROOT / 'data' / 'clips').glob('*/clip.json'))
        best = max(clips, key=lambda p: max(json.load(open(p))['people_per_frame']))
        folder = best.parent
        frames = sorted(folder.glob('*.jpg'))
        vid = ROOT / 'data' / 'logs' / 'sam3_clip.mp4'
        h, w = cv2.imread(str(frames[0])).shape[:2]
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (w, h))
        for f in frames:
            vw.write(cv2.imread(str(f)))
        vw.release()
        vp = SAM3VideoSemanticPredictor(overrides=dict(conf=0.3, task='segment', mode='predict', model=weights, half=True, save=False, verbose=False, imgsz=1008))
        t0 = time.time()
        ids_per_frame = []
        vp.setup_model(None) if getattr(vp, 'model', None) is None else None
        if getattr(vp, 'model', None) is not None and not hasattr(vp.model, 'mask_threshold'):
            vp.model.mask_threshold = 0.0
        for r in vp(source=str(vid), text=['person'], stream=True):
            tid = r.boxes.id.cpu().numpy().astype(int).tolist() if r.boxes is not None and r.boxes.id is not None else []
            ids_per_frame.append(tid)
        rep['video'] = {'clip': folder.name, 'frames': len(ids_per_frame), 's_per_frame': round((time.time() - t0) / max(1, len(ids_per_frame)), 3),
                        'identities': len({i for f in ids_per_frame for i in f}), 'people_per_frame': [len(f) for f in ids_per_frame],
                        'ours_identities': json.load(open(best))['identities'], 'ours_people_per_frame': json.load(open(best))['people_per_frame']}
    except Exception as e:
        import traceback
        rep['video'] = {'error': traceback.format_exc()[-800:]}
    save()
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
