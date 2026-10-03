"""SAM 3.1 (text "person") on single frames: the owner's /paint frames first (scored against his masks), then
every draft frame of data/seg_datasets (the 13 thousand varied frames of 13-23.09 that yolo26x + SAM 2.1 L
outlined) -- so that the stills and the 15-minute windows have one teacher, and so that two independent
teachers' drafts of the same frame can be compared.

Each frame is its own one-frame session of the video model (the same weights and prompt as the windows):
squeezed to 1008x1008, masks stretched back to 1280x720 and written as a label map like the old drafts
(bigger people first, so a small one stays on top).

  data/sam31_stills/paint/<id>.png, drafts/<id>.png     label maps, 0 = nobody
  data/sam31_stills/probs.jsonl                         {id, set, probs: [SAM's confidence per label]}
  data/sam31_stills/report.json                         exam and all-painted scores, speed, progress

Waits for 21:02, stops at 09:40 (the live counter's card by day); started again it goes on where it stopped.
Order: the frames the owner scored on /rate first, the rest shuffled by a hash of the id.

usage (venv_sam3): sam31_stills.py [now]      'now' skips the wait for the night"""
import datetime
import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'sam31_stills'
W, H = 1280, 720
SAM_IN = 1008
GATE = 0.85            # the exam's recall below this: the stills are not worth a night, stop after the exam


def night():
    t = datetime.datetime.now().time()
    return t >= datetime.time(21, 2) or t < datetime.time(9, 40)


def one(pred, img, folder):
    """[(prob, bool mask HxW)] of one BGR frame."""
    import cv2
    import torch
    import sam31_video as SV
    cv2.imwrite(str(folder / '00000.jpg'), cv2.resize(img, (SAM_IN, SAM_IN), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    sid = None
    try:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sid = pred.handle_request(dict(type='start_session', resource_path=str(folder), offload_video_to_cpu=True))['session_id']
            first = pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
        got = SV.masks_of(first.get('outputs', {}) or {}, (H, W))
    finally:
        if sid is not None:
            try:
                pred.handle_request(dict(type='close_session', session_id=sid))
            except Exception:
                pass
    return [(p, m) for _, p, m in got if m.any()]


def write(kind, ident, got):
    import cv2
    got = sorted(got, key=lambda x: -int(x[1].sum()))[:255]
    lab = np.zeros((H, W), np.uint8)
    for k, (_, m) in enumerate(got):
        lab[m] = k + 1
    cv2.imwrite(str(OUT / kind / ('%s.png' % ident)), lab)
    with open(OUT / 'probs.jsonl', 'a') as f:
        f.write(json.dumps({'id': ident, 'set': kind, 'probs': [round(p, 4) for p, _ in got]}) + '\n')


def score(results, thr):
    """results: [(truth masks, [(prob, mask)])] -> the exam's numbers (gold.evaluate's way)."""
    import gold
    found = total = false = sf = st = 0
    ious = []
    for truth, got in results:
        pred = [m for p, m in got if p >= thr and m.sum() >= gold.MIN_PX]
        pairs, iou = gold.match(truth, pred)
        total += len(truth); found += len(pairs); false += len(pred) - len(pairs)
        ious += [float(iou[i, j]) for i, j in pairs]
        hit = {i for i, _ in pairs}
        for i, t in enumerate(truth):
            ys = np.nonzero(t.any(1))[0]
            if ys[-1] - ys[0] < gold.SMALL:
                st += 1; sf += i in hit
    return {'thr': thr, 'frames': len(results), 'people': total, 'recall': round(found / max(1, total), 4), 'false': false,
            'precision': round(found / max(1, found + false), 4), 'mask_iou_median': round(float(np.median(ious)), 4) if ious else None,
            'small_recall': round(sf / max(1, st), 4)}


def jobs():
    """[(set, id, image path)]: the owner's frames (the exam first), then the drafts -- scored ones first."""
    import gold
    import rate
    exam = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    painted = exam + [i for i in gold.done_ids() if i not in set(exam)]
    idx = rate.index(str(ROOT), force=True)
    rated = set(rate.ratings(str(ROOT)))
    drafts = sorted(idx, key=lambda i: (i not in rated, hashlib.md5(i.encode()).hexdigest()))
    return [('paint', i, gold.PAINT / ('%s.jpg' % i)) for i in painted] + [('drafts', i, idx[i][0]) for i in drafts]


def report():
    """The written label maps of the owner's frames against his masks -> report.json (exam, painted)."""
    import cv2
    import gold
    probs = {}
    if (OUT / 'probs.jsonl').exists():
        for line in open(OUT / 'probs.jsonl'):
            r = json.loads(line)
            if r['set'] == 'paint':
                probs[r['id']] = r['probs']
    exam = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']

    def res(ids):
        out = []
        for i in ids:
            p = OUT / 'paint' / ('%s.png' % i)
            if not p.exists() or i not in probs:
                continue
            lab = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            out.append((gold.gold(i), [(pr, lab == k + 1) for k, pr in enumerate(probs[i])]))
        return out
    rep_path = OUT / 'report.json'
    rep = json.load(open(rep_path)) if rep_path.exists() else {}
    rep['exam'] = [score(res(exam), t) for t in (0.0, 0.3, 0.5, 0.7)]
    rep['painted'] = [score(res([i for i in gold.done_ids() if i not in set(exam)]), t) for t in (0.0, 0.3, 0.5, 0.7)]
    rep.update(drafts_done=len(list((OUT / 'drafts').glob('*.png'))), at=time.strftime('%d.%m %H:%M'))
    json.dump(rep, open(rep_path, 'w'), indent=1)
    return rep


def main():
    import cv2
    import gold
    import rate
    import sam31_segment as SG
    if 'report' in sys.argv:
        print(json.dumps(report(), indent=1))
        return
    while 'now' not in sys.argv and not night():
        time.sleep(60)
    for kind in ('paint', 'drafts'):
        (OUT / kind).mkdir(parents=True, exist_ok=True)
    rep_path = OUT / 'report.json'
    rep = json.load(open(rep_path)) if rep_path.exists() else {}
    exam = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    painted = exam + [i for i in gold.done_ids() if i not in set(exam)]
    idx = rate.index(str(ROOT), force=True)
    rated = set(rate.ratings(str(ROOT)))
    drafts = sorted(idx, key=lambda i: (i not in rated, hashlib.md5(i.encode()).hexdigest()))
    jobs = [('paint', i, gold.PAINT / ('%s.jpg' % i)) for i in painted] + [('drafts', i, idx[i][0]) for i in drafts]
    pred = SG.build()
    folder = Path(tempfile.mkdtemp(prefix='sam31still_'))
    results, t0, n = {}, time.time(), 0
    try:
        for k, (kind, ident, path) in enumerate(jobs):
            if 'now' not in sys.argv and not night():
                print('morning: stop', flush=True)
                break
            out = OUT / kind / ('%s.png' % ident)
            if kind == 'drafts' and out.exists():
                continue
            if kind == 'paint' and out.exists() and 'exam' in rep and 'painted' in rep:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            if img.shape[:2] != (H, W):
                img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            got = one(pred, img, folder)
            write(kind, ident, got)
            n += 1
            if kind == 'paint':
                results[ident] = (gold.gold(ident), got)
                if ident == exam[-1] and all(i in results for i in exam):
                    rep['exam'] = [score([results[i] for i in exam], t) for t in (0.0, 0.3, 0.5, 0.7)]
                    rep['s_per_frame'] = round((time.time() - t0) / n, 3)
                    json.dump(rep, open(rep_path, 'w'), indent=1)
                    print('exam', rep['exam'], flush=True)
                    if rep['exam'][1]['recall'] < GATE:
                        print('exam recall below %.2f: stop' % GATE, flush=True)
                        return
                if ident == painted[-1] and all(i in results for i in painted):
                    rest = [i for i in painted if i not in set(exam)]
                    rep['painted'] = [score([results[i] for i in rest], t) for t in (0.0, 0.3, 0.5, 0.7)]
                    results.clear()
            if n % 50 == 0:
                rep.update(done=k + 1, of=len(jobs), s_per_frame=round((time.time() - t0) / n, 3), at=time.strftime('%d.%m %H:%M'))
                json.dump(rep, open(rep_path, 'w'), indent=1)
                print(k + 1, len(jobs), rep['s_per_frame'], flush=True)
        else:
            rep['finished'] = time.strftime('%d.%m %H:%M')
        rep.update(drafts_done=len(list((OUT / 'drafts').glob('*.png'))), drafts_all=len(drafts))
        json.dump(rep, open(rep_path, 'w'), indent=1)
        print('done', rep.get('drafts_done'), flush=True)
    finally:
        shutil.rmtree(folder, ignore_errors=True)


if __name__ == '__main__':
    main()
