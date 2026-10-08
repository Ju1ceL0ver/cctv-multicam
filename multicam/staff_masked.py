"""/staff on the person's mask, not the box (08.10.2026): the teacher ReID vectors (TransReID clothes 3840, CSCI
shape 1024) of each person's crop with everything outside the person's mask painted grey -- so a neighbour inside
the box does not leak into the vector.
  machine-draft people: the mask from the draft label map (value of the person), crop = box + 5 % (as teacher_emb)
  counter snapshots: the counter's full frame, people found by the student yolo26n-seg, the mask that fits the
    counter's box best (IoU of boxes >= 0.3; otherwise the plain crop) -- its grey-background crop is also kept as
    data/staff/counter_masked/<id>.jpg for the page
-> data/staff/masked_emb.npz {ids, cloth, shape, masked}; then `compare`: every day scored by a model of the other
   days, box vectors against mask vectors, on the owner's answers.
usage (venv_rfdetr, the card): staff_masked.py run | compare"""
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'staff'
MARGIN, GREY = 0.05, 128


def crop_masked(img, mask, box):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    mx, my = MARGIN * (x2 - x1) + 1, MARGIN * (y2 - y1) + 1
    a, b = max(0, int(x1 - mx)), max(0, int(y1 - my))
    c, d = min(w, int(x2 + mx)), min(h, int(y2 + my))
    if c - a < 8 or d - b < 16:
        return None
    crop = img[b:d, a:c].copy()
    if mask is not None:
        crop[~mask[b:d, a:c]] = GREY
    return crop


def run():
    import cv2
    import rate
    import staff
    import track_emb as T
    from ultralytics import YOLO
    F = staff.feats(str(ROOT))
    frames = rate.index(str(ROOT), force=True)
    embed = T.teachers()
    seg = YOLO(str(ROOT / 'runs' / 'student_seg_all_n' / 'weights' / 'best.pt'))
    (OUT / 'counter_masked').mkdir(parents=True, exist_ok=True)
    ids, cloth, shape, masked = [], [], [], []
    batch, keys, flags = [], [], []

    def flush():
        if not batch:
            return
        v1, v2 = embed(batch)
        ids.extend(keys); cloth.extend(v1); shape.extend(v2); masked.extend(flags)
        batch.clear(); keys.clear(); flags.clear()

    cache = {}
    t0 = time.time()
    for n, pid in enumerate(F['ids']):
        m = F['meta'][pid]
        crop, ok = None, False
        if m.get('src') == 'counter':
            full = cv2.imread(m['full']) if m.get('full') else None
            if full is not None:
                bx = np.array(m['box_raw'], float)
                r = seg.predict(full, conf=0.25, classes=[0], verbose=False, retina_masks=True)[0]
                best = None
                if r.masks is not None:
                    for mk, b in zip(r.masks.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
                        iw = max(0, min(b[2], bx[2]) - max(b[0], bx[0])); ih = max(0, min(b[3], bx[3]) - max(b[1], bx[1]))
                        iou = iw * ih / max(1e-6, (b[2] - b[0]) * (b[3] - b[1]) + (bx[2] - bx[0]) * (bx[3] - bx[1]) - iw * ih)
                        if iou >= 0.3 and (best is None or iou > best[0]):
                            best = (iou, mk > 0.5)
                mk = None
                if best is not None:
                    mk = best[1]
                    if mk.shape != full.shape[:2]:
                        mk = cv2.resize(mk.astype(np.uint8), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
                crop = crop_masked(full, mk, bx)
                ok = mk is not None
                if crop is not None and ok:
                    cv2.imwrite(str(OUT / 'counter_masked' / (pid + '.jpg')), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        elif m.get('frame') in frames:
            fr = m['frame']
            if fr not in cache:
                cache.clear()
                image, draft = frames[fr]
                img = cv2.imread(str(image))
                lab = cv2.imread(str(draft), cv2.IMREAD_UNCHANGED)
                if lab is not None and lab.ndim == 3:
                    lab = lab[:, :, 0]
                if img is not None and lab is not None and lab.shape != img.shape[:2]:
                    lab = cv2.resize(lab, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
                cache[fr] = (img, lab)
            img, lab = cache[fr]
            if img is not None and lab is not None:
                sx, sy = img.shape[1] / 1280.0, img.shape[0] / 720.0
                box = np.array(m['box'], float) * [sx, sy, sx, sy]
                crop = crop_masked(img, lab == m['value'], box)
                ok = True
        if crop is None:
            continue
        batch.append(crop); keys.append(pid); flags.append(ok)
        if len(batch) >= 64:
            flush()
        if n % 2000 == 0:
            print(n, 'of', len(F['ids']), '%.0f s' % (time.time() - t0), flush=True)
    flush()
    np.savez_compressed(OUT / 'masked_emb.npz', ids=np.array(ids), cloth=np.array(cloth, np.float32),
                        shape=np.array(shape, np.float32), masked=np.array(masked))
    print('saved', len(ids), 'masked', int(np.sum(masked)), '%.0f s' % (time.time() - t0), flush=True)


def sam_counter():
    """08.10: the counter snapshots' masks again, by SAM 2.1 L prompted with the counter's own box (yolo26n-seg masks were
    rough); their vectors in masked_emb.npz and the grey crops are replaced, the drafts' rows stay."""
    import cv2
    import torch
    import staff
    import track_emb as T
    from ultralytics import SAM
    F = staff.feats(str(ROOT))
    z = np.load(OUT / 'masked_emb.npz')
    ids = [str(i) for i in z['ids']]
    cloth, shape, masked = z['cloth'].copy(), z['shape'].copy(), z['masked'].copy()
    at = {i: k for k, i in enumerate(ids)}
    sam = SAM('sam2.1_l.pt')
    embed = T.teachers()
    todo = [p for p in F['ids'] if F['meta'][p].get('src') == 'counter' and F['meta'][p].get('full')]
    batch, keys = [], []

    def flush():
        if not batch:
            return
        v1, v2 = embed(batch)
        for k_, a_, b_ in zip(keys, v1, v2):
            if k_ in at:
                cloth[at[k_]], shape[at[k_]], masked[at[k_]] = a_, b_, True
            else:
                at[k_] = len(ids); ids.append(k_)
                cloth.resize((len(ids), cloth.shape[1]), refcheck=False); shape.resize((len(ids), shape.shape[1]), refcheck=False)
                cloth[-1], shape[-1] = a_, b_
                masked.resize(len(ids), refcheck=False); masked[-1] = True
        batch.clear(); keys.clear()

    t0 = time.time()
    for n, pid in enumerate(todo):
        m = F['meta'][pid]
        full = cv2.imread(m['full'])
        if full is None:
            continue
        bx = [float(v) for v in m['box_raw']]
        with torch.autocast('cuda', dtype=torch.float16):
            px, py = (bx[0] + bx[2]) / 2, bx[1] + 0.35 * (bx[3] - bx[1])     # a point on the torso: the person, not a neighbour
            r = sam.predict(full, bboxes=[bx], points=[[px, py]], labels=[1], verbose=False)[0]
        if r.masks is None:
            continue
        mk = r.masks.data[0].cpu().numpy() > 0.5
        if mk.shape != full.shape[:2]:
            mk = cv2.resize(mk.astype(np.uint8), (full.shape[1], full.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        n_, lab, st, _ = cv2.connectedComponentsWithStats(mk.astype(np.uint8))
        if n_ > 2:                                        # keep the piece under the torso point (else the biggest)
            hit = lab[min(int(py), lab.shape[0] - 1), min(int(px), lab.shape[1] - 1)]
            mk = lab == (hit if hit else 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])))
        crop = crop_masked(full, mk, bx)
        if crop is None:
            continue
        cv2.imwrite(str(OUT / 'counter_masked' / (pid + '.jpg')), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        batch.append(crop); keys.append(pid)
        if len(batch) >= 64:
            flush()
        if n % 200 == 0:
            print(n, 'of', len(todo), '%.0f s' % (time.time() - t0), flush=True)
    flush()
    np.savez_compressed(OUT / 'masked_emb.npz', ids=np.array(ids), cloth=cloth, shape=shape, masked=masked)
    print('saved', len(todo), '%.0f s' % (time.time() - t0), flush=True)


def compare():
    """box vectors (feats.npz) against mask vectors, same answers, same PCA sizes, same net, by days."""
    import staff
    from sklearn.neural_network import MLPClassifier
    F, r = staff.feats(str(ROOT)), staff.labels(str(ROOT))
    z = np.load(OUT / 'masked_emb.npz')
    at = {str(i): k for k, i in enumerate(z['ids'])}
    ans = [(i, v['label']) for i, v in r.items() if v['label'] in (1, 2) and i in F['row'] and i in at]
    y = np.array([a == 2 for _, a in ans])
    days = np.array([F['meta'][i]['day'] for i, _ in ans])
    src = np.array([F['meta'][i].get('src', 'draft') for i, _ in ans])
    Xbox = F['X'][[F['row'][i] for i, _ in ans]]
    # the mask vectors through the same recipe as staff.build (PCA over everybody, then the answered rows)
    C = staff._norm(staff._pca(staff._norm(z['cloth']), staff.PCA_CLOTH))
    S = staff._norm(staff._pca(staff._norm(z['shape']), staff.PCA_SHAPE))
    rows = [at[i] for i, _ in ans]
    Xmask = np.concatenate([C[rows], S[rows], Xbox[:, -4:]], 1)
    for name, X in (('box', Xbox), ('mask', Xmask)):
        P = np.zeros(len(y))
        for d in sorted(set(days)):
            te = days == d
            P[te] = MLPClassifier((256, 64), max_iter=400, early_stopping=True, random_state=0).fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
        q = P >= 0.5
        cnt = src == 'counter'
        print('%-5s acc %.4f | staff->cust %d cust->staff %d | counter snapshots (%d) %s' % (
            name, (q == y).mean(), int((~q & y).sum()), int((q & ~y).sum()), int(cnt.sum()),
            '%.4f' % (q[cnt] == y[cnt]).mean() if cnt.any() else '-'), flush=True)


if __name__ == '__main__':
    {'run': run, 'compare': compare, 'sam': sam_counter}[sys.argv[1]]()
