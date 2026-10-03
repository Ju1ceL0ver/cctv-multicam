"""Short clips for tracking and identity: every person outlined on every frame, one number per person.

Where: the busy moments -- draft frames with 3 or more people -- of every recorded day but 18.09 (the
honest exam day) and the files of the held-out test (no frame near a test frame is trained on).
What: CLIP_FRAMES frames in a row at 12.5 fps from the raw recording (25 fps, every 2nd), one camera.
Teachers, as the drafts: yolo26x-seg @1536 finds the people, SAM 2.1 Large outlines them from their boxes
(fp16 autocast), the nearer person painted last. Numbers: a person keeps his number from one frame to
the next when his mask is the best match (Hungarian on mask IoU >= LINK_IOU); 0.08 s apart that is
nearly unambiguous. A person nobody matches gets a new number.

usage: clips.py build [N] [--wait-for SCRIPT]   -> data/clips/<clip>/NNN.jpg (1280x720), NNN.png (label =
                                                   person number), clip.json; data/clips/index.json"""
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
W, H = 1280, 720
CLIP_FRAMES = 24
STEP = 2                      # raw frames between clip frames: 25 fps -> 12.5 fps
MIN_PEOPLE = 3
LINK_IOU = 0.3
MIN_PX = 60
EXAM_DAY = '20260918'


def busy_moments(root=ROOT):
    """(day, cam, segment file name, second, people) of the draft frames with >= MIN_PEOPLE people, outside
    the exam day and the held-out files."""
    seg = Path(root) / 'data' / 'seg_datasets' / 'rf_20260925'
    held_files = {p.stem.rsplit('_', 1)[0] for p in (seg / 'valid' / 'images').glob('*.jpg')}
    out = []
    for lab in (seg / 'train' / 'labels').glob('*.txt'):
        ident = lab.stem
        day, cam, hms, idx, sec = ident.split('_')
        if day == EXAM_DAY or ident.rsplit('_', 1)[0] in held_files:
            continue
        n = sum(1 for _ in open(lab))
        if n >= MIN_PEOPLE:
            out.append((day, cam, '%s_%s.mp4' % (hms, idx), int(sec), n))
    return out


def read_run(path, second, frames=CLIP_FRAMES, step=STEP):
    """frames consecutive raw frames (every `step`-th) from `second` of a recording file, full size."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second * 1000.0))
    out, k = [], 0
    while len(out) < frames:
        ok = cap.grab()
        if not ok:
            break
        if k % step == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            out.append(f)
        k += 1
    cap.release()
    return out


def link(prev, cur, next_id):
    """Carry numbers from the previous label map to the current one by mask IoU (Hungarian)."""
    from scipy.optimize import linear_sum_assignment
    pv = [v for v in np.unique(prev) if v]
    cv = [v for v in np.unique(cur) if v]
    out = np.zeros_like(cur, np.uint16)
    if not cv:
        return out, next_id, {}
    iou = np.zeros((len(cv), len(pv)))
    for i, a in enumerate(cv):
        ma = cur == a
        for j, b in enumerate(pv):
            mb = prev == b
            inter = (ma & mb).sum()
            if inter:
                iou[i, j] = inter / (ma | mb).sum()
    got = {}
    if pv:
        r, c = linear_sum_assignment(-iou)
        got = {cv[i]: pv[j] for i, j in zip(r, c) if iou[i, j] >= LINK_IOU}
    mapping = {}
    for a in cv:
        if a in got:
            mapping[a] = got[a]
        else:
            mapping[a] = next_id
            next_id += 1
        out[cur == a] = mapping[a]
    return out, next_id, {int(k): float(iou[cv.index(k), pv.index(v)]) for k, v in got.items()}


class Teacher:
    def __init__(self):
        import seg_compare
        from ultralytics import SAM, YOLO
        self.yolo = YOLO(r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models\yolo26x-seg.pt', task='segment')
        self.sam = SAM(seg_compare.SAM_WEIGHTS)

    def label_map(self, frame):
        """Label map 1280x720, 1..n one per person (the nearer one on top)."""
        import cv2
        import torch
        r = self.yolo.predict(frame, imgsz=1536, conf=0.25, classes=[0], verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        labels = np.zeros((H, W), np.uint16)
        if not len(boxes):
            return labels
        with torch.autocast('cuda', dtype=torch.float16):
            res = self.sam.predict(frame, bboxes=boxes.tolist(), imgsz=1024, verbose=False)[0]
        masks = (res.masks.data.float().cpu().numpy() > 0.5) if res.masks is not None else []
        for k in np.argsort(boxes[:, 3]):
            if k < len(masks):
                m = cv2.resize(masks[k].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
                labels[m] = int(k) + 1
        for v in np.unique(labels):                          # the specks nobody meant
            if v and (labels == v).sum() < MIN_PX:
                labels[labels == v] = 0
        return labels


def build(n=150, root=ROOT, seed=0, log=print):
    import cv2
    from rawsource import segments
    rng = random.Random(seed)
    out_dir = Path(root) / 'data' / 'clips'
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / 'index.json'
    index = json.load(open(index_path)) if index_path.exists() else {}
    moments = busy_moments(root)
    # spread: at most one clip per recording file and camera, the busiest first within a random order
    rng.shuffle(moments)
    moments.sort(key=lambda m: -m[4])
    seen, chosen = set(), []
    for m in moments:
        key = (m[0], m[1], m[2])
        if key in seen:
            continue
        seen.add(key); chosen.append(m)
    rng.shuffle(chosen)
    teacher = Teacher()
    paths = {}
    t0, made = time.time(), 0
    for day, cam, fname, sec, people in chosen:
        if made >= n:
            break
        cid = '%s_%s_%s_%04d' % (day, cam, fname[:-4], sec)
        if cid in index:
            continue
        if (day, cam) not in paths:
            paths[(day, cam)] = {os.path.basename(p): p for p, _ in segments(cam, day)}
        path = paths[(day, cam)].get(fname)
        if not path:
            continue
        start = max(0, sec - CLIP_FRAMES * STEP / 25 / 2)            # the draft's moment in the middle
        run = read_run(path, start)
        if len(run) < CLIP_FRAMES:
            continue
        folder = out_dir / cid
        folder.mkdir(exist_ok=True)
        prev, next_id, links, counts = None, 1, [], []
        for k, frame in enumerate(run):
            lab = teacher.label_map(frame)
            if prev is None:
                ids = np.zeros_like(lab)
                for v in np.unique(lab):
                    if v:
                        ids[lab == v] = next_id; next_id += 1
                got = {}
            else:
                ids, next_id, got = link(prev, lab, next_id)
            prev = ids
            links.append(got); counts.append(int(len(np.unique(ids)) - 1))
            cv2.imwrite(str(folder / ('%03d.jpg' % k)), cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(folder / ('%03d.png' % k)), ids)
        meta = {'day': day, 'cam': cam, 'file': fname, 'start_s': round(start, 2), 'fps': 25 / STEP, 'frames': len(run),
                'people_per_frame': counts, 'identities': next_id - 1, 'teacher': 'yolo26x-seg@1536 + sam2.1_l (fp16)',
                'link': 'Hungarian on mask IoU >= %.2f between neighbouring frames' % LINK_IOU}
        json.dump(meta, open(folder / 'clip.json', 'w'), indent=1)
        index[cid] = {k: meta[k] for k in ('day', 'cam', 'frames', 'identities')}
        json.dump(index, open(index_path, 'w'), indent=1)
        made += 1
        if made % 10 == 0:
            log('%d clips, %.1f min' % (made, (time.time() - t0) / 60))
    log('done: %d new clips, %d in all, %.1f min' % (made, len(index), (time.time() - t0) / 60))


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'build':
        n = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 150
        if '--wait-for' in sys.argv:                          # let another GPU job finish first
            import subprocess
            what = sys.argv[sys.argv.index('--wait-for') + 1]
            def busy():
                out = subprocess.run(['powershell', '-c', 'Get-CimInstance Win32_Process -Filter "name=\'python.exe\'" | %{ $_.CommandLine }'],
                                     capture_output=True, text=True).stdout
                return any(what in l and 'clips.py' not in l for l in out.splitlines())   # not our own command line
            while busy():
                time.sleep(30)
        build(n)
    else:
        print(__doc__)
