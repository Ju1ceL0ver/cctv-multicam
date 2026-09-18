"""Turn reviewed clips into a segmentation dataset for the small model.

Only what the owner has looked at: a detection whose piece he marked "не разобрать"
(two people in one box, or unreadable) is dropped, and clips he has not reviewed are
skipped entirely. The teacher's outline is the target -- this is distillation of
segmentation, there is no separate detection task.

Frames are written at 0.625 scale (1600x900): the student will run well below the
teacher's 1536, and full-resolution frames would cost gigabytes for no gain.

usage: export_seg_dataset.py [OUT_DIR] [--gap SECONDS] [--val CLIP[,CLIP]]"""
import os, sys, json, glob, numpy as np, cv2
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from rawsource import Stream, FPS

TAG = 'yolo26x-seg'
SCALE = 0.625
CLIPS = os.path.join(ROOT, 'data', 'raw_clips')


def reviewed(clip):
    """(labels, pieces) if the owner has been through this clip, else None."""
    d = os.path.join(CLIPS, clip)
    lp = os.path.join(d, 'gt_manual.json')
    pp = os.path.join(d, 'pieces_%s.json' % TAG)
    if not (os.path.exists(lp) and os.path.exists(pp)):
        return None
    labels = json.load(open(lp, encoding='utf-8'))
    pieces = json.load(open(pp))
    if len(labels) < 0.5 * len(pieces):        # half-reviewed clips are not worth the doubt
        return None
    return labels, pieces


def bad_detections(labels, pieces):
    """Detections the owner could not read: excluded from the training targets."""
    bad = {'cam1': set(), 'cam2': set()}
    seen = {'cam1': set(), 'cam2': set()}
    for p in pieces:
        lab = labels.get(str(p['piece']))
        dets = p['dets']
        if lab == '?' or lab is None:
            bad[p['cam']].update(int(x) for x in dets)
        seen[p['cam']].update(int(x) for x in dets)
    return bad, seen


def main():
    out = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else os.path.join(ROOT, 'data', 'seg_dataset')
    args = sys.argv[1:]
    gap = float(args[args.index('--gap') + 1]) if '--gap' in args else 1.0   # seconds between kept frames
    val = (args[args.index('--val') + 1] if '--val' in args else 'c183400').split(',')
    clips = sorted(os.path.basename(d) for d in glob.glob(os.path.join(CLIPS, 'c*')))
    for split in ('train', 'val'):
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(out, split, sub), exist_ok=True)
    counts = {'train': 0, 'val': 0}
    instances = {'train': 0, 'val': 0}
    for clip in clips:
        r = reviewed(clip)
        if r is None:
            continue
        labels, pieces = r
        d = os.path.join(CLIPS, clip)
        dets = dict(np.load(os.path.join(d, 'dets_%s.npz' % TAG)))
        polys = dict(np.load(os.path.join(d, 'polys_%s.npz' % TAG)))
        meta = json.load(open(os.path.join(d, 'meta_%s.json' % TAG)))
        bad, seen = bad_detections(labels, pieces)
        split = 'val' if clip in val else 'train'
        stamp = os.path.join(out, split, 'labels', '.%s.done' % clip)
        if os.path.exists(stamp) and open(stamp).read().strip() == str(len(labels)):
            done = len(glob.glob(os.path.join(out, split, 'images', '%s_*.jpg' % clip)))
            counts[split] += done
            print('%s -> %s (already there, %d frames)' % (clip, split, done), flush=True)
            continue
        start = datetime.fromisoformat(meta['start'])
        for cam in ('cam1', 'cam2'):
            D = dets[cam]
            if not len(D):
                continue
            frame_of = np.round(D[:, 0] * FPS).astype(int)
            by_frame = {}
            for i, f in enumerate(frame_of):
                if i in bad[cam] or i not in seen[cam]:
                    continue
                by_frame.setdefault(int(f), []).append(i)
            wanted, last = [], -1e9        # neighbouring frames of a 25 fps camera are the same picture
            for f in sorted(by_frame):
                if f / FPS - last >= gap:
                    wanted.append(f); last = f / FPS
            if not wanted:
                continue
            s = Stream(cam, meta['day']); s.seek(start)
            want = set(wanted)
            pts, off = polys[cam + '_pts'], polys[cam + '_off']
            for k in range(int(meta['frames'])):
                t, fr = s.read()
                if fr is None:
                    break
                if k not in want:
                    continue
                h, w = fr.shape[:2]
                img = cv2.resize(fr, (int(w * SCALE), int(h * SCALE)))
                name = '%s_%s_%06d' % (clip, cam, k)
                rows = []
                for i in by_frame[k]:
                    a, b = off[i], off[i + 1]
                    if b - a < 3:
                        continue
                    p = pts[a:b].astype(np.float32)
                    p[:, 0] = np.clip(p[:, 0] / w, 0, 1); p[:, 1] = np.clip(p[:, 1] / h, 0, 1)
                    rows.append('0 ' + ' '.join('%.5f %.5f' % (x, y) for x, y in p))
                if not rows:
                    continue
                cv2.imwrite(os.path.join(out, split, 'images', name + '.jpg'), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
                open(os.path.join(out, split, 'labels', name + '.txt'), 'w').write('\n'.join(rows) + '\n')
                counts[split] += 1; instances[split] += len(rows)
        open(stamp, 'w').write(str(len(labels)))    # re-export only when the review changed
        print('%s -> %s (%d frames so far)' % (clip, split, counts[split]), flush=True)
    yaml = os.path.join(out, 'data.yaml')
    open(yaml, 'w').write('path: %s\ntrain: train/images\nval: val/images\nnames:\n  0: person\n' % out.replace('\\', '/'))
    print('dataset: train %d frames / %d people, val %d frames / %d people -> %s'
          % (counts['train'], instances['train'], counts['val'], instances['val'], yaml))


if __name__ == '__main__':
    main()
