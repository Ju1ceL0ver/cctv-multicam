"""Add segmentation polygons to an already-detected clip WITHOUT changing detection indices.

The detector runs again on the same frames; each new mask is matched to the existing
detection by IoU, so every stored detection gains its outline while the numbering the
manual labels are attached to stays exactly as it was.

usage: polys_pass.py CLIP [TAG]  -> data/raw_clips/CLIP/polys_<tag>.npz (pts + per-detection offsets)"""
import sys, os, time, json, numpy as np, cv2
from datetime import datetime
from ultralytics import YOLO
from scipy.optimize import linear_sum_assignment
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from clipdata import load
from rawsource import Stream, FPS
from imtrack import iou_matrix

MAXPTS = 64


def main():
    clip = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else 'yolo26x-seg'
    dets, feats, meta, _emb = load(clip, tag, apply_sync=False)
    model = YOLO(meta['weights'], task='segment')
    out = {}
    for cam in ('cam1', 'cam2'):
        d = dets[cam]
        frame_of = np.round(d[:, 0] * FPS).astype(int)
        by_frame = {}
        for i, f in enumerate(frame_of):
            by_frame.setdefault(int(f), []).append(i)
        pts = [np.zeros((0, 2), np.float32)] * len(d)
        s = Stream(cam, meta['day']); s.seek(datetime.fromisoformat(meta['start']))
        t0 = time.time(); matched = 0
        for k in range(int(meta['frames'])):
            t, fr = s.read()
            if fr is None:
                break
            idx = by_frame.get(k)
            if not idx:
                continue
            r = model.predict(source=fr, imgsz=meta['imgsz'], conf=0.25, classes=[0], half=True,
                              retina_masks=False, verbose=False)[0]
            if r.boxes is None or not len(r.boxes) or r.masks is None:
                continue
            B = r.boxes.xyxy.cpu().numpy()
            M = iou_matrix(d[idx, 1:5], B)
            rr, cc = linear_sum_assignment(-M)
            for i, j in zip(rr, cc):
                if M[i, j] < 0.6:
                    continue
                p = r.masks.xy[j]
                if p is None or len(p) < 3:
                    continue
                step = max(1, len(p) // MAXPTS)
                pts[idx[i]] = np.asarray(p[::step], np.float32)
                matched += 1
            if k % 2000 == 0:
                print('%s %s frame %d matched %d (%.1f fps)' % (time.strftime('%H:%M:%S'), cam, k, matched,
                                                                (k + 1) / (time.time() - t0)), flush=True)
        off = np.zeros(len(d) + 1, np.int64)
        for i, p in enumerate(pts):
            off[i + 1] = off[i] + len(p)
        out[cam + '_pts'] = np.concatenate(pts) if len(pts) else np.zeros((0, 2), np.float32)
        out[cam + '_off'] = off
        print('%s: %d/%d detections got an outline' % (cam, matched, len(d)), flush=True)
    np.savez_compressed(os.path.join('data', 'raw_clips', clip, 'polys_%s.npz' % tag), **out)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
