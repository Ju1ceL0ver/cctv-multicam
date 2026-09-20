"""Short CPU-only context videos, rendered on demand and cached by source/version."""
import hashlib
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
import cv2
import numpy as np
from rawsource import Stream, FPS
from storage import file_lock


def preview(folder, piece, at=None):
    d = Path(folder)
    meta = json.loads((d / 'meta_yolo26x-seg.json').read_text())
    with np.load(d / 'dets_yolo26x-seg.npz') as z:
        rows = z[piece['cam']][piece['dets']].copy()
    polygon_file = d / 'polys_yolo26x-seg.npz'
    polygons = {}
    if polygon_file.exists():
        with np.load(polygon_file) as zp:
            pts, offsets = zp[piece['cam']+'_pts'], zp[piece['cam']+'_off']
            polygons = {j: pts[offsets[di]:offsets[di+1]].copy() for j, di in enumerate(piece['dets'])}
    at = float(at) if at is not None else float(rows[0, 0])
    at = min(max(float(rows[0, 0]), at), float(rows[-1, 0]))
    start = max(0, min(at - 2, float(meta['seconds']) - 4))
    end = min(float(meta['seconds']), start + 4)
    identity = [piece['cam'], piece['dets'], round(start, 2), meta['start'], 2]
    key = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
    target = d / 'review_media' / (key + '.mp4')
    with file_lock(target.with_suffix('.lock'), timeout=45):
        if target.exists(): return target.name, start
        import imageio_ffmpeg
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(key + '.tmp.mp4')
        args = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', '960x540', '-r', '12.5', '-i', '-', '-an',
                '-c:v', 'libx264', '-threads', '2', '-preset', 'veryfast', '-crf', '24',
                '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)]
        stream = Stream(piece['cam'], meta['day'])
        stream.seek(datetime.fromisoformat(meta['start']) + timedelta(seconds=start))
        proc = subprocess.Popen(args, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        failure = None
        try:
            for k in range(int((end-start)*FPS)):
                _, frame = stream.read()
                if frame is None: break
                if k % 2: continue
                t = start + k/FPS
                scale = np.array([960/frame.shape[1], 540/frame.shape[0]])
                img = cv2.resize(frame, (960, 540))
                nearest = int(np.argmin(abs(rows[:, 0]-t)))
                if abs(rows[nearest, 0]-t) <= .15:
                    box = np.rint(rows[nearest, 1:5] * np.tile(scale, 2)).astype(int)
                    cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (70, 230, 100), 2)
                    poly = polygons.get(nearest)
                    if poly is not None and len(poly) >= 3:
                        cv2.polylines(img, [np.rint(poly * scale).astype(np.int32)], True, (70, 230, 100), 1)
                cv2.putText(img, '%s  %.2fs  piece %d' % (piece['cam'], t, piece['piece']),
                            (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 2)
                proc.stdin.write(img.tobytes())
        except Exception as exc:
            failure = exc
        finally:
            if stream.cap is not None: stream.cap.release()
            proc.stdin.close()
            error = proc.stderr.read().decode(errors='replace')
            code = proc.wait(timeout=20)
        if failure or code:
            if tmp.exists(): tmp.unlink()
            raise RuntimeError('Video preview failed: %s' % (failure or error[-300:]))
        tmp.replace(target)
        return target.name, start
