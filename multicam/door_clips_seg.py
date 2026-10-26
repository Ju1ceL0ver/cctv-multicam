"""14.10 owner: a short video of every live door event -- the owner labels who is who on it, and visit videos
are later cut from the same tape.

Tape: a thread keeps the last TAPE_S seconds of processed ticks of camera 1 at 1280 x 720 (JPEG in memory, ~40 MB)
with every person's mask (1/4 of 1280 x 720, packed bits) and number. The count loop only hands it references.
An event asks for [t - BEFORE, t + AFTER]; once the tape has reached t + AFTER the clip is drawn (the event's person
bright and numbered, the others dim) and encoded to H.264 by ffmpeg at low priority
-> data/live/clips_seg/<day>/<key>.mp4 (~0.5-1 MB)."""
import collections, os, queue, subprocess, threading, time
from pathlib import Path

import numpy as np
import cv2

TAPE_S, BEFORE, AFTER = 30.0, 10.0, 8.0                # 15.10 owner: longer clips (crowds need the whole passage)
W, H = 1280, 720                                       # 21.10 owner: 640x360 was too blurry to label
FPS = 12.5
CROWD_S = 3.0                                          # people counted within +-3 s of the crossing
FFMPEG = os.environ.get('RA_FFMPEG') or r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\cctv_base\Lib\site-packages\imageio_ffmpeg\binaries\ffmpeg-win-x86_64-v7.1.exe'


class Tape(threading.Thread):
    def __init__(self, folder, log=print):
        super().__init__(daemon=True)
        self.dir = Path(folder)
        self.log = log
        self.inbox = queue.Queue(maxsize=400)        # (t, frame 1280x720, [(pid, mask 180x320)])
        self.tape = collections.deque()              # (t, jpeg bytes, [(pid, packed, shape)])
        self.want = []                               # (event, key)
        self.lock = threading.Lock()

    def put(self, t, frame, out):
        try:
            self.inbox.put_nowait((t, frame, [(str(p), m) for p, m in out]))
        except queue.Full:                           # the count never waits for the tape
            pass

    def ask(self, ev, key):
        with self.lock:
            self.want.append((ev, key))

    def run(self):
        while True:
            try:
                t, fr, out = self.inbox.get(timeout=1.0)
                jpg = cv2.imencode('.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()   # 15.10: full 1280x720 for crowd frames
                masks = [(p, np.packbits(m.astype(bool)), m.shape) for p, m in out]
                self.tape.append((t, jpg, masks))
                while self.tape and self.tape[0][0] < t - TAPE_S:
                    self.tape.popleft()
            except queue.Empty:
                pass
            except Exception as exc:
                self.log('tape failed: %s' % str(exc)[:200])
            if not self.tape:
                continue
            now_t = self.tape[-1][0]
            with self.lock:
                ready = [w for w in self.want if now_t >= w[0]['t'] + AFTER]
                self.want = [w for w in self.want if w not in ready]
            for ev, key in ready:
                try:
                    self.write(ev, key)
                except Exception as exc:
                    self.log('clip failed: %s' % str(exc)[:200])

    def write(self, ev, key):
        t0, t1 = ev['t'] - BEFORE, ev['t'] + AFTER
        part = [x for x in list(self.tape) if t0 <= x[0] <= t1]
        if len(part) < 5:
            self.log('clip %s: only %d frames on the tape' % (key, len(part)))
            return
        day = time.strftime('%Y%m%d', time.localtime(ev['t']))
        out = self.dir / day
        out.mkdir(parents=True, exist_ok=True)
        path = out / (key + '.mp4')
        tmp = out / (key + '.tmp.mp4')
        p = subprocess.Popen([FFMPEG, '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % (W, H),
                              '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '26',
                              '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)],
                             stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                             creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0) | 0x08000000)
        me = str(ev['w'])
        col = (60, 220, 60) if ev['kind'] == 'in' else (30, 150, 255)
        title = '%s  %s  #%s' % ('ВХОД' if ev['kind'] == 'in' else 'ВЫХОД', ev['clock'][11:], me)
        for t, jpg, masks in part:
            img = cv2.resize(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR), (W, H), interpolation=cv2.INTER_AREA)
            for pid, packed, shape in masks:
                m = np.unpackbits(packed)[:shape[0] * shape[1]].reshape(shape)
                big = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
                cn, _ = cv2.findContours(big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if pid == me:
                    img[big > 0] = (0.85 * img[big > 0] + 0.15 * np.array(col)).astype(np.uint8)   # light tint: the face stays
                    cv2.drawContours(img, cn, -1, col, 3)
                    ys, xs = np.nonzero(big)
                    cv2.putText(img, '#' + pid, (int(xs.min()), max(14, int(ys.min()) - 4)), cv2.FONT_HERSHEY_SIMPLEX, .6, col, 2)
                else:
                    cv2.drawContours(img, cn, -1, (150, 150, 150), 1)
            mark = abs(t - ev['t']) < 0.6                # the moment of the crossing
            cv2.rectangle(img, (0, 0), (W, 34), (0, 0, 0), -1)
            cv2.putText(img, '%+.1f s' % (t - ev['t']), (W - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
            put_text(img, title, (8, 3), col if mark else (230, 230, 230))
            p.stdin.write(img.tobytes())
        p.stdin.close()
        err = p.stderr.read().decode('utf-8', 'replace')
        if p.wait(timeout=60) != 0 or not tmp.exists():
            self.log('clip %s: ffmpeg %s' % (key, err[-200:]))
            return
        os.replace(tmp, path)
        crowd = max((len(m) for t, _, m in part if abs(t - ev['t']) <= CROWD_S), default=0)
        if crowd >= 2:
            self.keep_crowd(ev, key, part, day, crowd)

    def keep_crowd(self, ev, key, part, day, crowd):
        """15.10 owner: two or more people at the door -> the raw frames (1280x720) and the model's masks of the whole
        passage, for his click labels and for training -> data/live/crowd/<day>/<key>/{NNNN.jpg, masks.npz, meta.json}"""
        import json
        out = self.dir.parent / 'crowd' / day / key
        out.mkdir(parents=True, exist_ok=True)
        meta = {'event': {k: ev[k] for k in ('kind', 't', 'w', 'clock') if k in ev}, 'crowd': crowd, 'frames': []}
        arrays = {}
        for i, (t, jpg, masks) in enumerate(part):
            (out / ('%04d.jpg' % i)).write_bytes(jpg)
            meta['frames'].append({'i': i, 't': round(t, 3), 'pids': [p for p, _, _ in masks]})
            for j, (p, packed, shape) in enumerate(masks):
                arrays['f%04d_%d' % (i, j)] = packed
                meta.setdefault('shape', list(shape))
        np.savez_compressed(out / 'masks.npz', **arrays)
        (out / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')


_FONT = None


def put_text(img, text, xy, col):
    """Cyrillic needs a TrueType font; cv2.putText cannot draw it."""
    global _FONT
    from PIL import Image, ImageDraw, ImageFont
    if _FONT is None:
        for f in ('arialbd.ttf', 'arial.ttf'):
            try:
                _FONT = ImageFont.truetype(f, 26); break
            except OSError:
                pass
        _FONT = _FONT or ImageFont.load_default()
    x, y = xy
    w = int(_FONT.getlength(text)) + 4
    roi = img[y:y + 32, x:x + w]
    im = Image.fromarray(roi[:, :, ::-1].copy()); ImageDraw.Draw(im).text((2, 0), text, fill=tuple(int(c) for c in col[::-1]), font=_FONT)
    img[y:y + 32, x:x + w] = np.asarray(im)[:, :, ::-1]
