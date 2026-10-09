"""A short GIF of one door crossing for /liveevents (10.10.2026, the owner: the clip just rolls, show who it is about).

30 ticks of the live window around the crossing (15 before, 15 after; a tick is 0.24 s), 600 px wide, the person of the
event filled and outlined with SAM's own masks (the event's tracks), the others left as they are, a caption on top
("ВХОД 18:57:26 покупатель") and a red frame on the tick of the crossing.

make(win_dir, times, e, out) -- from a live window (data/live/cam1/<window>/cam1: frames/, chunks.npz, info.json);
from_clip(clip, offset, e, out) -- without masks, from the event's clip when the window is gone.
usage: door_gif.py DAY        GIFs for a past day's events (windows still on disk; clips otherwise)"""
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIVE = Path(os.environ.get('RA_LIVE_OUT') or ROOT / 'data' / 'live')
N, WIDTH, MS = 15, 600, 160          # ticks each side, px, ms per GIF frame (0.24 s of the shop in 0.16 s)
SAM_W = 2176.0                       # the masks' frame (door_live W)


def caption(e):
    kind = 'ВХОД' if e['kind'] == 'in' else 'ВЫХОД'
    role = 'сотрудник' if e.get('role') == 'staff' else 'покупатель' if e.get('role') == 'customer' else ''
    return '%s  %s  %s' % (kind, str(e.get('clock', ''))[11:19], role)


def _text(img, text, col):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    font = None
    for f in ('arialbd.ttf', 'arial.ttf', 'DejaVuSans-Bold.ttf'):
        try:
            font = ImageFont.truetype(f, 26)
            break
        except OSError:
            continue
    d.rectangle([0, 0, im.width, 40], fill=(0, 0, 0))
    d.text((10, 6), text, fill=col, font=font)
    return np.asarray(im)


def _save(frames, out):
    from PIL import Image
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pal = Image.fromarray(frames[len(frames) // 2]).quantize(colors=64, method=Image.MEDIANCUT)   # one palette for all:
    ims = [Image.fromarray(f).quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]     # 3 MB -> well under
    tmp = out.with_suffix('.tmp.gif')
    ims[0].save(tmp, save_all=True, append_images=ims[1:], duration=MS, loop=0, optimize=True)
    os.replace(tmp, out)
    return str(out)


def make(win_dir, times, e, out):
    """win_dir: .../cam1/<window>/cam1; times: the clock time of every tick of the window"""
    import sam31_reid as R
    win_dir = Path(win_dir)
    win = win_dir.parent.name
    info = json.load(open(win_dir / 'info.json'))
    M = R.Masks(win_dir / 'chunks.npz')
    owned, _ = R.link_seams(M, {int(s): int(sh) for s, e_, sh in info['sessions']})
    pieces = {int(str(x).split(':')[1]) - 1000 for x in e.get('tracks') or [] if str(x).startswith(win + ':')}
    by_tick = {}
    for p in pieces:
        for r in owned.get(p, []):
            by_tick.setdefault(int(M.rows[r, 1]), []).append(r)
    times = np.asarray(times, float)
    k0 = int(np.argmin(np.abs(times - float(e['t']))))
    sc = 1280.0 / float(info.get('size', [SAM_W])[0])      # live windows keep their masks at 1280 x 720 (small_masks)
    col = (60, 220, 60) if e['kind'] == 'in' else (255, 150, 30)
    frames = []
    for k in range(k0 - N, k0 + N + 1):
        f = cv2.imread(str(win_dir / 'frames' / ('%05d.jpg' % k)))
        if f is None:
            continue
        f = cv2.resize(f, (1280, 720))[:, :, ::-1].copy()
        m = np.zeros((720, 1280), np.uint8)
        for r in by_tick.get(k, []):
            x1, y1 = int(M.rows[r, 4] * sc), int(M.rows[r, 5] * sc)
            c = M.crop(r).astype(np.uint8)
            cw, ch = max(1, int(round(c.shape[1] * sc))), max(1, int(round(c.shape[0] * sc)))
            cs = cv2.resize(c, (cw, ch), interpolation=cv2.INTER_NEAREST)[:720 - y1, :1280 - x1]
            m[y1:y1 + cs.shape[0], x1:x1 + cs.shape[1]] |= cs
        if m.any():
            f[m > 0] = (0.55 * f[m > 0] + 0.45 * np.array(col)).astype(np.uint8)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(f, cnts, -1, (255, 230, 0), 4)
        if k == k0:
            cv2.rectangle(f, (0, 0), (1279, 719), (230, 30, 30), 12)
        f = cv2.resize(f, (WIDTH, int(720 * WIDTH / 1280)), interpolation=cv2.INTER_AREA)
        frames.append(_text(f, caption(e) + ('   ◄ проход' if k == k0 else ''), col))
    return _save(frames, out) if frames else ''


def from_clip(clip, offset, e, out, fps=12.5):
    """no masks: the event's clip (12.5 a second) around the crossing, every third frame, captioned"""
    cap = cv2.VideoCapture(str(clip))
    col = (60, 220, 60) if e['kind'] == 'in' else (255, 150, 30)
    k0 = int(round(offset * fps))
    want = set(range(k0 - 3 * N, k0 + 3 * N + 1, 3))
    frames = []
    k = -1
    while k < k0 + 3 * N:                               # read in order: seeking these clips lands on one frame
        ok, f = cap.read()
        if not ok:
            break
        k += 1
        if k not in want:
            continue
        f = cv2.resize(f, (1280, 720))[:, :, ::-1].copy()
        if abs(k - k0) < 2:
            cv2.rectangle(f, (0, 0), (1279, 719), (230, 30, 30), 12)
        f = cv2.resize(f, (WIDTH, int(720 * WIDTH / 1280)), interpolation=cv2.INTER_AREA)
        frames.append(_text(f, caption(e) + '   (без маски)' + ('   ◄ проход' if abs(k - k0) < 2 else ''), col))
    cap.release()
    return _save(frames, out) if frames else ''


def backfill(day):
    """GIFs for a day's events: from the windows still on disk (tick times from records/<day>.jsonl), else clips"""
    import live_events as LE
    rec = {}
    p = LIVE / 'records' / ('%s.jsonl' % day)
    if p.exists():
        for line in open(p, encoding='utf-8'):
            try:
                o = json.loads(line)
            except Exception:                              # a torn line (the writer was killed mid-line)
                continue
            if not isinstance(o, dict) or 'win' not in o or 'k' not in o:
                continue
            rec.setdefault(o['win'], {})[int(o['k'])] = float(o['t'])
    cs = LE.clips(day)
    done = 0
    for e in LE.events(day):
        out = LIVE / 'gifs' / day / (LE.key_of(e) + '.gif')
        wd = LIVE / 'cam1' / e['window'] / 'cam1'
        try:
            if wd.exists() and e['window'] in rec:
                kt = rec[e['window']]
                ks = np.array(sorted(kt))
                times = np.interp(np.arange(ks.max() + N + 2), ks, [kt[k] for k in ks])
                r = make(wd, times, e, out)
            else:
                c = next((c for c in cs if c['t0'] - 0.5 <= e['t'] <= c['t1'] + 0.5), None)
                r = from_clip(LIVE / 'clips' / day / (c['name'] + '.mp4'), e['t'] - c['t0'], e, out) if c else ''
            done += bool(r)
            print(e['clock'][11:], e['kind'], 'mask' if wd.exists() else 'clip', r or 'none', flush=True)
        except Exception as exc:
            print(e.get('clock'), 'failed', exc, flush=True)
    print('done', done)


if __name__ == '__main__':
    backfill(sys.argv[1])
