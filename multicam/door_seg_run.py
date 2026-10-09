"""Offline door_seg + tracker -> the existing door evaluation format.

venv_sam3/python door_seg_run.py CKPT NAME DAY [DAY ...] [--tags tag,tag]
Writes only data/micro_door/NAME, data/door_v2/DAY_NAME.jsonl.gz.
Uses each 0.08-s tick, own previous union mask, no teacher identities or masks.
Clamps temporal inputs at video edges, so the first/last five ticks are retained.
No live counter, CRM, training, background scheduling or model changes here.
"""
import argparse
import json
import re
import time
from collections import deque
from pathlib import Path

import numpy as np

from door_seg_track import Config, Tracker


def contexts(cap, past=5, future=5):
    """Bounded sequential decoder, yields all frames with replicated edge context."""
    frames = deque()
    base, n, tick, eof = 0, 0, 0, False
    while True:
        while not eof and n <= tick + future:
            ok, frame = cap.read()
            if not ok:
                eof = True
                break
            frames.append(frame)
            n += 1
        if tick >= n:
            return
        yield tick, {o: frames[max(0, min(n - 1, tick + o)) - base] for o in range(-past, future + 1)}
        tick += 1
        while base < max(0, tick - past):
            frames.popleft()
            base += 1


def save_chunks(out, observations, width, height):
    """Exact sam31_reid.Masks layout, in the original video's pixel coordinates."""
    import cv2
    rows, buf, offs = [], [], [0]
    for tick, pid, p in observations:
        mask = cv2.resize(p[0][:180].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.nonzero(mask)
        if not len(xs):
            continue
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bits = np.packbits(mask[y1:y2, x1:x2].reshape(-1))
        rows.append([0, tick, pid, p[2], x1, y1, x2, y2])
        buf.append(bits)
        offs.append(offs[-1] + len(bits))
    np.savez_compressed(out / 'chunks.tmp.npz', rows=np.asarray(rows, dtype=np.float64).reshape(-1, 8),
                        buf=np.concatenate(buf) if buf else np.zeros(0, np.uint8), offs=np.asarray(offs, dtype=np.int64))
    (out / 'chunks.tmp.npz').replace(out / 'chunks.npz')
    return sorted({int(row[2]) for row in rows})


def run(ckpt, name, days, tags=None):
    import cv2
    import torch
    import door_seg as S
    import door_sam as DS
    if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in ('sam31', 'micro1s3', 'micro1s3c240p'):
        raise ValueError('use a new experiment name, e.g. doorseg_track_b1')
    base = S.ROOT / 'data' / 'micro_door' / name
    if base.exists():
        raise FileExistsError('experiment already exists; choose a new name')
    # Inference is an explicit offline invocation; never run concurrently with
    # training just to measure speed. No implicit model downloads for weights.
    net = S.DoorSeg('cuda')
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    net.load(state['model'])
    net.yolo.eval()
    net.off.eval()
    base.mkdir(parents=True)
    config = Config()
    for day in days:
        for src in sorted((S.ROOT / 'data' / 'sam31_door').glob('door_%s_*/cam1' % day)):
            tag = src.parent.name
            if tags and tag not in tags.split(','):
                continue
            video = src / 'video.mp4'
            if not video.exists():
                continue
            out = base / tag / 'cam1'
            out.mkdir(parents=True)
            cap = cv2.VideoCapture(str(video))
            width, height = int(cap.get(3)), int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if width <= 0 or height <= 0 or abs(fps - 12.5) > 0.05:
                cap.release()
                raise ValueError('%s: expected a readable 12.5-fps door video, got %s' % (video, fps))
            tracker = Tracker(config)
            prev = np.zeros((S.H, S.W), np.uint8)
            bg = S.background(day, 'cam1')
            t0, count = time.perf_counter(), 0
            def observations():
                nonlocal count, prev
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    for tick, frames in contexts(cap):
                        frames = {o: cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA) for o, f in frames.items()}
                        image = torch.from_numpy(S.channels(frames, bg, prev))[None].cuda().float() / 255
                        pred = S.predict(net, image)
                        prev.fill(0)
                        for pid, p in tracker.update(tick, pred):
                            prev |= cv2.resize(p[0][:S.H // 4].astype(np.uint8), (S.W, S.H), interpolation=cv2.INTER_NEAREST)
                            # Pack this mask now; don't retain full-frame masks
                            # for every observation of a long door stretch.
                            yield tick, pid, p
                        count += 1
            try:
                pids = save_chunks(out, observations(), width, height)
            finally:
                cap.release()
            torch.cuda.synchronize()
            sec = time.perf_counter() - t0
            info = dict(model=name, ckpt=str(Path(ckpt).resolve()), checkpoint_step=state.get('step'), stride=1,
                        ticks=count, sessions=[[0, count, 0]], tracker=config.__dict__, width=width, height=height,
                        s_per_live_tick=sec / max(1, count), own_previous_mask=True)
            (out / 'info.json').write_text(json.dumps(info, indent=1), encoding='utf-8')
            (out / 'ticks.json').write_bytes((src / 'ticks.json').read_bytes())
            # link_seams assigns zero-based piece IDs to sorted local IDs even
            # for our single session; report keys are PIECES, not local IDs.
            report = dict(person_of_piece={str(piece): pid for piece, pid in enumerate(pids)}, stats=tracker.stats)
            (out / 'report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
            print(tag, count, 'ticks', round(sec / max(1, count) * 1000, 1), 'ms/tick', tracker.stats, flush=True)
        DS.convert([day], door=base, name=name, tick=0.08)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ckpt')
    parser.add_argument('name')
    parser.add_argument('days', nargs='+')
    parser.add_argument('--tags')
    args = parser.parse_args()
    run(args.ckpt, args.name, args.days, args.tags)
