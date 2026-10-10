"""Bounded TRAIN-only RTX3060 profile. No training or production writes.

venv_sam3/python door_seg_night_probe.py --check
venv_sam3/python door_seg_night_probe.py --out data/quality_backups/doorseg_night_20261010
The --check path imports no torch and opens no GPU context, even at night.
"""
import argparse
import datetime
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TAGS = ('door_20260918_01405', 'door_20260918_15375')


def night(now):
    return now.strftime('%H:%M') >= '21:00' or now.strftime('%H:%M') < '09:35'


def cohort(root=ROOT):
    state = json.loads((root / 'data/door_side/20260918.json').read_text(encoding='utf-8'))
    for tag in TAGS:
        if not state.get(tag, {}).get('done') or state[tag].get('role') != 'train':
            raise ValueError('requires owner-completed TRAIN: ' + tag)
        if not (root / 'data/sam31_door' / tag / 'cam1/video.mp4').is_file():
            raise FileNotFoundError(tag)
    return list(TAGS)


def blockers(now=None, processes=None, parent_pids=()):
    """Fail closed on unreadable process state; never stop foreign processes."""
    if not night(now or datetime.datetime.now()):
        return ['daytime']
    import psutil
    found = []
    try:
        # Windows venv's redirector is another python.exe with our cmdline.
        # Ignore our ancestors, never an unrelated probe or trainer.
        own = {os.getpid(), *parent_pids}
        if processes is None:
            own.update(p.pid for p in psutil.Process().parents())
        for p in processes if processes is not None else psutil.process_iter(['pid', 'name', 'cmdline']):
            row = p if isinstance(p, dict) else p.info
            if row['pid'] in own:
                continue
            name = (row.get('name') or '').lower()
            if any(app in name for app in ('photoshop', '3dsmax', '3ds max')):
                found.append('foreign GPU app PID %s' % row['pid'])
            cmd = ' '.join(row.get('cmdline') or []).lower()
            if 'python' in name and any(key in cmd for key in
                    ('door_live.py', 'door_seg.py train', 'door_seg_ddp.py',
                     'door_seg_supervise.py', 'door_seg_night_probe.py', 'auto_label.py',
                     'sam31_segment.py', 'door_night.py', 'train_v2.py', 'tonight.py')):
                found.append('GPU job PID %s' % row['pid'])
    except (psutil.Error, OSError) as exc:
        found.append('process check failed: ' + type(exc).__name__)
    return found


def write_report(out, report):
    temp = out / 'report.tmp.json'
    temp.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temp.replace(out / 'report.json')


def compare_predictions(a, b):
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    if len(a) != len(b):
        return dict(count_equal=False, masks_equal=False, min_iou=0., max_offset_error=None)
    if not a:
        return dict(count_equal=True, masks_equal=True, min_iou=1., max_offset_error=0.)
    iou = np.array([[np.logical_and(x[0], y[0]).sum() / max(1, np.logical_or(x[0], y[0]).sum())
                     for y in b] for x in a])
    ii, jj = linear_sum_assignment(-iou)
    return dict(count_equal=True, masks_equal=all(np.array_equal(a[i][0], b[j][0]) for i, j in zip(ii, jj)),
                min_iou=float(iou[ii, jj].min()), max_offset_error=float(max(
                    np.abs(np.concatenate(a[i][3:]) - np.concatenate(b[j][3:])).max() for i, j in zip(ii, jj))))


def run(args):
    tags = cohort()
    waiting = blockers()
    ckpt = ROOT / args.ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(ckpt)
    if args.wait and not args.check:
        end = datetime.datetime.now().replace(hour=9, minute=35, second=0, microsecond=0)
        if end <= datetime.datetime.now():
            end += datetime.timedelta(days=1)
        print('CPU-only waiter; free GPU window 21:00-09:35, end', end.isoformat(), flush=True)
        while waiting and datetime.datetime.now() < end:
            time.sleep(20)
            waiting = blockers()
        if datetime.datetime.now() >= end:
            print('night window ended; no GPU probe', flush=True)
            return 2
    if args.check or waiting:
        print(json.dumps(dict(ready=not waiting, blockers=waiting, cohort=tags, gpu_opened=False)), flush=True)
        return 0 if args.check else 2
    # The supervisor's initial utilization gate avoids unknown active jobs.
    from door_seg_supervise import gpu_busy
    if gpu_busy():
        print('GPU busy; defer this short probe', flush=True)
        return 2
    import cv2
    import numpy as np
    import torch
    import door_seg as S
    import door_seg_fast as F
    from door_seg_run import contexts, save_chunks
    from door_seg_track import Tracker
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    S.checkpoint_finite(state)
    report = dict(complete=False, checkpoint=str(ckpt), step=state.get('step'),
                  sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
                  made=datetime.datetime.now().isoformat(), cohort=tags, sources={},
                  limitations='TRAIN diagnostics, not release quality. Side/role queues excluded; no event scoring.')
    write_report(out, report)
    begin = time.monotonic()

    def guard():
        if time.monotonic() - begin > args.seconds:
            raise TimeoutError('bounded research time exhausted')
        blocked = blockers()
        if blocked:
            raise RuntimeError('yielding GPU: ' + ', '.join(blocked))

    # Independent models: fusion must not alter the unfused reference's hooks.
    guard()
    reference, fast = S.DoorSeg('cuda'), S.DoorSeg('cuda')
    for net in (reference, fast):
        net.load(state['model']); net.yolo.eval(); net.off.eval()
    fast.yolo.fuse(verbose=False)
    try:
        for tag in tags:
            guard(); cohort()  # Reject a TRAIN->TEST role change before inference.
            cap = cv2.VideoCapture(str(ROOT / 'data/sam31_door' / tag / 'cam1/video.mp4'))
            width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
            if width <= 0 or height <= 0 or abs(fps - 12.5) > .05:
                cap.release(); raise ValueError('unreadable/non-12.5fps TRAIN source: ' + tag)
            bg = S.background('20260918', 'cam1')
            assemble = F.GPUChannels(bg)
            prev = np.zeros((S.H, S.W), np.uint8)
            tracker = Tracker()
            equiv, timing, observations = [], [], []
            ctx = iter(contexts(cap, transform=lambda f: cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA)))
            try:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    # Same input/previous mask: isolate channel+fusion+postprocess errors.
                    for _ in range(args.equivalence):
                        guard()
                        tick, frames = next(ctx)
                        exact = S.channels(frames, bg, prev)
                        img = assemble(frames, prev)
                        channel_equal = np.array_equal((img[0] * 255).round().byte().cpu().numpy(), exact)
                        p = S.predict(reference, torch.from_numpy(exact)[None].cuda().float() / 255)
                        q = F.predict(fast, img)
                        equiv.append(dict(tick=tick, channels_equal=channel_equal, **compare_predictions(p, q)))
                        prev = F.previous_union(p)
                    # Separate full-throughput measurement, own previous masks, fresh tracker.
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ctx = iter(contexts(cap, transform=lambda f: cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA)))
                    prev.fill(0)
                    torch.cuda.synchronize(); start = time.perf_counter()
                    guard_seconds = 0.
                    for i in range(args.ticks):
                        if i % 16 == 0:
                            g0 = time.perf_counter(); guard(); guard_seconds += time.perf_counter() - g0
                        t0 = time.perf_counter(); tick, frames = next(ctx); t1 = time.perf_counter()
                        img = assemble(frames, prev); torch.cuda.synchronize(); t2 = time.perf_counter()
                        pred = F.predict(fast, img); torch.cuda.synchronize(); t3 = time.perf_counter()
                        if any(not np.isfinite(np.concatenate([p[1], [p[2]], p[3], p[4]])).all() for p in pred):
                            raise FloatingPointError('nonfinite prediction')
                        observations.extend((tick, pid, p) for pid, p in tracker.update(tick, pred))
                        prev = F.previous_union(pred); t4 = time.perf_counter()
                        timing.append([t1-t0, t2-t1, t3-t2, t4-t3])
                    source_out = out / tag
                    source_out.mkdir()
                    t0 = time.perf_counter(); save_chunks(source_out, observations, width, height)
                    export_s = time.perf_counter() - t0
                    elapsed = time.perf_counter() - start - guard_seconds
            finally:
                cap.release()
            report['sources'][tag] = dict(ticks=len(timing), equivalence=equiv, stats=tracker.stats,
                detections=len(observations), complete_processing_ms=1000*elapsed/len(timing), export_seconds=export_s,
                excluded_guard_seconds=guard_seconds,
                stages=['decode_resize', 'channels', 'network_postprocess', 'tracker_previous'],
                mean_stage_ms=(np.mean(timing[5:], axis=0)*1000).tolist(),
                p90_stage_ms=(np.quantile(timing[5:], .9, axis=0)*1000).tolist())
            write_report(out, report)
            print(tag, report['sources'][tag]['complete_processing_ms'], 'ms/tick', flush=True)
        report['complete'] = True
    except Exception as exc:
        report['error'] = type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        write_report(out, report)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ckpt', default='runs/doorseg_b/step8604.pt')
    parser.add_argument('--out', default='data/quality_backups/doorseg_night_20261010')
    parser.add_argument('--ticks', type=int, default=120)
    parser.add_argument('--equivalence', type=int, default=24)
    parser.add_argument('--seconds', type=int, default=180)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--wait', action='store_true', help='CPU-only wait for the next free night window')
    a = parser.parse_args()
    if not 10 <= a.ticks <= 200 or not 1 <= a.equivalence <= 32 or not 30 <= a.seconds <= 300:
        parser.error('short probe only: ticks10..200, equivalence1..32, seconds30..300')
    raise SystemExit(run(a))
