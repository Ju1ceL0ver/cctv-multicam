"""The door stretches through SAM 3.1 -- the best tracker there is -- to see what the door rule gets from tracks that
do not swap people in groups (03.10.2026). The same job server as the 15-minute windows (sam31_jobs.py), camera 1
only, and its own folder data/sam31_door (so that the exam day 18.09 never reaches the segmentation training through
data/sam31_seg).

  prep DAY ...      every counter-event stretch of the days (door_v2.stretches, +-25 s) -> jobs for the workers
  hub               merges each stretch when all its jobs are back (seams + ReID, sam31_reid.py)
  convert DAY ...   merged stretches -> data/door_v2/<day>_sam31.jsonl.gz in door_v2's run format (per tick: the
                    person, box, feet = the bottom of the mask's main blob, SAM's score), for door_learn.py & co

usage: door_sam.py prep 20260919 20260917 20260918 | hub | convert 20260919 ..."""
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DOOR = ROOT / 'data' / 'sam31_door'
CAM = 'cam1'


def tag_of(day, a):
    return 'door_%s_%05d' % (day, int(round(a)))


def low_priority(threads=8):
    """By day the machine is the live counter's: below-normal priority, a few threads, no card."""
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == 'nt' else 10)
    except Exception:
        pass
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass


def prep(days):
    """Every tick of a stretch goes to SAM (people are at the door nearly all the time there, so no student pass to
    find the live ticks); frames are read by their time stamps (door_v2.Raw), video.mp4 for the ReID crops is
    encoded fast."""
    low_priority()
    import cv2
    import day_proxy
    import door_v2 as D
    import sam31_jobs as J
    import sam31_segment as SG
    J.SEG = DOOR
    for day in days:
        raw = D.Raw(day, CAM)
        for a, b in D.stretches(day):
            tag = tag_of(day, a)
            out = DOOR / tag / CAM
            if (out / 'ticks.json').exists():
                continue
            shutil.rmtree(out, ignore_errors=True)
            out.mkdir(parents=True, exist_ok=True)
            (out / 'preparing').write_text(time.strftime('%H:%M'))
            n = int(round((b - a) / SG.TICK))
            ticks = SG.tick_frames(day, CAM, a, n)
            (out / 'sam_in').mkdir()
            enc = subprocess.Popen([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                                    '-s', '%dx%d' % (SG.W, SG.H), '-r', str(1 / SG.TICK), '-i', '-', '-c:v', 'libx264',
                                    '-preset', 'ultrafast', '-crf', '20', '-g', '12', '-bf', '0', '-pix_fmt', 'yuv420p',
                                    str(out / 'video.mp4')], stdin=subprocess.PIPE)
            black = np.zeros((SG.H, SG.W, 3), np.uint8)
            for i, tk in enumerate(ticks):
                img = raw.get(tk)
                g = black if img is None else cv2.resize(img, (SG.W, SG.H), interpolation=cv2.INTER_LINEAR)
                enc.stdin.write(g.tobytes())
                cv2.imwrite(str(out / 'sam_in' / ('%05d.jpg' % i)), cv2.resize(g, (SG.SAM_IN, SG.SAM_IN), interpolation=cv2.INTER_AREA),
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
            enc.stdin.close(); enc.wait()
            live = np.ones(n, bool)
            json.dump({'ticks': ticks, 'people': [], 'live': live.astype(int).tolist()}, open(out / 'ticks.json', 'w'))
            (J.JOBS / 'inputs').mkdir(parents=True, exist_ok=True)
            for s, e, shared in J.sessions(live):
                jid = '%s_%s_%05d' % (tag, CAM, s)
                mp4 = J.JOBS / 'inputs' / (jid + '.mp4')
                subprocess.run([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-framerate', '12.5', '-start_number', str(s),
                                '-i', str(out / 'sam_in' / '%05d.jpg'), '-frames:v', str(e - s), '-c:v', 'libx264', '-preset', 'fast',
                                '-crf', '14', '-pix_fmt', 'yuv420p', str(mp4)], check=True)
                J.write(J.JOBS / 'jobs' / (jid + '.json'), {'id': jid, 'tag': tag, 'day': day, 'film_start': a, 'seconds': b - a,
                                                             'cam': CAM, 'start': s, 'stop': e, 'shared': shared, 'status': 'open',
                                                             'made': time.time(), 'door': True})
            shutil.rmtree(out / 'sam_in', ignore_errors=True)
            (out / 'preparing').unlink()
            print(time.strftime('%H:%M'), 'prepared', tag, n, 'ticks, mismatched stamps so far', raw.off, flush=True)


def hub():
    import sam31_jobs as J
    J.SEG = DOOR
    while True:
        for tag, cam in J.finished_cameras():
            if not tag.startswith('door_'):
                continue
            try:
                J.merge(tag, cam)
            except Exception as e:
                print(time.strftime('%H:%M'), 'merge failed', tag, cam, repr(e)[:300], flush=True)
        import bg
        if not any(p.name.startswith('door_') for p in (J.JOBS / 'jobs').glob('*.json')) and not bg.running('door_sam.py prep'):
            print(time.strftime('%H:%M'), 'no door jobs left', flush=True)
            return
        time.sleep(60)


def foot_of(m):
    """The bottom of the mask's main blob: x mean of its lowest rows, y its lowest row (crop coordinates)."""
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == k)
    low = ys >= ys.max() - 2
    return float(xs[low].mean()), float(ys.max())


def static_people(M, owned, person, n_ticks, max_px=8.0, min_share=0.7):
    """Numbers of the "people" that never move (the poster stand by the door): the median distance of the box centre from
    its median below max_px (2176 frame) -- robust to SAM lending the number to a passer-by for a moment -- while seen in
    more than min_share of the stretch's ticks."""
    by = {}
    for p, rs in owned.items():
        pid = person.get(int(p)) or 0
        for r in rs:
            x1, y1, x2, y2 = M.rows[r, 4:8]
            by.setdefault(pid, {})[int(M.rows[r, 1])] = ((x1 + x2) / 2, (y1 + y2) / 2)
    out = set()
    for pid, c in by.items():
        P = np.array(list(c.values()))
        if len(c) >= min_share * n_ticks and float(np.median(np.linalg.norm(P - np.median(P, 0), axis=1))) < max_px:
            out.add(pid)
    return out


def convert(days, door=None, name='sam31', tick=None):
    """door: the folder of merged stretches (the teacher's data/sam31_door by default; door_micro.py passes the small
    model's); name: the run's name in data/door_v2/<day>_<name>.jsonl.gz."""
    import door_v2 as D
    import sam31_reid as R
    door = Path(door) if door else DOOR
    tick = tick or D.TICK                                  # door_micro.py --stride keeps every n-th tick
    summary = []
    for day in days:
        spans = D.stretches(day)
        rows_out = {}
        done = []
        for si, (a, b) in enumerate(spans):
            base = door / tag_of(day, a) / CAM
            if not (base / 'report.json').exists():
                continue
            info = json.load(open(base / 'info.json'))
            rep = json.load(open(base / 'report.json'))
            M = R.Masks(base / 'chunks.npz')
            overlap = {int(s): int(sh) for s, e, sh in info['sessions']}
            owned, _ = R.link_seams(M, overlap)
            person = {int(p): v for p, v in rep['person_of_piece'].items()}
            static = static_people(M, owned, person, info['ticks'] if 'stride' not in info else info['ticks'])
            for p, rs in owned.items():
                if (person.get(int(p)) or 0) in static:
                    continue                                   # the poster stand: never moves, not a person
                w = si * 10000 + int(person.get(int(p)) or 0)
                for r in rs:
                    tick_i = int(M.rows[r, 1])
                    x1, y1, x2, y2 = M.rows[r, 4:8].astype(int)
                    f = foot_of(M.crop(r))
                    t = round(a + tick_i * tick, 2)
                    q = {'w': w, 's': round(float(M.rows[r, 3]), 3),
                         'box': [round((x1 + x2) / 2 / 2176, 4), round((y1 + y2) / 2 / 1248, 4), round((x2 - x1) / 2176, 4), round((y2 - y1) / 1248, 4)],
                         'foot': [x1 + f[0], y1 + f[1]] if f else None, 'new': False, 'piece': int(p)}
                    rows_out.setdefault((si, t), []).append(q)
            done.append(si)
        path = ROOT / 'data' / 'door_v2' / ('%s_%s.jsonl.gz' % (day, name))
        start, _ = __import__('day_movie').clock(day, str(ROOT))
        with gzip.open(path, 'wt') as fo:
            fo.write(json.dumps({'day': day, 'cam': CAM, 'ckpt': 'sam3.1' if name == 'sam31' else name, 'film_start': start, 'pad': 25.0, 'spans': spans,
                                 'tick': tick, 'sam_done': done}) + '\n')
            for (si, t) in sorted(rows_out):
                fo.write(json.dumps({'s': si, 't': t, 'p': rows_out[(si, t)]}) + '\n')
        print(day, 'stretches', len(done), 'of', len(spans), 'ticks', len(rows_out), '->', path, flush=True)
        summary.append('%s: %d of %d stretches, %d ticks' % (day, len(done), len(spans), len(rows_out)))
    return '; '.join(summary)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'prep':
        prep(sys.argv[2:])
    elif cmd == 'hub':
        hub()
    elif cmd == 'convert':
        convert(sys.argv[2:])
