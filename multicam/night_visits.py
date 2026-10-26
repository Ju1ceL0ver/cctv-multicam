"""15.10 owner: the night pass behind the visits -- the day's raw recording of both cameras (raw_hours, 13:00-17:00)
cut into 10-minute windows, each: yolo26x-seg on both cameras (every 3rd frame) -> OSNet looks -> camera clock offset
-> people on the floor (run_clip.py). Then visits_build.py (when it is there) joins them into visits.
Waits for 21:05, stops before 09:30; a window already done is skipped, so a second night goes on.
usage: night_visits.py [DAY] [FROM-TO]   log: data/night_visits/<day>.log"""
import json, os, subprocess, sys, time
from datetime import datetime, timedelta
from pathlib import Path
ROOT = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\cctv_base\python.exe'
DAY = sys.argv[1] if __name__ == '__main__' and len(sys.argv) > 1 else time.strftime('%Y%m%d')
SPAN = sys.argv[2] if __name__ == '__main__' and len(sys.argv) > 2 else '13:01-17:00'          # the recorder starts a few seconds after 13:00
WIN = 600
OUT = ROOT / 'data' / 'night_visits'
LOG = None


def log(m):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / ('%s.log' % DAY), 'a', encoding='utf-8') as f:
        f.write('%s %s\n' % (time.strftime('%m-%d %H:%M:%S'), m))


def night(now=None):
    now = now or datetime.now()
    # The morning of DAY is before its recording, not the following night.
    recorded = datetime.strptime(DAY + SPAN.split('-')[1], '%Y%m%d%H:%M')
    h = now.strftime('%H:%M')
    return now >= recorded and (h >= '21:05' or h < '09:30')


def coverage(day, start, seconds):
    """Verify both files and frame-clock coverage before loading a GPU model.

    rawsource addresses frames at 25 fps; use that same clock, not container fps.
    Reject holes and truncated/unreadable segments instead of skipping over them.
    """
    import cv2
    import rawsource
    end = start + timedelta(seconds=seconds)
    for cam in ('cam1', 'cam2'):
        cursor = start
        for path, began in sorted(rawsource.segments(cam, day), key=lambda r: r[1]):
            if began >= end or began + timedelta(seconds=rawsource.SEG) <= cursor:
                continue
            cap = cv2.VideoCapture(path)
            try:
                count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                readable = cap.isOpened() and count > 0
            finally:
                cap.release()
            if not readable:
                continue
            until = began + timedelta(seconds=min(rawsource.SEG, count / rawsource.FPS))
            if began > cursor + timedelta(seconds=1 / rawsource.FPS):
                continue
            if until > cursor:
                cursor = until
            if cursor >= end:
                break
        if cursor < end:
            return '%s: no readable continuous video at %s' % (cam, cursor.isoformat())
    return None


def step(name, cmd, env=None):
    t0 = time.time()
    now = datetime.now()
    deadline = now.replace(hour=9, minute=30, second=0, microsecond=0)
    if now >= deadline:
        deadline += timedelta(days=1)
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace',
                           timeout=max(1, (deadline-now).total_seconds()),
                           env=dict(os.environ, PYTHONIOENCODING='utf-8', RA_VISITS_NIGHT='1', **(env or {})))
    except subprocess.TimeoutExpired as exc:
        r = subprocess.CompletedProcess(cmd, 124, str(exc.stdout or ''), 'Morning cutoff: process stopped')
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (DAY + '_' + name.replace(' ', '_') + '.log')).write_text(
        (r.stdout or '') + '\nSTDERR\n' + (r.stderr or ''), encoding='utf-8')
    log('%s rc=%d %.0f s %s' % (name, r.returncode, time.time() - t0, (r.stderr or '')[-300:].replace('\n', ' | ') if r.returncode else ''))
    return r


if __name__ == '__main__':
    while not night():
        time.sleep(30)
    a, b = SPAN.split('-')
    t, end = datetime.strptime(DAY + a, '%Y%m%d%H:%M'), datetime.strptime(DAY + b, '%Y%m%d%H:%M')
    clips = []
    while t + timedelta(seconds=WIN) <= end:
        clips.append(('v%s_%s' % (DAY[4:], t.strftime('%H%M')), t.strftime('%H:%M:%S'))); t += timedelta(seconds=WIN)
    log('start %s %s: %d windows' % (DAY, SPAN, len(clips)))
    done, skipped, failed = [], {}, {}
    for clip, hms in clips:
        if not night():
            log('morning: stop'); break
        d = ROOT / 'data' / 'raw_clips' / clip
        if (d / 'people_yolo26x-seg.json').exists():
            done.append(clip); continue
        missing = coverage(DAY, datetime.strptime(DAY + hms, '%Y%m%d%H:%M:%S'), WIN)
        if missing:
            skipped[clip] = missing
            log('skip ' + clip + ': ' + missing)
            continue
        if not (d / 'dets_yolo26x-seg.npz').exists():
            r = step('detect ' + clip, [PY, 'detect_raw.py', clip, DAY, hms, str(WIN)], {'RA_STRIDE': '3'})
            if r.returncode:
                failed[clip] = 'detect'
                continue
        if not list(d.glob('emb_*.npz')):
            r = step('embed ' + clip, [PY, 'embed_clip.py', clip, 'osnet_ain_x1_0_msmt17.pt'])
            if r.returncode:
                failed[clip] = 'embed'
                continue
        s = step('sync ' + clip, [PY, 'sync_estimate.py', clip, '--write', '--accept-consistent'])
        if s.returncode:
            failed[clip] = 'sync'
            continue
        if not night():
            log('morning: stop before association'); break
        (d / 'sync_estimate.log').write_text((s.stdout or '')[-2000:], encoding='utf-8')
        r = step('people ' + clip, [PY, 'run_clip.py', clip])
        if r.returncode:
            # One clean CPU retry; retain both attempt logs and fail if it repeats.
            r = step('people_retry ' + clip, [PY, 'run_clip.py', clip],
                     {'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        if r.returncode == 0:
            done.append(clip)
        else:
            failed[clip] = 'people'
    json.dump({'day': DAY, 'span': SPAN, 'clips': [c for c, _ in clips], 'done': done, 'skipped': skipped, 'failed': failed, 'at': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(OUT / ('%s.json' % DAY), 'w'), indent=1)
    if done and (ROOT / 'visits_build.py').exists() and night():
        step('visits', [PY, 'visits_build.py', DAY])
    log('end: %d of %d windows' % (len(done), len(clips)))
