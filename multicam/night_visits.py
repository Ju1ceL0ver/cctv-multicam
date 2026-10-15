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
DAY = sys.argv[1] if len(sys.argv) > 1 else time.strftime('%Y%m%d')
SPAN = sys.argv[2] if len(sys.argv) > 2 else '13:01-17:00'          # the recorder starts a few seconds after 13:00
WIN = 600
OUT = ROOT / 'data' / 'night_visits'; OUT.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / ('%s.log' % DAY), 'a', encoding='utf-8')


def log(m):
    LOG.write('%s %s\n' % (time.strftime('%m-%d %H:%M:%S'), m)); LOG.flush()


def night():
    h = time.strftime('%H:%M')
    return h >= '21:05' or h < '09:30'


def step(name, cmd, env=None):
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding='utf-8', errors='replace',
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', **(env or {})))
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
    done = []
    for clip, hms in clips:
        if not night():
            log('morning: stop'); break
        d = ROOT / 'data' / 'raw_clips' / clip
        if (d / 'people_yolo26x-seg.json').exists():
            done.append(clip); continue
        if not (d / 'dets_yolo26x-seg.npz').exists():
            r = step('detect ' + clip, [PY, 'detect_raw.py', clip, DAY, hms, str(WIN)], {'RA_STRIDE': '3'})
            if r.returncode:
                continue
        if not list(d.glob('emb_*.npz')):
            step('embed ' + clip, [PY, 'embed_clip.py', clip, 'osnet_ain_x1_0_msmt17.pt'])
        s = step('sync ' + clip, [PY, 'sync_estimate.py', clip])
        (d / 'sync_estimate.log').write_text((s.stdout or '')[-2000:], encoding='utf-8')
        r = step('people ' + clip, [PY, 'run_clip.py', clip])
        if r.returncode == 0:
            done.append(clip)
    json.dump({'day': DAY, 'span': SPAN, 'clips': [c for c, _ in clips], 'done': done, 'at': time.strftime('%Y-%m-%d %H:%M:%S')},
              open(OUT / ('%s.json' % DAY), 'w'), indent=1)
    if (ROOT / 'visits_build.py').exists() and night():
        step('visits', [PY, 'visits_build.py', DAY])
    log('end: %d of %d windows' % (len(done), len(clips)))
