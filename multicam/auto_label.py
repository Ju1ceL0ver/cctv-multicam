"""Auto-annotate trading days with the strongest models we have, for human review.

Nothing is trained here. The point is to produce labels good enough that a person only
has to correct them: segmentation masks from yolo26x-seg @1536, appearance from
OSNet-AIN x1.0, tracking and cross-camera fusion on top, and the review strips the
labelling page shows. Corrections come back as `gt_manual.json` per clip and are what
any future training will use.

The whole trading day is covered, in order, window after window -- not just the door.
The system has to work everywhere in the hall: at the racks, at the desk, in the depth
of the room where only cam2 sees anyone. Each run continues where the last one stopped
and halts at a deadline so the GPU is free when the shop opens.

usage: auto_label.py DAY [window_minutes] [deadline HH:MM]"""
import os, sys, json, subprocess, time, tempfile
from collections import Counter
from datetime import datetime, timedelta
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from rawsource import segments
from clip_registry import (clip_key, clip_ready, load_failures, record_failure, clear_failure,
                           set_aside, MAX_FAILURES)

PY = sys.executable
MODELS = r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models'
IMGSZ = int(os.environ.get('RA_IMGSZ', '1536'))
WEIGHTS = os.path.join(MODELS, 'yolo26x-seg.pt')   # a TensorRT build of it measured slower, not faster
STRIDE = int(os.environ.get('RA_STRIDE', '3'))
MARGIN = int(os.environ.get('RA_MARGIN', '45'))    # a 10-minute window takes about this long; do not start one to throw it away     # 8.3 fps: measured as good as 25 fps for identity, three times cheaper
LIVE = r'C:\Users\ArykovAA\cctv_ai\retail_analytics\runs\live\entrance_events.jsonl'


HEARTBEAT = os.path.join(ROOT, 'data', 'logs', 'autolabel%s.heartbeat' % os.environ.get('RA_WORKER', ''))
LAST_ERROR = ['']      # tail of stderr of the step that failed last; goes into the failure book


def beat(text=''):
    with open(HEARTBEAT, 'w', encoding='utf-8') as f:
        f.write('%d %s %s\n' % (os.getpid(), time.strftime('%H:%M:%S'), text))


def log(*a):
    print('%s %s' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)), flush=True)


def windows(day, minutes):
    """Every recorded window of the day, in order. Coverage is the point: a system that
    only works near the entrance is not a system."""
    segs1, segs2 = segments('cam1', day), segments('cam2', day)
    if not segs1 or not segs2:
        return []
    first = max(segs1[0][1], segs2[0][1]) + timedelta(seconds=30)
    last = min(segs1[-1][1], segs2[-1][1]) + timedelta(seconds=870)
    step = timedelta(minutes=minutes)
    out, t = [], first
    while t + step <= last:
        out.append(t)
        t += step
    return out


def door_events(day):
    """Only for reporting: how many door events fall in each window."""
    iso = '%s-%s-%s' % (day[:4], day[4:6], day[6:])
    out = []
    try:
        with open(LIVE, encoding='utf-8') as f:
            for line in f:
                if iso in line:
                    try:
                        out.append(datetime.fromisoformat(json.loads(line)['time_local']).replace(tzinfo=None))
                    except Exception:
                        pass
    except OSError:
        pass
    return out


def run(args, timeout=None):
    t0 = time.time()
    with tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stdout, tempfile.TemporaryFile(mode='w+', encoding='utf-8') as stderr:
        proc = subprocess.Popen([PY] + args, cwd=ROOT, stdout=stdout, stderr=stderr)
        while proc.poll() is None:
            beat(args[0])
            if timeout and time.time() - t0 > timeout:
                proc.kill(); proc.wait()
                raise subprocess.TimeoutExpired(args, timeout)
            time.sleep(15)
        stdout.seek(0); stderr.seek(0)
        r = subprocess.CompletedProcess(args, proc.returncode, stdout.read(), stderr.read())
    tail = (r.stdout or '').strip().splitlines()
    log('   ', args[0], '%.0f s' % (time.time() - t0), (tail[-1][:120] if tail else ''))
    if r.returncode:
        LAST_ERROR[0] = (r.stderr or '')[-600:]
        log('    FAILED rc=%d:' % r.returncode, (r.stderr or '')[-700:].replace('\n', ' | '))
    return r.returncode


def main():
    day = sys.argv[1]
    minutes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    hh, mm = (sys.argv[3] if len(sys.argv) > 3 else '09:45').split(':')
    now = datetime.now()
    deadline = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    log('day %s, %d-minute windows, deadline %s' % (day, minutes, deadline.strftime('%H:%M')))
    log('teacher %s @%d, every %d-th frame (%.1f fps), empty stretches skipped'
        % (os.path.basename(WEIGHTS), IMGSZ, STRIDE, 25.0 / STRIDE))
    todo = windows(day, minutes)
    rng = os.environ.get('RA_RANGE')      # 'HH:MM-HH:MM': lets a second worker take another part of the day
    if rng:
        a, b = [datetime.strptime(x, '%H:%M').time() for x in rng.split('-')]
        todo = [t for t in todo if a <= t.time() < b]
        log('worker limited to %s: %d windows' % (rng, len(todo)))
    events = door_events(day)
    log('windows to cover: %d (%s .. %s), door events that day: %d'
        % (len(todo), todo[0].strftime('%H:%M') if todo else '-', todo[-1].strftime('%H:%M') if todo else '-', len(events)))
    done = 0
    beat('start')
    for start in todo:
        n = sum(1 for e in events if start <= e < start + timedelta(minutes=minutes))
        if datetime.now() >= deadline - timedelta(minutes=MARGIN):
            log('less than %d minutes left before %s: a window would not finish, stopping'
                % (MARGIN, deadline.strftime('%H:%M'))); break
        clip = clip_key(day, start, os.path.join(ROOT, 'data', 'raw_clips'))
        d = os.path.join('data', 'raw_clips', clip)
        if clip_ready(d):
            continue
        if set_aside(load_failures(ROOT).get(clip)):
            log('window %s keeps failing, set aside until tomorrow night' % start.strftime('%H:%M'))
            continue
        log('window %s (%d door events in it)' % (start.strftime('%H:%M'), n))
        beat(clip)

        def failed(stage):
            tries = record_failure(ROOT, clip, day, stage, LAST_ERROR[0])
            log('    %s failed, attempt %d of %d before the window is set aside' % (stage, tries, MAX_FAILURES))

        if not os.path.exists(os.path.join(d, 'dets_yolo26x-seg.npz')):
            if run(['detect_raw.py', clip, day, start.strftime('%H:%M:%S'), str(minutes * 60),
                    WEIGHTS, str(IMGSZ), str(STRIDE)]):
                log('    segmentation failed, skipping'); failed('detect_raw.py'); continue
        if not os.path.exists(os.path.join(d, 'emb_osnet_ain_x1_0_msmt17.npz')):
            run(['embed_clip.py', clip, 'osnet_ain_x1_0_msmt17.pt'])   # only if the pass above was run without it
        if run(['sync_estimate.py', clip, '--write', '--accept-consistent']):
            log('    synchronization failed; review required'); failed('sync_estimate.py'); continue
        if run(['run_clip.py', clip]): failed('run_clip.py'); continue
        if run(['export_pieces.py', clip]): failed('export_pieces.py'); continue
        if run(['group_pieces.py', clip]): failed('group_pieces.py'); continue      # what the system thinks is one person: review is per person, not per piece
        clear_failure(ROOT, clip)
        run(['day_visits.py', day])
        done += 1
        log('window %s ready for review' % start.strftime('%H:%M'))
    log('AUTO LABEL DONE, %d new windows' % done)


if __name__ == '__main__':
    main()
