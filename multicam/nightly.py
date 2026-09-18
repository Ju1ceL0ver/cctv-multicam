"""The box works on its own: teachers label whole days at night, the review site stays
up, and the GPU is handed back before the shop opens.

Nothing here decides what is interesting -- it walks every recorded day from the oldest
one that is not yet covered, window by window, because a system that only works near the
entrance is not a system. Runs as a Windows logon task, so a reboot does not end it."""
import os, sys, time, glob
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from bg import spawn, running
from rawsource import RAW
from auto_label import windows

NIGHT_START = os.environ.get('RA_NIGHT_START', '21:00')
NIGHT_STOP = os.environ.get('RA_NIGHT_STOP', '09:45')
MINUTES = int(os.environ.get('RA_WINDOW', '10'))
LOG = os.path.join(ROOT, 'data', 'logs', 'nightly_supervisor.log')


def log(*a):
    line = '%s %s' % (datetime.now().strftime('%m-%d %H:%M:%S'), ' '.join(str(x) for x in a))
    print(line, flush=True)
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def recorded_days():
    days = set()
    for cam in ('cam1', 'cam2'):
        d = os.path.join(RAW, cam)
        if os.path.isdir(d):
            days |= {x for x in os.listdir(d) if x.isdigit() and len(x) == 8}
    return sorted(days)


def coverage(day):
    """(covered windows, all windows) of a day, by what the review site can already show."""
    try:
        todo = windows(day, MINUTES)
    except Exception:
        return 0, 0
    done = 0
    for t in todo:
        p = os.path.join(ROOT, 'data', 'raw_clips', 'c%s' % t.strftime('%H%M%S'), 'pieces_yolo26x-seg.json')
        done += os.path.exists(p)
    return done, len(todo)


def still_recording(day):
    last = 0.0
    for cam in ('cam1', 'cam2'):
        for p in glob.glob(os.path.join(RAW, cam, day, '*.mp4')):
            last = max(last, os.path.getmtime(p))
    return last and (time.time() - last) < 3600


def next_day():
    for day in recorded_days():
        if still_recording(day):
            continue
        done, total = coverage(day)
        if total and done < total:
            return day, done, total
    return None, 0, 0


def pass_alive():
    """A pass is running if its heartbeat is fresh; process names lie after execv."""
    hb = os.path.join(ROOT, 'data', 'logs', 'autolabel.heartbeat')
    if os.path.exists(hb) and time.time() - os.path.getmtime(hb) < 3600:
        return True
    return bool(running('auto_label.py')) or bool(running('detect_raw.py'))


def in_night(now):
    a = datetime.strptime(NIGHT_START, '%H:%M').time()
    b = datetime.strptime(NIGHT_STOP, '%H:%M').time()
    t = now.time()
    return t >= a or t < b


def main():
    log('supervisor up: nights %s-%s, %d-minute windows' % (NIGHT_START, NIGHT_STOP, MINUTES))
    while True:
        try:
            if not running('label_pieces.py'):
                spawn('labeler', ['label_pieces.py'])
                log('review site was down, restarted')
            now = datetime.now()
            job = running('auto_label.py')
            if in_night(now):
                if not pass_alive():
                    day, done, total = next_day()
                    if day:
                        log('night work on %s (%d/%d windows covered)' % (day, done, total))
                        spawn('autolabel_%s' % day, ['auto_label.py', day, str(MINUTES), NIGHT_STOP])
                    else:
                        log('every recorded day is covered, nothing to label')
                        time.sleep(1800)
            elif job:
                for l in job:
                    pid = l.split('|')[0].strip()
                    os.system('taskkill /PID %s /F /T > nul 2>&1' % pid)
                log('shop hours: stopped the teachers, GPU is free')
        except Exception as e:
            log('supervisor error:', repr(e))
        time.sleep(120)


if __name__ == '__main__':
    main()
