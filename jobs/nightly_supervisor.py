"""Run the nightly collector once per night, for as many nights as the raw
recorders run.

Starts each night at 21:10 -- ten minutes after both the live service and the
raw recorders close the trading day, so the day's last segment is on disk and
the GPU is free. If this supervisor itself is (re)started in the early hours,
it processes the day that just ended right away rather than skipping it.

Each night is a separate child process: a crash in one night's collection
cannot stop the next.
"""
import os, time, subprocess
from datetime import datetime, timedelta

HOME = r'C:\Users\ArykovAA'
PY = os.path.join(HOME, 'AppData', 'Local', 'miniconda3', 'envs', 'cctv_base', 'python.exe')
LOG = os.path.join(HOME, '_nightly_supervisor.log')
NIGHTS = int(os.environ.get('RA_NIGHTS', '5'))
START = (21, 10)
F = 0x00000200 | 0x00000008 | 0x01000000


def log(*a):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(str(x) for x in a)))


def next_run():
    now = datetime.now()
    start = now.replace(hour=START[0], minute=START[1], second=0, microsecond=0)
    return (start if start > now else now), now.strftime('%Y%m%d')


def main():
    log('supervisor up: %d night(s)' % NIGHTS)
    done = set()
    while len(done) < NIGHTS:
        at, day = next_run()
        if day in done:
            time.sleep(600)
            continue
        log('next night: day %s at %s' % (day, at.isoformat(timespec='minutes')))
        while datetime.now() < at:
            time.sleep(30)
        env = dict(os.environ, RA_NIGHT_DAY=day)
        env.pop('RA_NIGHT_SMOKE', None)
        p = subprocess.Popen([PY, os.path.join(HOME, '_nightly_collect.py')], cwd=HOME, env=env,
                             creationflags=F,
                             stdout=open(os.path.join(HOME, '_nightly_%s.stdout.log' % day), 'w'),
                             stderr=subprocess.STDOUT)
        log('collector for %s started, pid %d' % (day, p.pid))
        p.wait()
        log('collector for %s exited with %s' % (day, p.returncode))
        done.add(day)
    log('all nights done')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        log('FATAL\n' + traceback.format_exc())
