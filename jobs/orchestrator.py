"""Hand the GPU from the teacher harvest to the tracklet harvest at 08:00.

The teacher pass will not finish its 14k-frame budget much before the 09:55
deadline, and the tracklet pass needs a usable slice of GPU time tonight. By
08:00 the teacher pass has had three hours and roughly 10k frames, which is
already more than the student detector needs, so the remainder of the window
is worth more spent on ReID material.

Killing by PID alone is not safe on this machine -- several python.exe
processes run concurrently (Jupyter, its kernel, the live service under
SYSTEM) and PIDs get reused. Every kill here is matched on the command line
first.
"""

import os, sys, time, json, subprocess
from datetime import datetime, timedelta

HOME = r'C:\Users\ArykovAA'
PY = os.path.join(HOME, 'AppData', 'Local', 'miniconda3', 'envs', 'cctv_base', 'python.exe')
LOG = os.path.join(HOME, '_orchestrator.log')
SWITCH_AT = (8, 0)


def log(*a):
    line = '%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(str(x) for x in a))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def pids_running(needle):
    """PIDs of python.exe whose command line contains `needle`."""
    found = []
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
                       capture_output=True)
    for line in r.stdout.decode('cp866', errors='replace').splitlines():
        if '","' not in line:
            continue
        pid = line.split('","')[1]
        q = ('Get-CimInstance Win32_Process -Filter "ProcessId=%s" | '
             'Select-Object -ExpandProperty CommandLine' % pid)
        cl = subprocess.run(['powershell', '-NoProfile', '-Command', q],
                            capture_output=True).stdout.decode('utf-8', errors='replace')
        if needle in cl:
            found.append((pid, cl.strip()[:120]))
    return found


def main():
    target = datetime.now().replace(hour=SWITCH_AT[0], minute=SWITCH_AT[1],
                                    second=0, microsecond=0)
    if target <= datetime.now():
        target += timedelta(days=1)
    log('orchestrator up; will switch at %s' % target.isoformat(timespec='seconds'))

    while datetime.now() < target:
        time.sleep(30)
        if not pids_running('_teacher_harvest.py'):
            log('teacher harvest finished on its own; switching early')
            break

    for pid, cl in pids_running('_teacher_harvest.py'):
        log('stopping teacher harvest pid=%s cmd=%s' % (pid, cl))
        subprocess.run(['taskkill', '/PID', pid, '/T', '/F'], capture_output=True)
    time.sleep(20)      # let CUDA unwind before the next process claims VRAM

    log('starting tracklet harvest')
    env = dict(os.environ)
    env.pop('RA_TRACKLETS_SMOKE', None)
    p = subprocess.Popen([PY, os.path.join(HOME, '_harvest_tracklets.py')],
                         cwd=HOME, env=env,
                         creationflags=0x00000200 | 0x00000008 | 0x01000000,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log('tracklet harvest pid=%d' % p.pid)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        log('FATAL\n' + traceback.format_exc())
