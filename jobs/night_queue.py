"""Tonight's GPU queue: nano student -> small student -> raw 25 fps tracklets.

Each stage is started only after the previous one has actually released the
GPU (matched by command line, never by bare PID), and each has its own hard
stop, so a stage that overruns cannot push the next one into the live
service's 10:00 window.
"""
import os, time, subprocess
from datetime import datetime

HOME = r'C:\Users\ArykovAA'
PY = os.path.join(HOME, 'AppData', 'Local', 'miniconda3', 'envs', 'cctv_base', 'python.exe')
LOG = os.path.join(HOME, '_night_queue.log')
F = 0x00000200 | 0x00000008 | 0x01000000


def log(*a):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('%s %s\n' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(str(x) for x in a)))


def running(needle):
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV', '/NH'],
                       capture_output=True)
    for line in r.stdout.decode('cp866', 'replace').splitlines():
        if '","' not in line:
            continue
        pid = line.split('","')[1]
        q = ('Get-CimInstance Win32_Process -Filter "ProcessId=%s" | '
             'Select-Object -ExpandProperty CommandLine' % pid)
        cl = subprocess.run(['powershell', '-NoProfile', '-Command', q],
                            capture_output=True).stdout.decode('utf-8', 'replace')
        if needle in cl and '_night_queue' not in cl:
            return True
    return False


def wait_for_exit(needle):
    while running(needle):
        time.sleep(60)
    time.sleep(20)


def start(script, env_extra, stdout_name):
    env = dict(os.environ, **env_extra)
    p = subprocess.Popen([PY, os.path.join(HOME, script)], cwd=HOME, env=env, creationflags=F,
                         stdout=open(os.path.join(HOME, stdout_name), 'w'),
                         stderr=subprocess.STDOUT)
    log('started', script, env_extra, 'pid', p.pid)


def hours_until(hh, mm, margin_min):
    now = datetime.now()
    end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    return max(0.25, ((end - now).total_seconds() / 60 - margin_min) / 60)


def main():
    log('queue up; waiting for the nano run')
    wait_for_exit('_train_student.py')
    log('nano finished')
    if datetime.now().hour < 6:
        budget = hours_until(6, 10, 10)
        start('_train_student.py', {'RA_TRAIN_SIZE': 's', 'RA_TRAIN_BATCH': '4',
                                    'RA_TRAIN_HOURS': '%.2f' % budget,
                                    'RA_TRAIN_STOP': '06:20'}, '_train_student_s.stdout.log')
        time.sleep(120)
        wait_for_exit('_train_student.py')
        log('small finished')
    if datetime.now().hour < 9 or (datetime.now().hour == 9 and datetime.now().minute < 20):
        open(os.path.join(HOME, '_tracklets.log'), 'a').close()
        start('_harvest_tracklets.py', {'RA_TRACKLETS_SOURCE': 'raw'}, '_tracklets_raw.stdout.log')
    log('queue done')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        log('FATAL\n' + traceback.format_exc())
