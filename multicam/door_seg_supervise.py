"""Keep one door_seg training run until plateau, yielding the GPU to the live door.

python door_seg_supervise.py RUN [--until 09:40]
No torch/GPU context in supervisor; train checkpoint validation is mandatory.
STOP means completion or nonfinite fault and is never removed automatically.
At the daytime boundary save/pause, resume the same run after 21:00 if needed.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def night(now, until='09:40'):
    clock = now.strftime('%H:%M')
    return clock >= '21:00' or clock < until


def deadline(now, until):
    hh, mm = map(int, until.split(':'))
    end = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    if end <= now:
        end += datetime.timedelta(days=1)
    return end


def live_up():
    import psutil
    try:
        pid = int((ROOT / 'data/live/door_live.pid').read_text().split()[0])
        return 'door_live' in ' '.join(psutil.Process(pid).cmdline())
    except (OSError, ValueError, psutil.Error):
        return False


def must_yield():
    if live_up():
        return True
    plan = json.loads((ROOT / 'data/keeper_plan.json').read_text(encoding='utf-8'))
    a, b = plan.get('door_hours', '09:50-21:00').split('-')
    if not plan.get('door') or not (a <= datetime.datetime.now().strftime('%H:%M') < b):
        return False
    import socket
    try:
        with socket.create_connection(('192.168.99.241', 554), timeout=3):
            return True
    except OSError:
        return False


def gpu_busy():
    try:
        r = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, timeout=10, check=True)
        return int(r.stdout.strip().split()[0]) > 20
    except (OSError, ValueError, subprocess.SubprocessError):
        return True


def main(name, until='09:40'):
    import psutil
    run = ROOT / 'runs' / name
    if not (run / 'last.pt').exists():
        raise FileNotFoundError('stage the last verified checkpoint as runs/%s/last.pt first' % name)
    pidfile = run / 'supervisor.pid'
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text())
            if 'door_seg_supervise.py' in ' '.join(psutil.Process(pid).cmdline()):
                raise RuntimeError('supervisor already running: %d' % pid)
        except (ValueError, psutil.NoSuchProcess):
            pass
    pidfile.write_text(str(os.getpid()))
    def say(text):
        print(time.strftime('%d.%m %H:%M:%S'), text, flush=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith('RA_')}
    env.update(RA_DS_GLUED='1', RA_DS_LR='1e-4', RA_DS_EVAL_N='400', PYTHONIOENCODING='utf-8')
    say('waiting for free GPU; stop on plateau; daytime pause until 21:00')
    try:
        while not (run / 'STOP').exists():
            idle = 0
            while idle < 3 and not (run / 'STOP').exists():
                idle = idle + 1 if night(datetime.datetime.now(), until) and not must_yield() and not gpu_busy() else 0
                time.sleep(20)
            if (run / 'STOP').exists():
                break
            (run / 'PAUSE').unlink(missing_ok=True)
            end = deadline(datetime.datetime.now(), until)
            with (run / 'supervised.log').open('a', encoding='utf-8') as output:
                p = subprocess.Popen([sys.executable, str(ROOT / 'door_seg.py'), 'train', name, until], cwd=ROOT,
                                     env=env, stdout=output, stderr=subprocess.STDOUT)
                say('started own training PID %d; boundary %s' % (p.pid, end))
                paused = False
                while p.poll() is None:
                    if datetime.datetime.now() >= end or must_yield():
                        (run / 'PAUSE').write_text('day boundary or live door needs GPU', encoding='utf-8')
                        paused = True
                        try:
                            p.wait(timeout=45)
                        except subprocess.TimeoutExpired:
                            # Only our own verified child; never kill python by name.
                            cmd = psutil.Process(p.pid).cmdline()
                            if str(ROOT / 'door_seg.py') in cmd and name in cmd:
                                subprocess.run(['taskkill', '/PID', str(p.pid), '/F', '/T'], capture_output=True)
                                p.wait(timeout=15)
                            else:
                                raise RuntimeError('child command changed; refusing to kill PID %d' % p.pid)
                        say('paused, last verified checkpoint retained')
                        break
                    time.sleep(10)
                say('training exit %s' % p.returncode)
            if (run / 'STOP').exists():
                break
            if p.returncode != 0 and not paused:
                say('unexpected failure; no automatic retry: inspect supervised.log')
                return 1
            # A completed time window resumes only during another allowed night.
            time.sleep(20)
        say('finished: %s' % (run / 'STOP').read_text(encoding='utf-8'))
        return 0
    finally:
        if pidfile.exists() and pidfile.read_text() == str(os.getpid()):
            pidfile.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name')
    parser.add_argument('--until', default='09:40')
    args = parser.parse_args()
    sys.exit(main(args.name, args.until))
