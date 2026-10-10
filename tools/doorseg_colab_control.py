"""Serialize Mac-side Colab checks and keep the active background job observable.

python tools/doorseg_colab_control.py keepalive
python tools/doorseg_colab_control.py exec --file /tmp/check.py

Based on tennis-pose/tools/colab_keepalive.sh. All checks for this session
should use this wrapper: overlapping Colab executions have previously hung.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / 'data' / 'logs'
PROBE = '''import json,os
from pathlib import Path
import psutil
r=Path('/content/doorseg/runs/doorseg_c_background')
status=json.loads((r/'colab_status.json').read_text()) if (r/'colab_status.json').exists() else {}
pidfile=Path('/content/doorseg_colab_runner.pid')
alive=False
if pidfile.exists():
    try:
        p=psutil.Process(int(pidfile.read_text()))
        alive=p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and '/content/doorseg/door_seg_colab.py' in p.cmdline()
    except (psutil.Error,ValueError): pass
print(json.dumps({'kernel_ping':'ok','runner_alive':alive,'phase':status.get('phase'),'utc':status.get('utc')}))
'''


def execute(session, file=None, code=None):
    LOGS.mkdir(parents=True, exist_ok=True)
    lockpath = LOGS / ('colab_' + session + '.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        command = ['colab', 'exec', '-s', session, '--timeout', '60']
        if file:
            command.extend(['--file', file])
        return subprocess.run(command, input=code, text=True, capture_output=True, timeout=100)


def main(args):
    if args.command == 'exec':
        result = execute(args.session, file=args.file, code=None if args.file else sys.stdin.read())
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        return result.returncode
    pidfile = LOGS / ('colab_' + args.session + '_keepalive.pid')
    LOGS.mkdir(parents=True, exist_ok=True)
    # OS lock prevents two pingers even after a stale PID file.
    with pidfile.with_suffix('.singleton').open('a') as singleton:
        try:
            fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Keepalive already running for ' + args.session)
        pidfile.write_text(str(os.getpid()))
        try:
            while True:
                try:
                    result = execute(args.session, code=PROBE)
                    print(time.strftime('%d.%m %H:%M:%S'), result.stdout.strip(),
                          'exit', result.returncode, flush=True)
                    if result.returncode == 0 and '"runner_alive": false' in result.stdout:
                        print('Job no longer active; keepalive stopped.', flush=True)
                        break
                except subprocess.TimeoutExpired:
                    print(time.strftime('%d.%m %H:%M:%S'), 'kernel probe timed out', flush=True)
                time.sleep(180)
        finally:
            pidfile.unlink(missing_ok=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session', default='doorseg-c')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('keepalive')
    check = sub.add_parser('exec')
    check.add_argument('--file')
    sys.exit(main(parser.parse_args()))
