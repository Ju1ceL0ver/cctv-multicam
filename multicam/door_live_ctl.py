"""Start / stop / status of the live door on camera 1 (08.10.2026).

  door_live_ctl.py start   -- camera 1 answers? then door_live.py (venv_sam3, the card) in the background, its pid in
                              data/live/door_live.pid; a second start while it runs does nothing
  door_live_ctl.py stop    -- kills exactly that pid (its tree: the role worker and ffmpeg children), nothing by name
  door_live_ctl.py status  -- running or not, the last session line, events today

The side setting is data/door_v2/side_final.json (door_combo.Live reads it), the stride comes from it as well."""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LIVE = ROOT / 'data' / 'live'
PID = LIVE / 'door_live.pid'
SAM3 = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
STUDENT = r'runs\s31micro_a\last.pt'
RULE = 'micro1s3_live3'                 # only a fallback name: with side_final.json the door rule is not used
VCVARS = r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'


def alive(pid):
    try:
        out = subprocess.run(['tasklist', '/FI', 'PID eq %d' % pid, '/FO', 'CSV', '/NH'], capture_output=True, text=True).stdout
        return str(pid) in out
    except OSError:
        return False


def camera_ok(host='192.168.99.241', port=554):
    try:
        with socket.create_connection((host, port), timeout=3):
            return True
    except OSError:
        return False


def running():
    if PID.exists():
        try:
            pid = int(PID.read_text().strip())
        except ValueError:
            return None
        return pid if alive(pid) else None
    return None


def start(until='21:00'):
    LIVE.mkdir(parents=True, exist_ok=True)
    pid = running()
    if pid:
        print('already running, pid', pid)
        return
    if not camera_ok():
        print('camera 1 does not answer (192.168.99.241:554) -- not started')
        return
    log = open(LIVE / 'door_live.out', 'a', encoding='utf-8')
    log.write('\n==== start %s ====\n' % time.strftime('%Y-%m-%d %H:%M:%S'))
    sf = ROOT / 'data' / 'door_v2' / 'side_final.json'
    side = json.load(open(sf)) if sf.exists() else {}
    env = dict(os.environ, PYTHONIOENCODING='utf-8', RA_DOOR_PIECES='1', RA_DOOR_UNTIL=os.environ.get('RA_DOOR_UNTIL', until))
    if side.get('fps_scale'):                     # 08.10: SAM 3.1's track rules counted in frames of this stride
        env['RA_S31_FPS_SCALE'] = str(side.get('stride', 6))
    args = [SAM3, str(ROOT / 'door_live.py'), STUDENT, '-', RULE]
    if side.get('compile'):                       # 08.10: SAM 3.1's own torch.compile -- needs the MSVC environment
        env['RA_S31_COMPILE'] = '1'
        args = 'call "%s" >nul && %s' % (VCVARS, subprocess.list2cmdline(args))
    p = subprocess.Popen(args, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env, shell=isinstance(args, str),
                         creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) | getattr(subprocess, 'DETACHED_PROCESS', 0))
    PID.write_text(str(p.pid))
    print('started, pid', p.pid, '-- log', LIVE / 'live_cam1.log')


def night(day=None):
    """The night teacher over the day's clips (door_night.py, venv_sam3, the card) -- once per day."""
    day = day or time.strftime('%Y%m%d')
    mark = LIVE / 'night' / ('%s.started' % day)
    if mark.exists() or (LIVE / 'night' / ('%s.json' % day)).exists():
        print('night teacher already ran for', day)
        return
    mark.parent.mkdir(parents=True, exist_ok=True)
    mark.write_text(time.strftime('%Y-%m-%dT%H:%M:%S'))
    log = open(LIVE / 'night' / 'night.out', 'a', encoding='utf-8')
    p = subprocess.Popen([SAM3, str(ROOT / 'door_night.py'), day], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT,
                         env=dict(os.environ, PYTHONIOENCODING='utf-8'),
                         creationflags=getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) | getattr(subprocess, 'DETACHED_PROCESS', 0))
    print('night teacher started for', day, 'pid', p.pid)


def stop():
    pid = running()
    if not pid:
        print('not running')
        return
    r = subprocess.run(['taskkill', '/PID', str(pid), '/F', '/T'], capture_output=True, text=True)
    print('stopped' if r.returncode == 0 else 'taskkill: %s' % (r.stdout + r.stderr).strip())
    PID.unlink(missing_ok=True)


def status():
    pid = running()
    print('running, pid %s' % pid if pid else 'not running')
    lg = LIVE / 'live_cam1.log'
    if lg.exists():
        lines = lg.read_text(encoding='utf-8', errors='replace').splitlines()
        sess = [l for l in lines if ' session ' in l]
        print('last session:', sess[-1] if sess else '-')
    ev = LIVE / 'events_cam1.jsonl'
    if ev.exists():
        today = time.strftime('%Y-%m-%d')
        es = [json.loads(l) for l in ev.read_text(encoding='utf-8').splitlines() if today in l]
        print('today: %d in, %d out (staff %d)' % (sum(e['kind'] == 'in' for e in es), sum(e['kind'] == 'out' for e in es),
                                                    sum(e.get('role') == 'staff' for e in es)))


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
    if cmd == 'night':
        night(*sys.argv[2:3])
    else:
        {'start': start, 'stop': stop, 'status': status}[cmd]()
