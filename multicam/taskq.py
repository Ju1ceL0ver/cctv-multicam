"""One queue for every long job, round the clock (04.10.2026).

data/taskq/queue.json: [{"id", "cmd": [...], "lane": "cpu"|"gpu", "python": optional interpreter, "env": {...},
"after": [ids], "status": "wait"|"run"|"done"|"fail"|"paused", ...}] -- edited by `add` (or by hand: it is re-read
every pass). Each pass (30 s):
  - cpu lane: up to CPU_SLOTS at once, below-normal priority, no card (CUDA_VISIBLE_DEVICES=-1);
  - gpu lane: one at a time (two at night). By day the card is the live counter's: its real speed is read from
    run_live.log (growth of frames= over the last ~2 min); below COUNTER_MIN frames/s the gpu task is suspended
    (psutil), and resumed when the counter is fine again. 21:00-10:00 nobody is guarded.
  - a task starts when everything in "after" is done; a failed one is retried once, then left as "fail".
Logs: data/logs/taskq_<id>.log(.err); state: data/taskq/status.json; the runner's own log: data/taskq/taskq.log.

usage: taskq.py run | add ID LANE [--after A,B] [--python PATH] [--env K=V,...] -- CMD ... | status"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
Q = ROOT / 'data' / 'taskq'
QUEUE = Q / 'queue.json'
CPU_SLOTS = int(os.environ.get('RA_TASKQ_CPU', '3'))
PASS_S = float(os.environ.get('RA_TASKQ_PASS', '30'))
COUNTER_MIN = float(os.environ.get('RA_TASKQ_COUNTER_MIN', '8'))
LIVE_LOG = ROOT / 'data' / 'logs' / 'run_live.log'


def load():
    return json.load(open(QUEUE, encoding='utf-8')) if QUEUE.exists() else []


def save(q):
    Q.mkdir(parents=True, exist_ok=True)
    tmp = str(QUEUE) + '.tmp'
    json.dump(q, open(tmp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    os.replace(tmp, QUEUE)


def say(*a):
    Q.mkdir(parents=True, exist_ok=True)
    with open(Q / 'taskq.log', 'a', encoding='utf-8') as f:
        print(time.strftime('%m-%d %H:%M:%S'), *a, file=f)


def night():
    h = time.localtime().tm_hour
    return h >= 21 or h < 10


def counter_fps():
    """Frames per second the live counter really processes now (None when it is not running / no news)."""
    try:
        lines = open(LIVE_LOG, encoding='utf-8', errors='ignore').read().splitlines()[-400:]
    except OSError:
        return None
    pts = []
    for l in lines:
        m = re.match(r'(\d\d):(\d\d):(\d\d) .*frames=(\d+)', l)
        if m:
            pts.append((int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3]), int(m[4])))
    if len(pts) < 2:
        return None
    now = time.localtime()
    sec_now = now.tm_hour * 3600 + now.tm_min * 60 + now.tm_sec
    if sec_now - pts[-1][0] > 180:
        return None                                   # no fresh line: the counter is down or idle
    recent = [p for p in pts if pts[-1][0] - p[0] <= 150]
    a, b = recent[0], recent[-1]
    if b[0] - a[0] < 30 or b[1] < a[1]:
        return None
    return (b[1] - a[1]) / (b[0] - a[0])


def alive(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return p.is_running() and p.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def tree(pid):
    import psutil
    try:
        p = psutil.Process(pid)
        return [p] + p.children(recursive=True)
    except psutil.Error:
        return []


def start(t):
    import psutil
    logs = ROOT / 'data' / 'logs'
    out = open(logs / ('taskq_%s.log' % t['id']), 'a', encoding='utf-8')
    err = open(logs / ('taskq_%s.log.err' % t['id']), 'a', encoding='utf-8')
    env = dict(os.environ, **{k: str(v) for k, v in (t.get('env') or {}).items()})
    if t['lane'] == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    py = t.get('python') or sys.executable.replace('pythonw.exe', 'python.exe')
    cmd = [py] + t['cmd'] if t['cmd'][0].endswith('.py') else t['cmd']
    flags = 0x00004000 if os.name == 'nt' else 0                     # BELOW_NORMAL_PRIORITY_CLASS
    p = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=out, stderr=err, creationflags=flags)
    t.update(status='run', pid=p.pid, started=time.strftime('%m-%d %H:%M:%S'), tries=t.get('tries', 0) + 1)
    say('start', t['id'], t['lane'], p.pid)
    return p


def run():
    procs = {}
    first = True
    while True:
        q = load()
        by = {t['id']: t for t in q}
        if first:                                     # adopt children of a previous runner that are still alive
            for t in q:
                if t['status'] in ('run', 'paused') and t.get('pid') and alive(t['pid']):
                    procs[t['id']] = _Handle(t['pid'])
            first = False
        # finished ones
        for t in q:
            if t['status'] in ('run', 'paused') and t['id'] in procs:
                rc = procs[t['id']].poll()
                if rc is not None:
                    del procs[t['id']]
                    if rc == 0:
                        t.update(status='done', ended=time.strftime('%m-%d %H:%M:%S'))
                    else:
                        t.update(status='wait' if t.get('tries', 0) < 2 else 'fail', rc=rc, ended=time.strftime('%m-%d %H:%M:%S'))
                    say('end', t['id'], rc, t['status'])
            elif t['status'] in ('run', 'paused') and t['id'] not in procs:
                if not t.get('pid') or not alive(t['pid']):                # the runner restarted: a lost child
                    t['status'] = 'wait' if t.get('tries', 0) < 2 else 'fail'
                    say('lost', t['id'], t['status'])
        # the card by day: the live counter first
        fps = counter_fps()
        guard = (not night()) and fps is not None and fps < COUNTER_MIN
        for t in q:
            if t['lane'] != 'gpu' or t['id'] not in procs:
                continue
            if guard and t['status'] == 'run':
                for p in tree(procs[t['id']].pid):
                    try:
                        p.suspend()
                    except Exception:
                        pass
                t['status'] = 'paused'; say('pause', t['id'], 'counter %.1f fps' % fps)
            elif not guard and t['status'] == 'paused':
                for p in tree(procs[t['id']].pid):
                    try:
                        p.resume()
                    except Exception:
                        pass
                t['status'] = 'run'; say('resume', t['id'], 'counter', fps)
        # start what is ready
        ready = lambda t: t['status'] == 'wait' and all(by.get(a, {}).get('status') == 'done' for a in t.get('after', []))
        running = lambda lane: sum(1 for t in q if t['lane'] == lane and t['status'] in ('run', 'paused')
                                   and t['cmd'][0] != 'taskq_wait.py')                 # a waiter takes no slot
        gpu_slots = 2 if night() else 1
        for t in q:
            if not ready(t):
                continue
            if t['cmd'][0] == 'taskq_wait.py' or t['lane'] == 'cpu' and running('cpu') < CPU_SLOTS or t['lane'] == 'gpu' and running('gpu') < gpu_slots and not guard:
                procs[t['id']] = start(t)
        # a task still running from before the runner restarted: follow it by its pid
        for t in q:
            if t['status'] in ('run', 'paused') and t['id'] not in procs and t.get('pid') and alive(t['pid']):
                procs[t['id']] = _Handle(t['pid'])
        st = {'updated': time.strftime('%m-%d %H:%M:%S'), 'night': night(), 'counter_fps': fps, 'guard': guard,
              'tasks': [{k: t.get(k) for k in ('id', 'lane', 'status', 'started', 'ended', 'tries', 'rc')} for t in q]}
        save(q)
        json.dump(st, open(Q / 'status.json', 'w'), indent=1)
        time.sleep(PASS_S)


class _Handle:
    """poll() for a child we only know by pid (psutil.wait with a zero timeout)."""
    def __init__(self, pid):
        import psutil
        self.pid = pid
        try:
            self.p = psutil.Process(pid)
        except psutil.Error:
            self.p = None

    def poll(self):
        import psutil
        if self.p is None:
            return 1
        try:
            return self.p.wait(timeout=0)
        except psutil.TimeoutExpired:
            return None
        except psutil.Error:
            return 1


def add(argv):
    tid, lane = argv[0], argv[1]
    rest = argv[2:]
    after, python, env = [], None, {}
    while rest and rest[0] != '--':
        k, v = rest[0], rest[1]
        if k == '--after':
            after = [x for x in v.split(',') if x]
        elif k == '--python':
            python = v
        elif k == '--env':
            env = dict(kv.split('=', 1) for kv in v.split(','))
        rest = rest[2:]
    cmd = rest[1:]
    q = [t for t in load() if t['id'] != tid]
    q.append({'id': tid, 'lane': lane, 'cmd': cmd, 'after': after, 'python': python, 'env': env, 'status': 'wait'})
    save(q)
    print('added', tid)


if __name__ == '__main__':
    if sys.argv[1] == 'run':
        run()
    elif sys.argv[1] == 'add':
        add(sys.argv[2:])
    elif sys.argv[1] == 'status':
        print(json.dumps(json.load(open(Q / 'status.json')), indent=1, ensure_ascii=False))
