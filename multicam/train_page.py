"""/train: live dashboard of a training run (v2).

Reads runs/<run>/status.json, log.jsonl (a row every 20 steps), epochs.jsonl (the tests after every epoch).
Extra for the live feel: a 1 Hz sampler of the video card (nvidia-smi) and of the machine (psutil, when there),
kept in a ring buffer; /api/train/live hands out only what is new since the page's last call, so the page can ask
once a second at a few hundred bytes."""
import collections
import json
import os
import subprocess
import sys
import threading
import time

HW = collections.deque(maxlen=1800)          # 30 minutes at 1 Hz
_HW_LOCK = threading.Lock()
_HW_STARTED = []
_GPU_Q = 'utilization.gpu,memory.used,memory.total,power.draw,power.limit,temperature.gpu,clocks.sm,fan.speed'


def _num(x):
    try:
        return float(x)
    except ValueError:
        return None


def _sample():
    s = {'ts': round(time.time(), 2)}
    try:
        kw = {'creationflags': 0x08000000} if sys.platform == 'win32' else {}
        out = subprocess.run(['nvidia-smi', '--query-gpu=' + _GPU_Q, '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5, **kw).stdout.strip().split('\n')[0].split(',')
        v = [_num(x.strip()) for x in out]
        for k, x in zip(('gpu', 'vram', 'vram_total', 'power', 'power_limit', 'temp', 'clock', 'fan'), v):
            s[k] = x
    except Exception:
        pass
    try:
        import psutil
        s['cpu'] = psutil.cpu_percent(interval=None)
        vm = psutil.virtual_memory()
        s['ram'] = round(vm.used / 2 ** 30, 1)
        s['ram_total'] = round(vm.total / 2 ** 30, 1)
    except Exception:
        pass
    return s


def _sampler():
    while True:
        t0 = time.time()
        s = _sample()
        with _HW_LOCK:
            HW.append(s)
        time.sleep(max(0.05, 1.0 - (time.time() - t0)))


def _start_sampler():
    if not _HW_STARTED:
        _HW_STARTED.append(1)
        threading.Thread(target=_sampler, daemon=True, name='train-hw').start()


def _lines(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8', errors='replace') as f:
        return f.read().splitlines()


def _parse(lines):
    out = []
    for l in lines:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out


def _read(path, last=None):
    lines = _lines(path)
    return _parse(lines[-last:] if last else lines)


def _status(d):
    p = os.path.join(d, 'status.json')
    if not os.path.exists(p):
        return {}
    try:
        st = json.load(open(p))
    except ValueError:
        return {}
    st['age_s'] = round(max(0.0, time.time() - os.path.getmtime(p)), 1)
    return st


def register(app, root):
    from flask import jsonify, render_template, request
    _start_sampler()

    def newest():
        runs_dir = os.path.join(root(), 'runs')
        cands = [x for x in os.listdir(runs_dir) if x.startswith('v2') and os.path.exists(os.path.join(runs_dir, x, 'status.json'))
                 and not os.path.exists(os.path.join(runs_dir, x, 'remote.json'))] if os.path.isdir(runs_dir) else []   # a borrowed machine's run: by hand
        return max(cands, key=lambda x: os.path.getmtime(os.path.join(runs_dir, x, 'status.json'))) if cands else 'v2_a'

    def run_dir():
        run = os.path.basename(request.args.get('run') or newest())
        return run, os.path.join(root(), 'runs', run)

    @app.get('/train')
    def train_page():
        return render_template('train.html')

    @app.get('/api/train/live')
    def train_live():
        """Everything new since the caller's counters: n log rows, en epoch rows, hw = last sample time."""
        run, d = run_dir()
        n, en = request.args.get('n', 0, int), request.args.get('en', 0, int)
        hw = request.args.get('hw', 0, float)
        log = _lines(os.path.join(d, 'log.jsonl'))
        eps = _lines(os.path.join(d, 'epochs.jsonl'))
        reset = n > len(log) or en > len(eps)
        if reset:
            n = en = 0
        with _HW_LOCK:
            samples = [s for s in HW if s['ts'] > hw] if hw else list(HW)
        runs_dir = os.path.join(root(), 'runs')
        return jsonify({
            'run': run,
            'runs': sorted(x for x in os.listdir(runs_dir) if x.startswith('v2')) if os.path.isdir(runs_dir) else [],
            'reset': reset, 'n': len(log), 'en': len(eps),
            'log': _parse(log[n:]), 'epochs': _parse(eps[en:]),
            'status': _status(d), 'hw': samples, 'now': round(time.time(), 2), 'newest': newest(),
        })

    @app.get('/api/train/queue')
    def train_queue():
        """The variant queue (variant_queue.py): what runs, what waits, why each one stopped."""
        p = os.path.join(root(), 'data', 'v2_queue', 'state.json')
        try:
            return jsonify(json.load(open(p, encoding='utf-8')))
        except (OSError, ValueError):
            return jsonify({'items': []})

    @app.get('/api/train')
    def train_api():
        """The old full dump, kept for anything that still reads it."""
        run, d = run_dir()
        keep = ('step', 'obj', 'l1', 'giou', 'bce', 'dice', 'cen', 'bnd', 'state', 'place', 'zone', 'reid', 'supcon', 'same',
                'step_s', 'gpu_s', 'mem_gb')
        log = [{k: r[k] for k in keep if k in r} for r in _read(os.path.join(d, 'log.jsonl'))]
        runs_dir = os.path.join(root(), 'runs')
        choices = sorted(x for x in os.listdir(runs_dir) if x.startswith('v2')) if os.path.isdir(runs_dir) else []
        return jsonify({'run': run, 'runs': choices, 'status': _status(d), 'log': log,
                        'epochs': _read(os.path.join(d, 'epochs.jsonl'))})
