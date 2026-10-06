"""06.10.2026: the small SAM 3.1, one stage after another on the card (no live counter any more, the card is free all day).
encoder distillation (runs/s31micro_a, already running) -> real-module speed (sam31_micro_speed.py, waits for it)
-> short heads in the teacher's shadow until HEADS_STOP (runs/s31heads_a) -> every door stretch of 17-19.09 through
the small SAM (door_micro.py, data/door_v2/<day>_micro1.jsonl.gz). Each step's log: data/logs/<name>.log."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = str(ROOT.parent / 'venv_sam3' / 'Scripts' / 'python.exe')
HEADS_STOP = sys.argv[1] if len(sys.argv) > 1 else '16:00'
WAIT_FOR = sys.argv[2] if len(sys.argv) > 2 else 'sam31_distill.py'           # the encoder training to wait for
STUDENT = sys.argv[3] if len(sys.argv) > 3 else 'runs/s31micro_a/last.pt'     # its checkpoint


def wait_proc(pattern):
    import psutil
    while any(pattern in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])):
        time.sleep(30)


def step(name, args):
    print(time.strftime('%H:%M'), 'start', name, args, flush=True)
    with open(ROOT / 'data' / 'logs' / (name + '.log'), 'a', encoding='utf-8', errors='replace') as log:
        r = subprocess.run([PY] + args, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    print(time.strftime('%H:%M'), 'done', name, r.returncode, flush=True)
    return r.returncode


wait_proc(WAIT_FOR)
best = ROOT / Path(STUDENT).parent / 'best.pt'
if STUDENT.endswith('/last.pt') and best.exists():         # the encoder that did best in the full-pipeline check
    STUDENT = str(Path(STUDENT).parent / 'best.pt').replace('\\', '/')
print(time.strftime('%H:%M'), 'student encoder', STUDENT, flush=True)
step('microspeed', ['sam31_micro_speed.py', STUDENT])
step('s31heads', ['sam31_heads.py', 's31heads_a', STUDENT, HEADS_STOP])
step('doormicro', ['door_micro.py', STUDENT, 'micro1', '20260919', '20260917', '20260918',
                   '--heads', 'runs/s31heads_a/heads.pt'])
