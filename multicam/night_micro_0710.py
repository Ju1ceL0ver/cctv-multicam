"""07.10 night: SAM-micro -- a 2-step smoke, the speed of an untrained one at work, then training until 09:30.
Log: data/logs/nightmicro.log"""
import os, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
log = open(R / 'data' / 'logs' / 'nightmicro.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
env = dict(os.environ, PYTHONIOENCODING='utf-8')
say('smoke')
r = subprocess.run([PY, 'micro_sam.py', 'train', 'msam_smoke'], cwd=R, env=dict(env, RA_MS_SMOKE='2'), capture_output=True, text=True)
log.write(r.stdout[-1500:] + r.stderr[-2500:] + '\n'); log.flush()
if 'smoke 2' not in r.stdout:
    say('smoke failed'); sys.exit(1)
say('training until 09:30')
subprocess.run([PY, 'micro_sam.py', 'train', 'msam_a', '09:30'], cwd=R, env=env, stdout=log, stderr=log)
say('done')
