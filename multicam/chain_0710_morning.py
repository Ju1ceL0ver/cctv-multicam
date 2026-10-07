"""07.10 morning: when the night detector (s31e2e_b) ends -> a 3-step smoke of the clip tracker -> the detector without
text (learned prompt), from the night's best, until 20:00. Log: data/logs/chain0710m.log."""
import os, subprocess, time
from pathlib import Path
import psutil
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
log = open(R / 'data' / 'logs' / 'chain0710m.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
while any('sam31_e2e_det.py' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])):
    time.sleep(60)
say('night detector ended; tracker smoke')
env = dict(os.environ, PYTHONIOENCODING='utf-8')
subprocess.run([PY, 'sam31_e2e_trk.py', 's31trk_smoke', 's31e2e_b'], cwd=R, env=dict(env, RA_TRK_SMOKE='3'), stdout=log, stderr=log)
say('detector without text, from the night best')
b = R / 'runs' / 's31e2e_b'
subprocess.run([PY, 'sam31_e2e_det.py', 's31e2e_c', str(b / 'best_last.pt'), '20:00'], cwd=R,
               env=dict(env, RA_E2E_PROMPT='1', RA_E2E_INIT_HEADS=str(b / 'best_heads.pt')), stdout=log, stderr=log)
say('done')
