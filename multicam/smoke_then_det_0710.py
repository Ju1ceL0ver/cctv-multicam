"""07.10: the tracker smoke alone on the card, then the detector without text goes on from s31e2e_c's best (run s31e2e_d)."""
import os, subprocess, time
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
log = open(R / 'data' / 'logs' / 'chain0710m.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
env = dict(os.environ, PYTHONIOENCODING='utf-8')
say('tracker smoke, alone on the card')
subprocess.run([PY, 'sam31_e2e_trk.py', 's31trk_smoke', 's31e2e_b'], cwd=R, env=dict(env, RA_TRK_SMOKE='3'), stdout=log, stderr=log)
c = R / 'runs' / 's31e2e_c'
say('detector without text goes on from s31e2e_c best')
subprocess.run([PY, 'sam31_e2e_det.py', 's31e2e_d', str(c / 'best_last.pt'), '20:00'], cwd=R,
               env=dict(env, RA_E2E_PROMPT='1', RA_E2E_INIT_HEADS=str(c / 'best_heads.pt')), stdout=log, stderr=log)
say('done')
