"""07.10: the live test on recorded stretches (card), then the detector goes on (s31e2e_f from s31e2e_e's best) until
20:00, then the night tracker queue (night_trk_0710.py waits for it by itself)."""
import os, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
env = dict(os.environ, PYTHONIOENCODING='utf-8')
subprocess.run([sys.executable, 'live_test_0710.py', '4'], cwd=R, env=env,
               stdout=open(R / 'data' / 'logs' / 'livetest.log', 'w'), stderr=subprocess.STDOUT)
e = R / 'runs' / 's31e2e_e'
src = e if (e / 'best_heads.pt').exists() else R / 'runs' / 's31e2e_d_door'
det = subprocess.Popen([PY, 'sam31_e2e_det.py', 's31e2e_f', str(src / 'best_last.pt'), '20:00'], cwd=R,
                       env=dict(env, RA_E2E_PROMPT='1', RA_E2E_INIT_HEADS=str(src / 'best_heads.pt')),
                       stdout=open(R / 'data' / 'logs' / 'e2edet_f.log', 'w'), stderr=subprocess.STDOUT)
time.sleep(120)                       # the night queue waits for a running detector, so start it after the detector
subprocess.Popen([sys.executable, 'night_trk_0710.py'], cwd=R, env=env)
det.wait()
