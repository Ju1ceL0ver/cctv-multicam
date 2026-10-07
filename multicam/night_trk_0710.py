"""07.10 night: after the detector (s31e2e_e) ends -> a 3-step smoke of the slimmed clip tracker; if it ran, the tracker
learns on clips until 09:30 (run s31trk_a) on the night's detector. Log: data/logs/nighttrk.log."""
import os, subprocess, time
from pathlib import Path
import psutil
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
log = open(R / 'data' / 'logs' / 'nighttrk.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
busy = lambda: any(('sam31_e2e_det.py' in ' '.join(p.info['cmdline'] or []) or 'door_micro.py' in ' '.join(p.info['cmdline'] or []))
                   for p in psutil.process_iter(['cmdline']))
while busy():
    time.sleep(60)
env = dict(os.environ, PYTHONIOENCODING='utf-8')
det = 's31e2e_e' if (R / 'runs' / 's31e2e_e' / 'best_heads.pt').exists() else 's31e2e_b'
say('smoke on ' + det)
r = subprocess.run([PY, 'sam31_e2e_trk.py', 's31trk_smoke2', det], cwd=R, env=dict(env, RA_TRK_SMOKE='3'), capture_output=True, text=True)
log.write(r.stdout[-1500:] + r.stderr[-1500:] + '\n'); log.flush()
if 'smoke step 3' not in r.stdout:
    say('smoke failed, no night training')
else:
    say('tracker training until 09:30')
    subprocess.run([PY, 'sam31_e2e_trk.py', 's31trk_a', det, '09:30'], cwd=R, env=env, stdout=log, stderr=log)
say('done')
