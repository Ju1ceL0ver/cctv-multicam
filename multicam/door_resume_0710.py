"""07.10: the door of 19.09 with the new small SAM goes on (finished stretches are skipped), the comparison, then the
detector goes on (s31e2e_e) until 20:00. Log: data/logs/doornew.log."""
import os, subprocess, time
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
log = open(R / 'data' / 'logs' / 'doornew.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
env = dict(os.environ, PYTHONIOENCODING='utf-8')
snap = R / 'runs' / 's31e2e_d_door'
say('door 19.09 goes on')
subprocess.run([PY, 'door_micro.py', str(snap / 'best_last.pt'), 'e2e_d', '20260919', '--stride', '3', '--heads', str(snap / 'best_heads.pt')],
               cwd=R, env=dict(env, RA_S31_NEWDET='0.5'), stdout=log, stderr=log)
say('compare')
subprocess.run([r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\cctv_base\python.exe', 'door_cmp_day.py', '20260919', 'micro1s3',
                '20260917,20260918', 'micro1s3', 'e2e_d', 'sam31'], cwd=R, env=env, stdout=log, stderr=log)
say('detector goes on')
subprocess.run([PY, 'sam31_e2e_det.py', 's31e2e_e', str(snap / 'best_last.pt'), '20:00'], cwd=R,
               env=dict(env, RA_E2E_PROMPT='1', RA_E2E_INIT_HEADS=str(snap / 'best_heads.pt')), stdout=log, stderr=log)
say('done')
