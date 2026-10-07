"""07.10: as soon as s31e2e_d has a best -> stop it, the door of 19.09 with the new small SAM (threshold 0.5),
the fair comparison with micro1s3, then s31e2e_d goes on (run s31e2e_e) until 20:00. Log: data/logs/doornew.log."""
import os, subprocess, time
from pathlib import Path
import psutil
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
log = open(R / 'data' / 'logs' / 'doornew.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
env = dict(os.environ, PYTHONIOENCODING='utf-8')
d = R / 'runs' / 's31e2e_d'
while not (d / 'best_heads.pt').exists() or time.time() - (d / 'best_heads.pt').stat().st_mtime > 3600:
    time.sleep(30)
time.sleep(int(os.environ.get('RA_WAIT_S', '0')))
for p in psutil.process_iter(['pid', 'cmdline']):
    c = ' '.join(p.info['cmdline'] or [])
    if 'sam31_e2e_det.py' in c and 'venv_sam3' in c or 'smoke_then_det_0710' in c:
        subprocess.run(['taskkill', '/PID', str(p.info['pid']), '/F', '/T'], capture_output=True)
        say('stopped %d %s' % (p.info['pid'], c[-60:]))
time.sleep(10)
import shutil
snap = R / 'runs' / 's31e2e_d_door'
snap.mkdir(exist_ok=True)
for f in ('best_last.pt', 'best_heads.pt'):
    shutil.copy(d / f, snap / f)
say('door 19.09 with the new small SAM')
subprocess.run([PY, 'door_micro.py', str(snap / 'best_last.pt'), 'e2e_d', '20260919', '--stride', '3', '--heads', str(snap / 'best_heads.pt')],
               cwd=R, env=dict(env, RA_S31_NEWDET='0.5'), stdout=log, stderr=log)
say('compare')
subprocess.run([r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\cctv_base\python.exe', 'door_cmp_day.py', '20260919', 'micro1s3',
                '20260917,20260918', 'micro1s3', 'e2e_d', 'sam31'], cwd=R, env=env, stdout=log, stderr=log)
say('detector goes on')
subprocess.run([PY, 'sam31_e2e_det.py', 's31e2e_e', str(snap / 'best_last.pt'), '20:00'], cwd=R,
               env=dict(env, RA_E2E_PROMPT='1', RA_E2E_INIT_HEADS=str(snap / 'best_heads.pt')), stdout=log, stderr=log)
say('done')
