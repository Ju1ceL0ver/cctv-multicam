"""07.10 evening: the live test again (the record's last short session now counted), then the detector goes on through
the night (s31e2e_f from s31e2e_e's best, until 09:30). Log: data/logs/evening.log"""
import os, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
env = dict(os.environ, PYTHONIOENCODING='utf-8')
log = open(R / 'data' / 'logs' / 'evening.log', 'a', encoding='utf-8')
say = lambda m: (log.write(time.strftime('%m-%d %H:%M:%S ') + m + '\n'), log.flush())
say('live test')
subprocess.run([sys.executable, 'live_test_0710.py', '4'], cwd=R, env=env,
               stdout=open(R / 'data' / 'logs' / 'livetest2.log', 'w'), stderr=subprocess.STDOUT)
if (R / 'sam_micro_grid_bench.py').exists():
    say('speed bench')
    subprocess.run([PY, 'sam_micro_grid_bench.py'], cwd=R, env=env,
                   stdout=open(R / 'data' / 'logs' / 'gridbench.log', 'w'), stderr=subprocess.STDOUT)
e = R / 'runs' / 's31e2e_e'
say('detector through the night')
subprocess.run([PY, 'sam31_e2e_det.py', 's31e2e_f', str(e / 'best_last.pt'), '09:30'], cwd=R,
               env=dict(env, RA_E2E_PROMPT='1', RA_E2E_INIT_HEADS=str(e / 'best_heads.pt')),
               stdout=open(R / 'data' / 'logs' / 'e2edet_f.log', 'w'), stderr=subprocess.STDOUT)
say('done')
