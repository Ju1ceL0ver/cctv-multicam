"""07.10.2026: wait for io_fast.py to end, then resume the door distillation from its last.pt (same run, same stop)."""
import os, subprocess, time
from pathlib import Path
import psutil
R = Path(__file__).resolve().parent
while any('io_fast.py' in ' '.join(p.info['cmdline'] or []) for p in psutil.process_iter(['cmdline'])):
    time.sleep(30)
log = open(R / 'data' / 'logs' / 'chain0710.log', 'a', encoding='utf-8')
log.write(time.strftime('%m-%d %H:%M:%S ') + 'io_fast finished, resume sam31_distill\n'); log.flush()
subprocess.run([r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe', 'sam31_distill.py', 's31micro_door', '03:00',
                '--init', str(R / 'runs' / 's31micro_door' / 'last.pt'), '--student', 'mobilenetv4_conv_medium'],
               cwd=R, env=dict(os.environ, PYTHONIOENCODING='utf-8', RA_DISTILL_KINDS='sam31_door', RA_DISTILL_SKIP='20260918'),
               stdout=log, stderr=log)
log.write(time.strftime('%m-%d %H:%M:%S ') + 'distill end, start the short heads\n'); log.flush()
subprocess.run([r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe', 'sam31_heads.py', 's31heads_a',
                str(R / 'runs' / 's31micro_door' / 'last.pt'), '08:30'],
               cwd=R, env=dict(os.environ, PYTHONIOENCODING='utf-8'), stdout=log, stderr=log)
log.write(time.strftime('%m-%d %H:%M:%S ') + 'heads end\n')
