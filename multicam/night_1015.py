"""15.10 night: the SAM 3.1 teacher on the owner's labelled crowd frames, then the comparison (crowd_compare.py).
Waits for 21:05; log data/night_1015/crowd_compare.log."""
import os, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'night_1015'; OUT.mkdir(parents=True, exist_ok=True)
while not (time.strftime('%H:%M') >= '21:05' or time.strftime('%H:%M') < '09:30'):
    time.sleep(30)
with open(OUT / 'crowd_compare.log', 'w', encoding='utf-8') as f:
    subprocess.run([r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe', 'crowd_compare.py', 'teacher'], cwd=str(ROOT),
                   stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
