import os, subprocess
from pathlib import Path
R = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
env = dict(os.environ, PYTHONIOENCODING='utf-8')
subprocess.run([PY, 'staff_masked.py', 'run'], cwd=R, env=env)
subprocess.run([PY, 'staff_masked.py', 'compare'], cwd=R, env=env)
print('all done', flush=True)
