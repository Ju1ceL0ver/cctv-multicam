import os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
env = dict(os.environ, PYTHONIOENCODING='utf-8')
subprocess.run([r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe', 'staff_counter.py'], cwd=R, env=env)
subprocess.run([sys.executable, 'staff.py', 'build'], cwd=R, env=env)
print('built', flush=True)
