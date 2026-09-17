"""Run a shell-less command detached, stdout+stderr to a log; survives tunnel drops.
usage (from jctl): bg.py LOGNAME script.py args..."""
import sys, os, subprocess
ROOT = os.path.dirname(os.path.abspath(__file__))
PY = r'C:\Users\ArykovAA\AppData\Local\miniconda3\envs\cctv_base\python.exe'
log = os.path.join(ROOT, 'data', 'logs', sys.argv[1] + '.log')
os.makedirs(os.path.dirname(log), exist_ok=True)
p = subprocess.Popen([PY] + sys.argv[2:], cwd=ROOT, stdout=open(log, 'w'), stderr=subprocess.STDOUT,
                     creationflags=0x00000200 | 0x00000008 | 0x01000000)
print('started', p.pid, '->', log)
