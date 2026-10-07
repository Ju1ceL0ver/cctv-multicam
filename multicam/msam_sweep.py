import json, os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
ck = str(R / 'runs' / 'msam_a' / 'last.pt')
for new, det in (('0.25', '1'), ('0.3', '1'), ('0.35', '1')):
    r = subprocess.run([sys.executable, 'micro_sam.py', 'eval', ck], cwd=R, capture_output=True, text=True,
                       env=dict(os.environ, PYTHONIOENCODING='utf-8', RA_MS_NEW=new, RA_MS_DETONLY=det))
    print('detonly', det, 'new', new, (r.stdout.strip().splitlines() or ['?'])[-1], r.stderr[-400:] if r.returncode else '', flush=True)
