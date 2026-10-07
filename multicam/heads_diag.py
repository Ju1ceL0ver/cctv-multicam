"""07.10.2026: which short stack breaks the small SAM -- the 23.09 check with one stack installed at a time."""
import json, os, subprocess, sys
from pathlib import Path
R = Path(__file__).resolve().parent
py = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
heads = R / 'runs' / 's31heads_a' / 'heads.pt'
for only in sys.argv[1:] or ['det_enc', 'det_dec', 'trk']:
    out = R / 'runs' / 's31heads_a' / 'evals' / ('diag_%s.json' % only)
    r = subprocess.run([py, 'sam31_lite_eval.py', str(R / 'runs' / 's31micro_door' / 'last.pt'), '2', '96',
                        '--cache', str(R / 'runs' / 's31micro_a' / 'evals' / 'teacher_cache'), '--out', str(out),
                        '--preview', str(out.with_suffix('.jpg')), '--film', 'none', '--heads', str(heads)],
                       cwd=R, env=dict(os.environ, RA_HEADS_ONLY=only, PYTHONIOENCODING='utf-8'), capture_output=True, text=True)
    m = json.load(open(out))['mean'] if out.exists() else {'error': (r.stderr or r.stdout)[-400:]}
    print(only, json.dumps(m), flush=True)
