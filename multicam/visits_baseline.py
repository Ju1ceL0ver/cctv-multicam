"""Baseline: existing fusion2 association on the owner-labelled clips, one log."""
import subprocess, sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))
CLIPS = ['c100054', 'c101054', 'c103700', 'c112000', 'c155236', 'c171400', 'c183400']
os.makedirs(os.path.join(ROOT, 'data', 'visits_lab'), exist_ok=True)
with open(os.path.join(ROOT, 'data', 'visits_lab', 'baseline_runclip.log'), 'w', encoding='utf-8') as f:
    for c in CLIPS:
        r = subprocess.run([sys.executable, 'run_clip.py', c], cwd=ROOT, capture_output=True, text=True,
                           encoding='utf-8', errors='replace', env=dict(os.environ, PYTHONIOENCODING='utf-8'))
        f.write('=== %s rc=%d\n%s\n%s\n' % (c, r.returncode, r.stdout, r.stderr[-3000:])); f.flush()
