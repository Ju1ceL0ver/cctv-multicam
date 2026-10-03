"""Runs the three benchmarks one after another on a free card (bench_decoder, bench_infer, check_fast with torch.compile)."""
import os
import subprocess
import sys
import time

ck = sys.argv[1] if len(sys.argv) > 1 else 'runs/v2_a/epoch_24.pt'
os.makedirs('data/logs', exist_ok=True)
for name, args in (('bench_decoder', []), ('bench_infer', [ck]), ('check_fast', [ck, '3', 'compile'])):
    t = time.time()
    with open('data/logs/%s.out' % name, 'w') as f:
        subprocess.run([sys.executable, name + '.py'] + args, stdout=f, stderr=subprocess.STDOUT)
    print(name, 'took %.0f s' % (time.time() - t), flush=True)
open('data/logs/suite.done', 'w').write(time.strftime('%H:%M:%S'))
