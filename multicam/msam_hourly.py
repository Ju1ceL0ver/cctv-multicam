"""07.10 night: every hour the latest SAM-micro weights against SAM 3.1 on 23.09 -- detector alone and with the
tracker, new person at 0.25 -> data/logs/msam_hourly.log (until 09:40)."""
import os, shutil, subprocess, sys, time
from pathlib import Path
R = Path(__file__).resolve().parent
log = open(R / 'data' / 'logs' / 'msam_hourly.log', 'a', encoding='utf-8')
while time.strftime('%H:%M') < '09:40' or time.strftime('%H:%M') > '20:00':
    ck = R / 'runs' / 'msam_c' / 'last.pt'
    if ck.exists():
        snap = R / 'runs' / 'msam_c' / 'hourly.pt'
        shutil.copy(ck, snap)
        for det in ('1', '0'):
            r = subprocess.run([sys.executable, 'micro_sam.py', 'eval', str(snap)], cwd=R, capture_output=True, text=True,
                               env=dict(os.environ, PYTHONIOENCODING='utf-8', RA_MS_NEW='0.25', RA_MS_DETONLY=det))
            log.write('%s %s %s\n' % (time.strftime('%m-%d %H:%M'), 'detector' if det == '1' else 'det+tracker',
                                      (r.stdout.strip().splitlines() or [r.stderr[-300:]])[-1])); log.flush()
    time.sleep(3600)
