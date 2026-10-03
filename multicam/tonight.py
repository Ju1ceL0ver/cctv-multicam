"""27.09 evening and night: the data for the tracking teacher and the model, one GPU job after another.

  1. wait for the /xcam questions of 19-23.09 (xcam.py model teacher) to finish
  2. all the data, two lanes side by side:
     A  track_emb.py + stitch.py --looks teachers for 17-23.09   (whole visits; 17.09 scored on the owner's people,
        the other days joined with 17.09's settings)
     B  clips.py build 150, then 300 more                         (short clips, every person numbered)
  3. then the training: the slot model fine-tuned on the owner's /paint masks (+ its held-out score); crop ReID,
     the full recipe and the teachers-copy-only one, 20 epochs each, each scored on the /door pairs
A step that fails is written down and the next one goes on. Log: data/logs/tonight.json."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
LOG = ROOT / 'data' / 'logs' / 'tonight.json'


def running(what):
    out = subprocess.run(['powershell', '-c', 'Get-CimInstance Win32_Process -Filter "name=\'python.exe\'" | %{ $_.CommandLine }'],
                         capture_output=True, text=True).stdout
    return any(what in l and 'tonight.py' not in l for l in out.splitlines())


def main():
    rep = {'started': time.strftime('%Y-%m-%dT%H:%M:%S'), 'steps': []}
    save = lambda: json.dump(rep, open(LOG, 'w'), indent=1)
    save()
    while running('xcam.py model'):
        time.sleep(30)
    days = ('20260917', '20260918', '20260919', '20260920', '20260921', '20260922', '20260923')
    lane_a = [x for d in days for x in (['track_emb.py', d], ['stitch.py', d, '--looks', 'teachers'])]
    lane_b = [['clips.py', 'build', '150'], ['clips.py', 'build', '300']]
    train = [['train_slots.py', 'runs/slots_v1d_paint', '--init', 'runs/slots_v1c_dn/last.pt', '--paint', '--epochs', '20',
              '--batch', '8', '--accum', '1', '--lr', '5e-5', '--warmup', '100', '--probe-every', '0', '--workers', '12', '--ignore-day'],
             ['eval_slots.py', 'runs/slots_v1d_paint', '--once'],
             ['crop_reid.py', 'train', 'runs/crop_reid/v3_full.pt', '--epochs', '20'],
             ['crop_reid.py', 'door', 'runs/crop_reid/v3_full.pt', 'runs/slots_v1c_dn/last.pt'],
             ['crop_reid.py', 'train', 'runs/crop_reid/v3_copy.pt', '--epochs', '20', '--rank', '0', '--identity', '0'],
             ['crop_reid.py', 'door', 'runs/crop_reid/v3_copy.pt', 'runs/slots_v1c_dn/last.pt']]
    import threading
    lock = threading.Lock()

    def lane(steps, tag):
        for s in steps:
            t0 = time.time()
            name = '_'.join(x.replace('.py', '').replace('/', '-') for x in s[:3] if not x.startswith('--'))
            rec = {'lane': tag, 'step': ' '.join(s), 'at': time.strftime('%H:%M:%S')}
            with lock:
                rep['steps'].append(rec); save()
            with open(ROOT / 'data' / 'logs' / ('tonight_%s.log' % name), 'w') as out:
                r = subprocess.run([PY] + s, cwd=str(ROOT), stdout=out, stderr=subprocess.STDOUT)
            with lock:
                rec.update(code=r.returncode, minutes=round((time.time() - t0) / 60, 1)); save()
    # all the data first, in two lanes side by side (reading video and the CPU / the card), then the training
    ta = threading.Thread(target=lane, args=(lane_a, 'A: teachers + visits')); tb = threading.Thread(target=lane, args=(lane_b, 'B: clips'))
    ta.start(); tb.start(); ta.join(); tb.join()
    lane(train, 'C: training')
    rep['finished'] = time.strftime('%Y-%m-%dT%H:%M:%S'); save()


if __name__ == '__main__':
    main()
