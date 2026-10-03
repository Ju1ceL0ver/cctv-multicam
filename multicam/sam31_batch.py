"""SAM 3.1 + seams + ReID on several busy moments of other days and both cameras, one after another
(sam31_video.py in venv_sam3, then sam31_reid.py in venv_rfdetr), to see whether the ReID threshold picked
on 18.09 cam1 15:17 holds where it was not picked. The moments: per day and camera the one with the most
people from the draft selection (data/seg_datasets/pseudo_20260925_more/select).

usage: sam31_batch.py [TAU]   -> data/logs/sam31/batch.json"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
S3 = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
PICK = [('20260917', 'cam2'), ('20260919', 'cam1'), ('20260919', 'cam2'), ('20260920', 'cam1'),
        ('20260921', 'cam2'), ('20260922', 'cam1'), ('20260923', 'cam2')]


def moments():
    out = [('20260918_cam1_1517', ['film', '20260918', 'cam1', '19059.34', '15']),
           ('20260918_cam1_1517_s1008', ['film', '20260918', 'cam1', '19059.34', '15'])]   # the same at 1008, to compare
    sel = ROOT / 'data' / 'seg_datasets' / 'pseudo_20260925_more' / 'select'
    for day, cam in PICK:
        p = sel / ('%s_%s.json' % (day, cam))
        if not p.exists():
            continue
        best = max(json.load(open(p)), key=lambda it: (it['n'], it.get('hard', 0)))
        if not Path(best['path']).exists():
            continue
        out.append(('%s_%s_%s_%d' % (day, cam, Path(best['path']).stem, best['second']),
                    ['raw', best['path'], str(best['second']), '15']))
    return out


def main():
    tau = sys.argv[1] if len(sys.argv) > 1 else '0.35'
    log = ROOT / 'data' / 'logs' / 'sam31' / 'batch.json'
    log.parent.mkdir(parents=True, exist_ok=True)
    rep = []
    import bg
    for tag, args in moments():
        t0 = time.time()
        while bg.running('sam31_video.py'):             # a run left over from a restart finishes first
            time.sleep(20)
        if (ROOT / 'data' / 'logs' / 'sam31' / tag / 'info.json').exists():
            a = subprocess.CompletedProcess([], 0)        # its chunks are already there
        else:
            a = subprocess.run([S3, 'sam31_video.py', tag] + args, cwd=str(ROOT))
        b = subprocess.run([RF, 'sam31_reid.py', tag, tau], cwd=str(ROOT)) if a.returncode == 0 else None
        r = {'tag': tag, 'sam': a.returncode, 'reid': b.returncode if b else None, 'seconds': round(time.time() - t0)}
        rp = ROOT / 'data' / 'logs' / 'sam31' / tag / 'report.json'
        if b is not None and b.returncode == 0 and rp.exists():
            x = json.load(open(rp))
            r.update(sam_tracks=x['sam_tracks'], pieces=x['pieces'], people=x['people'],
                     seams_linked=sum(s['linked'] for s in x['seams']))
        rep.append(r)
        json.dump(rep, open(log, 'w'), indent=1)
        print(r, flush=True)
    json.dump(rep + [{'done': True}], open(log, 'w'), indent=1)


if __name__ == '__main__':
    main()
