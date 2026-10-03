"""The night queue of SAM 3.1 windows: per day the busiest 15-minute file of camera 1 (the most people by the
draft selection, data/seg_datasets/pseudo_20260925_more/select), both cameras on the same clock
(sam31_segment.py in venv_sam3), then the sessions linked and pieces joined by ReID (sam31_reid.py in
venv_rfdetr). Never 18.09 (the exam day). Stops starting new work at DEADLINE.

usage: sam31_night.py [DEADLINE HH:MM] [DAY ...]   -> data/sam31_seg/queue.json"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
S3 = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
DAYS = ['20260919', '20260922', '20260917', '20260920', '20260921', '20260923']
SEL = ROOT / 'data' / 'seg_datasets' / 'pseudo_20260925_more' / 'select'


def window(day, rank=0):
    """Film time of the start of camera 1's busiest 15-minute file of the day (rank 1 -- the next busiest ...)."""
    import day_movie
    import day_player
    items = json.load(open(SEL / ('%s_cam1.json' % day)))
    score = {}
    for it in items:
        name = os.path.basename(it['path'])
        score[name] = score.get(name, 0) + it['n'] + it.get('hard', 0)
    start, _ = day_movie.clock(day, str(ROOT))
    segs = {s.name: s for s in day_player.segments(day, str(ROOT))['cam1'] if s.ready and not s.broken}
    for name in [n for n in sorted(score, key=lambda k: -score[k]) if n in segs][rank:rank + 1]:
        s = segs[name]
        return round(s.start + float(s.times()[0]) - start + 0.5, 2), name
    return None, None


def main():
    deadline = sys.argv[1] if len(sys.argv) > 1 else '09:40'
    days = sys.argv[2:] or DAYS
    log = ROOT / 'data' / 'sam31_seg' / 'queue.json'
    log.parent.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(log)) if log.exists() else []
    for day in days:
        if time.strftime('%H:%M') >= deadline:
            break
        t0, name = window(day)
        if t0 is None:
            rep.append({'day': day, 'skipped': 'no window'}); continue
        tag = '%s_%05d' % (day, int(t0))
        if any(r.get('tag') == tag and r.get('done') for r in rep):
            continue
        t = time.time()
        a = subprocess.run([S3, 'sam31_segment.py', day, str(t0), '900', deadline], cwd=str(ROOT))
        r = {'day': day, 'tag': tag, 'file': name, 'film_start': t0, 'segment_rc': a.returncode}
        for cam in ('cam1', 'cam2'):
            info = ROOT / 'data' / 'sam31_seg' / tag / cam / 'info.json'
            if not info.exists():
                continue
            i = json.load(open(info))
            b = subprocess.run([RF, 'sam31_reid.py', '%s/%s' % (tag, cam), '0.35'], cwd=str(ROOT), capture_output=True, text=True,
                               env=dict(os.environ, RA_S31_ROOT=str(ROOT / 'data' / 'sam31_seg'), RA_S31_VIDEO='0'))
            rj = ROOT / 'data' / 'sam31_seg' / tag / cam / 'report.json'
            x = json.load(open(rj)) if b.returncode == 0 and rj.exists() else {}
            r[cam] = {'live_ticks': i['live_ticks'], 'ticks': i['ticks'], 'partial': i['partial'], 'sam_s': i['sam_s'],
                      's_per_live_tick': i['s_per_live_tick'], 'pieces': x.get('pieces'), 'people': x.get('people'),
                      'reid_rc': b.returncode, 'reid_err': None if b.returncode == 0 else b.stderr[-800:]}
        r['done'] = all(r.get(c) and not r[c]['partial'] for c in ('cam1', 'cam2'))
        r['seconds'] = round(time.time() - t)
        rep.append(r)
        json.dump(rep, open(log, 'w'), indent=1)
        print(r, flush=True)


if __name__ == '__main__':
    main()
