"""14.10.2026 night queue (owner: fill the night so everything runs by itself). Waits for 21:05 (the live door
stops at 21:00), then in order, each step logged to data/night_1014/<step>.log, summary -> data/night_1014/summary.json:
  1 gpu  doorseg_d step19604 (trained on crowds) on all 43 /doorside stretches -> data/door_v2/dsd_cache
  2 cpu  events F1 of both models, live config and the owner's rules-first
  3 cpu  sweep around rules-first on doorseg_c (tracker continuation, threshold, door window, inheritance, pairs)
  4 cpu  entry <-> exit pairs of the day (door_pairs.py)
Stops before 09:30 whatever is left (the door starts at 09:50)."""
import json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'night_1014'; OUT.mkdir(parents=True, exist_ok=True)
SAM = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
ZL = ROOT / 'zone_lab'
DSC, DSD = ROOT / 'data/door_v2/dsc_cache', ROOT / 'data/door_v2/dsd_cache'
CK_D = ROOT / 'runs/doorseg_d_crowds/step19604.pt'
summary = {'started': time.strftime('%Y-%m-%d %H:%M:%S'), 'steps': {}}


def note(k, v):
    summary['steps'][k] = v
    json.dump(summary, open(OUT / 'summary.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)


def late():
    h = time.strftime('%H:%M')
    return '09:30' <= h < '21:00'


def run(name, cmd, cwd, env=None, timeout=6 * 3600):
    if late():
        note(name, {'skipped': 'morning'}); return False
    t0 = time.time()
    with open(OUT / (name + '.log'), 'w', encoding='utf-8') as f:
        p = subprocess.Popen(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONIOENCODING='utf-8', **(env or {})),
                             creationflags=0x00004000)
        while p.poll() is None:
            if late():
                p.kill(); note(name, {'killed': 'morning', 'sec': round(time.time() - t0)}); return False
            time.sleep(20)
    note(name, {'rc': p.returncode, 'sec': round(time.time() - t0)})
    return p.returncode == 0


def variants():
    live = dict(SM=0., thr=.6, inh_overlap=.3, door_s=2., pair_s=3.)
    V = [{'id': 'live_old', 'tracker': {'max_gap': 25}, 'cfg': live},
         {'id': 'live_rules', 'tracker': {'max_gap': 25}, 'cfg': dict(live, rules_first=1)}]
    return V


def sweep():
    base = dict(SM=0., thr=.6, inh_overlap=.3, door_s=2., pair_s=3., rules_first=1)
    V = []
    for cont in (.3, .4, .5):
        for gap in (15, 25, 40):
            V.append({'id': 'trk_c%.1f_g%d' % (cont, gap), 'tracker': {'max_gap': gap, 'cont': cont}, 'cfg': base})
    for k, vals in (('thr', (.5, .55, .65, .7, .8)), ('door_s', (1., 1.5, 2.5, 3., 4.)), ('inh_overlap', (.1, .2, .4, .5)),
                    ('pair_s', (0, 2., 4., 5.)), ('inh_s', (1., 3.)), ('door_px', (5, 10, 40))):
        for v in vals:
            V.append({'id': '%s_%s' % (k, v), 'tracker': {'max_gap': 25}, 'cfg': dict(base, **{k: v})})
    return V


if __name__ == '__main__':
    while time.strftime('%H:%M') < '21:05' and time.strftime('%H:%M') >= '09:30':
        time.sleep(30)
    summary['go'] = time.strftime('%H:%M:%S'); note('wait', 'done')
    json.dump(variants(), open(OUT / 'variants_cmp.json', 'w'))
    json.dump(sweep(), open(OUT / 'variants_sweep.json', 'w'))
    ok_d = CK_D.exists() and run('1_cache_doorseg_d', [SAM, 'door_seg_cache_all.py'], ROOT,
                                 {'RA_CACHE_CK': str(CK_D), 'RA_CACHE_OUT': str(DSD)})
    for name, cache, rc in (('2_cmp_doorseg_c', DSC, 'rowcache'), ('2_cmp_doorseg_d', DSD, 'rowcache_dsd')):
        if cache == DSD and not ok_d:
            note(name, {'skipped': 'no doorseg_d cache'}); continue
        run(name, [SAM, 'night_sweep.py', str(OUT / 'variants_cmp.json'), str(OUT / (name + '.json'))], ZL,
            {'ZL_CACHE': str(cache), 'ZL_ROWCACHE': rc, 'CUDA_VISIBLE_DEVICES': '-1'})
    run('3_sweep_rules', [SAM, 'night_sweep.py', str(OUT / 'variants_sweep.json'), str(OUT / '3_sweep_rules.json')], ZL,
        {'ZL_CACHE': str(DSC), 'ZL_ROWCACHE': 'rowcache', 'CUDA_VISIBLE_DEVICES': '-1'})
    run('4_pairs', [RF, 'door_pairs.py', time.strftime('%Y%m%d', time.localtime(time.time() - 6 * 3600))], ROOT,
        {'CUDA_VISIBLE_DEVICES': '-1'})
    summary['finished'] = time.strftime('%Y-%m-%d %H:%M:%S'); note('end', 'done')
