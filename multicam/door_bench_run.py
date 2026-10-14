"""13.10.2026, the owner: stop the live door for the benchmark, run door_live_bench.py, put the live door back
(keeper_plan door: true; the keeper restarts it). The plan is restored even if the benchmark fails."""
import json, shutil, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
P = ROOT / 'data' / 'keeper_plan.json'
SAM3 = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
VC = r'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
CK = r'data\quality_backups\doorseg_c_kaggle_final_20261010\step14604_ce9b0a8498d3.pt'
shutil.copy(P, ROOT / 'data' / 'demo' / 'keeper_plan_before_bench.json')
plan = json.load(open(P, encoding='utf-8'))
plan['door'] = False
plan['why_door_bench'] = '13.10 owner: live door off for the speed benchmark, door_bench_run.py puts it back'
json.dump(plan, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import door_live_ctl
door_live_ctl.stop()
time.sleep(15)
try:
    for tags, vs in (('door_20260919_27663', 'eager,compile,trt'), ('door_20260919_32350', 'eager,trt')):
        cmd = 'call "%s" >nul && "%s" door_live_bench.py %s %s %s' % (VC, SAM3, CK, tags, vs)
        r = subprocess.run(cmd, cwd=str(ROOT), shell=True)
        print('bench', tags, 'exit', r.returncode, flush=True)
        shutil.copy(ROOT / 'data/demo/bench.json', ROOT / 'data/demo' / ('bench_%s.json' % tags))
finally:
    plan = json.load(open(P, encoding='utf-8'))
    plan['door'] = True
    plan.pop('why_door_bench', None)
    json.dump(plan, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('plan restored: door true', time.ctime(), flush=True)
