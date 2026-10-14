"""13.10.2026, the owner: stop the live door now, build/check TensorRT (door_live_bench --equivalence-only), then the
speed bench eager vs trt on a busy stretch, then put the live door back (plan restored even on failure)."""
import json, os, shutil, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
P = ROOT / 'data' / 'keeper_plan.json'
SAM3 = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
CK = r'data\quality_backups\doorseg_c_kaggle_final_20261010\step14604_ce9b0a8498d3.pt'
plan = json.load(open(P, encoding='utf-8')); plan['door'] = False
plan['why_door_bench'] = '13.10 owner: live door off for TensorRT checks, door_trt_day.py puts it back'
json.dump(plan, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
import door_live_ctl
door_live_ctl.stop(); time.sleep(15)
env = dict(os.environ, RA_GPU_OWNER_OK='1', PYTHONIOENCODING='utf-8')
try:
    if True:                                       # 13.10: speed only (equivalence looked at by hand)
        r = subprocess.run([SAM3, 'door_live_bench.py', CK, 'door_20260919_32350,door_20260919_27663', 'eager,trt'], cwd=str(ROOT), env=env)
        print('speed exit', r.returncode, flush=True)
        shutil.copy(ROOT / 'data/demo/bench.json', ROOT / 'data/demo/bench_trt.json')
finally:
    plan = json.load(open(P, encoding='utf-8')); plan['door'] = True; plan.pop('why_door_bench', None)
    json.dump(plan, open(P, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('plan restored: door true', time.ctime(), flush=True)
