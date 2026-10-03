"""After the clean-data run (runs/v2_l_clean2) ends: the same model learns on the SAM 3.1 clips too (tracks, ReID, memory -- everything but
det-only), the advertising stand taken out of the clips' targets (RA_V2_NOPOSTER), the cleaned single frames and the owner's frames mixed in.
It starts from the clean run's best weights when they beat the owner test of live_r07 (F1 0.92), else from live_r07.
usage: chain_clips.py   (run through bg.spawn; log data/logs/chainclips.log)"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import bg

PREV, OUT, BASE = 'runs/v2_l_clean2', 'runs/v2_m_clips', 'runs/live_r07/best.pt'
BASELINE_F1 = 0.92


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def best_f1(run):
    best = 0.0
    p = ROOT / run / 'epochs.jsonl'
    if p.exists():
        for l in p.read_text(encoding='utf-8').splitlines():
            try:
                best = max(best, json.loads(l)['test']['f1'])
            except (ValueError, KeyError, TypeError):
                pass
    return best


def main():
    while bg.running('v2_l_clean2'):
        time.sleep(30)
    f1 = best_f1(PREV)
    init = PREV + '/best.pt' if f1 > BASELINE_F1 and (ROOT / PREV / 'best.pt').exists() else BASE
    log('clean run ended, best owner-test F1 %.4f -> init %s' % (f1, init))
    tags = json.load(open(ROOT / 'runs' / 'v2_b' / 'args.json'))['tags']
    args = ['train_v2.py', '--out', OUT, '--init', init, '--tags'] + tags + [
        '--heldout', '20260923_21601', '20260923_25201', '20260923_01801', '--test', 'fix', '--steps', '4000', '--epoch-steps', '300',
        '--lr', '0.0001', '--freeze-steps', '0', '--freeze-vit-only', '--backbone', 'repvit_m0_9.dist_450e_in1k', '--decoder-layers', '6',
        '--clips', '1', '--T', '4', '--far', '0.3', '--ema', '0.9995', '--vit-lr-mult', '0.26667', '--drafts', '0.3', '--draft-items', '3',
        '--draft-only-steps', '0', '--workers', '3', '--draft-workers', '2', '--far-workers', '1', '--track-every', '3',
        '--owner-exam-every', '6', '--keep-every', '5', '--lr-floor', '0.05']
    env = dict(os.environ, RA_V2_DRAFTS='sam31clean', RA_V2_CLEAN_DIR='sam31_stills_clean2', RA_V2_GOLD='0.15', RA_V2_NOPOSTER='1')
    r = subprocess.run([sys.executable] + args, cwd=str(ROOT), env=env)
    log('clip run ended, rc', r.returncode, 'best F1 %.4f' % best_f1(OUT))


if __name__ == '__main__':
    main()
