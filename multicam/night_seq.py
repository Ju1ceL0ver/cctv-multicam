"""One job at a time on the card until the morning (the owner's plan of 01.10, 06:00):

  A  the find-only slot model on SAM 3.1 only: its windows + its single-frame drafts (runs/v2_i_sam31)
  B  a soft fine-tune of A's best on the owner's corrected frames only (/fix, taken): his labels leave out prams and
     people in the other shops, which SAM outlines, so the two are not mixed (runs/v2_j_owner)
  Y  YOLO26n (already trained on the SAM 3.1 windows) fine-tuned softly on his frames only (runs/v2_y_yolo_n_owner)

Every score is on his test (/fix, 23.09 taken frames). A stage ends when it levels off (at least MIN_EPOCHS tested
epochs, the mean F1 of the last 3 no better than the best before the last 6 by GAIN), at its step count, or at its
deadline. Log: data/logs/night_seq.log."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MIN_EPOCHS, GAIN = 6, 0.003
HARD = '09:35'
COMMON = ['--heldout', '20260923_21601', '20260923_25201', '20260923_01801', '--test', 'fix', '--track-every', '1000',
          '--owner-exam-every', '4', '--keep-every', '4', '--workers', '3', '--far-workers', '1', '--draft-workers', '2',
          '--far', '0', '--T', '1', '--clips', '3', '--ema', '0.9995', '--vit-lr-mult', '0.26667', '--draft-items', '3',
          '--draft-only-steps', '0', '--freeze-steps', '0', '--freeze-vit-only', '--backbone', 'repvit_m0_9.dist_450e_in1k',
          '--decoder-layers', '6', '--det-only']
A = ('v2_i_sam31', ['train_v2.py', '--out', 'runs/v2_i_sam31', '--init', 'runs/v2_f_detonly/best.pt', '--steps', '1500',
                    '--epoch-steps', '300', '--lr', '0.0001', '--lr-floor', '0.05', '--drafts', '0.5', '--until', '08:10'] + COMMON,
     {'RA_V2_GOLD': '0', 'RA_V2_DRAFTS': 'sam31'})
B = ('v2_j_owner', ['train_v2.py', '--out', 'runs/v2_j_owner', '--init', 'runs/v2_i_sam31/best.pt', '--steps', '900',
                    '--epoch-steps', '150', '--lr', '0.00002', '--lr-floor', '0.1', '--drafts', '1.0', '--until', '09:05'] + COMMON,
     {'RA_V2_GOLD': '1', 'RA_V2_DRAFTS': 'sam31'})
Y = ('v2_y_yolo_n_owner', ['yolo_baseline.py', '--out', 'runs/v2_y_yolo_n_owner', '--model', 'runs/v2_y_yolo_n/best.pt',
                           '--data', 'data/seg_datasets/fix_yolo/data_owner.yaml', '--test', 'fix', '--epochs', '20',
                           '--imgsz', '1088', '--batch', '8', '--lr0', '0.0005', '--warmup', '0'], {})
YOLO_STOP = '09:10'


def log(*a):
    with open(ROOT / 'data' / 'logs' / 'night_seq.log', 'a', encoding='utf-8') as f:
        f.write(time.strftime('%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a) + '\n')


def alive(run):
    import bg
    return [l for l in bg.running('runs/' + run) if 'train_v2.py' in l or 'yolo_baseline.py' in l or 'unetpp.py' in l]


def f1s(run):
    p = ROOT / 'runs' / run / 'epochs.jsonl'
    out = []
    if p.exists():
        for line in open(p):
            t = json.loads(line).get('test') or {}
            if 'f1' in t:
                out.append(t['f1'])
    return out


def level(f):
    return len(f) > max(6, MIN_EPOCHS - 1) and sum(f[-3:]) / 3 <= max(f[:-6]) + GAIN


def run(stage, plateau=True, stop_at=None):
    import bg
    name, args, env = stage
    if not alive(name):
        d = ROOT / 'runs' / name
        if d.exists() and (d / 'epochs.jsonl').exists():
            d.rename(d.with_name(name + '_old_%s' % time.strftime('%H%M')))
        bg.spawn(name, args, env=env, wait=30)
        log('start', name)
    sent = False
    while alive(name):
        f = f1s(name)
        now = time.strftime('%H:%M')
        if not sent and ((plateau and level(f)) or (stop_at and now >= stop_at)):
            (ROOT / 'runs' / name / 'STOP').write_text('level' if plateau and level(f) else 'morning')
            sent = True
            log(name, 'STOP, F1', f)
        if now >= HARD:
            for line in alive(name):
                subprocess.run(['taskkill', '/PID', line.split('|')[0], '/F', '/T'], capture_output=True)
            log(name, 'killed at', now)
            break
        time.sleep(60)
    log(name, 'done, F1', f1s(name))


def owner_yaml():
    base = ROOT / 'data' / 'seg_datasets'
    (base / 'fix_yolo' / 'data_owner.yaml').write_text('train: %s\nval: %s\nnames:\n  0: person\n' % (
        (base / 'fix_yolo' / 'images' / 'train').as_posix(), (base / 'sam31_yolo' / 'images' / 'val').as_posix()))


def main():
    log('up: A -> B -> YOLO')
    run(A)
    if not (ROOT / 'runs' / A[0] / 'best.pt').exists():
        log('A has no best.pt: B starts from the find-only best')
        B[1][B[1].index('--init') + 1] = 'runs/v2_f_detonly/best.pt'
    if time.strftime('%H:%M') < '08:50':
        run(B, plateau=False)
    if time.strftime('%H:%M') < '09:00':
        owner_yaml()
        run(Y, plateau=False, stop_at=YOLO_STOP)
    log('night done')


if __name__ == '__main__':
    main()
