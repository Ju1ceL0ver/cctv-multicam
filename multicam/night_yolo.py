"""YOLO26n the way the slot model went on 01.10: A -- the SAM 3.1 windows plus SAM 3.1's single-frame labels of the
varied frames (from the SAM-windows YOLO), B -- a soft fine-tune on the owner's corrected frames only (backbone frozen,
no mosaic, a small rate). Scores on his test (/fix). Stops before the live counter needs the card.
Log: data/logs/night_seq.log."""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import night_seq as NS

BASE = ROOT / 'data' / 'seg_datasets'
YA = ('v2_y_yolo_n_sam31', ['yolo_baseline.py', '--out', 'runs/v2_y_yolo_n_sam31', '--model', 'runs/v2_y_yolo_n/best.pt',
                            '--data', 'data/seg_datasets/sam31_stills_yolo/data_both.yaml', '--test', 'fix', '--epochs', '3',
                            '--imgsz', '1088', '--batch', '8', '--lr0', '0.001', '--warmup', '0'], {})
YB = ('v2_y_yolo_n_owner2', ['yolo_baseline.py', '--out', 'runs/v2_y_yolo_n_owner2', '--model', 'runs/v2_y_yolo_n_sam31/best.pt',
                             '--data', 'data/seg_datasets/fix_yolo/data_owner.yaml', '--test', 'fix', '--epochs', '12',
                             '--imgsz', '1088', '--batch', '8', '--lr0', '0.0001', '--warmup', '0',
                             '--extra', 'freeze=10', 'mosaic=0', 'close_mosaic=0'], {})


def main():
    import subprocess
    NS.log('yolo up: A (SAM 3.1) -> B (owner)')
    r = subprocess.run([sys.executable, 'sam31_stills_yolo.py'], cwd=str(ROOT), capture_output=True, text=True)
    NS.log('export', r.stdout.strip()[-200:], r.stderr.strip()[-300:] if r.returncode else '')
    (BASE / 'sam31_stills_yolo' / 'data_both.yaml').write_text('train:\n  - %s\n  - %s\nval: %s\nnames:\n  0: person\n' % (
        (BASE / 'sam31_yolo' / 'images' / 'train').as_posix(), (BASE / 'sam31_stills_yolo' / 'images' / 'train').as_posix(),
        (BASE / 'sam31_yolo' / 'images' / 'val').as_posix()))
    NS.run(YA, plateau=False, stop_at='08:20')
    if not (ROOT / 'runs' / YA[0] / 'best.pt').exists():
        YB[1][YB[1].index('--model') + 1] = 'runs/v2_y_yolo_n/best.pt'
    if time.strftime('%H:%M') < '09:00':
        NS.run(YB, plateau=False, stop_at='09:10')
    NS.log('yolo done')


if __name__ == '__main__':
    main()
