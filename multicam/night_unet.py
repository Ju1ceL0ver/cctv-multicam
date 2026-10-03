"""After the YOLO run of 01.10 morning: the plain UNet++ (unetpp.py) until the live counter needs the card.
If YOLO on SAM 3.1 (stage A) is still below the plain YOLO26n on the owner's test (F1 0.8925) after its second
epoch, it is stopped and its stage B skipped: there is not enough morning for both. Log: data/logs/night_seq.log."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import night_seq as NS

YOLO_BASE_F1 = 0.8925
UNET = ('unetpp_a', ['unetpp.py', '--out', 'runs/unetpp_a', '--n', '5000', '--epochs', '20', '--until', '09:20'], {})


def main():
    import bg
    NS.log('unet up: waiting for YOLO')
    while True:
        f = NS.f1s('v2_y_yolo_n_sam31')
        if not NS.alive('v2_y_yolo_n_sam31') and not NS.alive('v2_y_yolo_n_owner2') and not bg.running('night_yolo.py'):
            break
        if len(f) >= 2 and max(f) < YOLO_BASE_F1:
            for line in bg.running('night_yolo.py'):
                subprocess.run(['taskkill', '/PID', line.split('|')[0], '/F'], capture_output=True)
            (ROOT / 'runs' / 'v2_y_yolo_n_sam31' / 'STOP').write_text('below the plain YOLO26n')
            NS.log('YOLO on SAM 3.1 below the plain one', f, '-> stopped, stage B skipped')
            for _ in range(60):
                if not NS.alive('v2_y_yolo_n_sam31'):
                    break
                time.sleep(20)
            for line in NS.alive('v2_y_yolo_n_sam31'):
                subprocess.run(['taskkill', '/PID', line.split('|')[0], '/F', '/T'], capture_output=True)
            break
        time.sleep(60)
    if time.strftime('%H:%M') < '09:00':
        NS.run(UNET, plateau=False)
    NS.log('unet done')


if __name__ == '__main__':
    main()
