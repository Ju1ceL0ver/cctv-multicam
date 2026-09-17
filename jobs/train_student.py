"""Train the -seg student on teacher pseudo-labels, inside a fixed GPU window.

``time=`` makes Ultralytics size the epoch count to the budget itself (it
re-plans after the first epoch), so the run ends on schedule with a finished
cosine schedule and close_mosaic tail, instead of being cut off mid-plan.
The watchdog is the backstop: last.pt is written every epoch, so a hard kill
loses at most one epoch.
"""
import os, sys, time, shutil, threading
from datetime import datetime

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
LOG = os.path.join(HOME, '_train_student.log')
HARD_STOP = tuple(int(v) for v in os.environ.get('RA_TRAIN_STOP', '06:20').split(':'))
BUDGET_HOURS = float(os.environ.get('RA_TRAIN_HOURS', '6.0'))
# yolo26s-seg at batch 8 / imgsz 1280 needs ~12.1 GB and spills out of the
# 3060's 12 GB into shared memory: measured 9 s/iteration, two hours an epoch.
SIZE = os.environ.get('RA_TRAIN_SIZE', 'n')
BATCH = int(os.environ.get('RA_TRAIN_BATCH', '8'))


def log(*a):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('%s %s\n' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)))


def watchdog():
    while True:
        now = datetime.now()
        if HARD_STOP <= (now.hour, now.minute) and now.hour < 10:
            log('WATCHDOG: hard stop, freeing the GPU for the next job')
            os._exit(0)
        time.sleep(20)


def main():
    os.chdir(ROOT)
    from ultralytics import YOLO
    weights = 'yolo26%s-seg.pt' % SIZE
    base = os.path.join(ROOT, 'models', weights)
    if not os.path.exists(base):
        shutil.copy2(os.path.join(HOME, '_bench_models', weights), base)
    model = YOLO(base)
    log('training', base, 'batch', BATCH, 'budget %.1f h' % BUDGET_HOURS)
    model.train(
        data=os.path.join(ROOT, 'data', 'student_seg_v1', 'data.yaml'),
        imgsz=1280, epochs=200, time=BUDGET_HOURS, batch=BATCH, workers=4,
        patience=25, cos_lr=True, close_mosaic=10, single_cls=True,
        hsv_v=0.5, degrees=3.0, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0,
        project=os.path.join(ROOT, 'runs', 'student_seg'), name='v1_%s' % SIZE, exist_ok=True,
        seed=42, plots=True, verbose=True)
    log('training finished')


if __name__ == '__main__':
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        main()
    except Exception:
        import traceback; log('FATAL\n' + traceback.format_exc())
    finally:
        os._exit(0)
