"""Train the small segmentation model on what the owner has actually checked.

Distillation with a human in the loop: the big model draws the outlines, the owner
throws out what he could not read, and this trains the student that will later run on
full frames at full speed. Every run first rebuilds the dataset, so a night picks up
whatever was reviewed during the day.

usage: train_student_seg.py [--model yolo26n-seg.pt] [--imgsz 960] [--epochs 80]"""
import os, sys, json, glob, subprocess, time
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
MODELS = r'C:\Users\ArykovAA\cctv_ai\retail_analytics\models'
DATA = os.path.join(ROOT, 'data', 'seg_dataset')


def arg(name, default):
    a = sys.argv[1:]
    return a[a.index(name) + 1] if name in a else default


def main():
    model = arg('--model', os.path.join(MODELS, 'yolo26n-seg.pt'))
    imgsz = int(arg('--imgsz', '960'))
    epochs = int(arg('--epochs', '80'))
    batch = int(arg('--batch', '8'))
    print('%s rebuilding the dataset from reviewed clips' % time.strftime('%H:%M:%S'), flush=True)
    r = subprocess.run([sys.executable, 'export_seg_dataset.py'], cwd=ROOT, capture_output=True, text=True)
    print((r.stdout or '').strip().split('\n')[-1] if r.stdout else (r.stderr or '')[-300:], flush=True)
    n_train = len(glob.glob(os.path.join(DATA, 'train', 'images', '*.jpg')))
    n_val = len(glob.glob(os.path.join(DATA, 'val', 'images', '*.jpg')))
    if n_train < 200:
        print('only %d reviewed frames -- not training yet' % n_train, flush=True); return
    print('%s training on %d frames, checking against %d' % (time.strftime('%H:%M:%S'), n_train, n_val), flush=True)
    from ultralytics import YOLO
    name = 'student_seg_%s' % datetime.now().strftime('%Y%m%d')
    m = YOLO(model)
    m.train(data=os.path.join(DATA, 'data.yaml'), imgsz=imgsz, epochs=epochs, batch=batch,
            device=0, workers=8, patience=15, project=os.path.join(ROOT, 'runs'), name=name,
            exist_ok=True, pretrained=True, seed=0, val=True, plots=False)
    best = os.path.join(ROOT, 'runs', name, 'weights', 'best.pt')
    if os.path.exists(best):
        print('%s done: %s' % (time.strftime('%H:%M:%S'), best), flush=True)
        res = YOLO(best).val(data=os.path.join(DATA, 'data.yaml'), imgsz=imgsz, device=0, plots=False)
        out = {'when': datetime.now().isoformat(timespec='seconds'), 'frames_train': n_train, 'frames_val': n_val,
               'imgsz': imgsz, 'weights': best,
               'box_mAP50': float(getattr(res.box, 'map50', 0)), 'box_mAP': float(getattr(res.box, 'map', 0)),
               'mask_mAP50': float(getattr(res.seg, 'map50', 0)), 'mask_mAP': float(getattr(res.seg, 'map', 0))}
        json.dump(out, open(os.path.join(ROOT, 'data', 'student_seg_last.json'), 'w'), indent=1)
        print('student: masks mAP50 %.3f, mAP50-95 %.3f' % (out['mask_mAP50'], out['mask_mAP']), flush=True)


if __name__ == '__main__':
    main()
