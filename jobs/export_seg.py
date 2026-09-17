"""Teacher suggestions -> Ultralytics YOLO-seg dataset, split by day.

Pseudo-labels go straight to training here (that is the distillation), which is
why this does not reuse annotate/export.py: that exporter deliberately refuses
anything a human has not reviewed, and writes boxes only. The student is -seg,
so each object is written as its polygon, normalised.

Validation during training is two held-out DAYS of pseudo-labels. That number
measures how well the student copies the teacher, not ground truth -- the real
yardstick is the hand-reviewed set harvested from the raw 13-14 September
recordings, which share neither days nor source with anything here.
"""
import os, json, glob, shutil, yaml, collections

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
SRC = os.path.join(ROOT, 'data', 'annotate')
DST = os.path.join(ROOT, 'data', 'student_seg_v1')
VAL_DAYS = {'day_20260903', 'day_20260911'}


def norm_poly(poly, w, h):
    pts = []
    for x, y in poly:
        pts.append('%.5f %.5f' % (min(max(x / w, 0.0), 1.0), min(max(y / h, 0.0), 1.0)))
    return ' '.join(pts)


def main():
    shutil.rmtree(DST, ignore_errors=True)
    stats = collections.Counter()
    for sug in sorted(glob.glob(os.path.join(SRC, 'suggestions', '*', '*.json'))):
        group = os.path.basename(os.path.dirname(sug))
        stem = os.path.basename(sug)[:-5]
        img = os.path.join(SRC, 'images', group, stem + '.jpg')
        if not os.path.exists(img):
            stats['missing_image'] += 1
            continue
        d = json.load(open(sug, encoding='utf-8'))
        w, h = d['width'], d['height']
        split = 'val' if group in VAL_DAYS else 'train'
        lines = []
        for b in d['boxes']:
            poly = b.get('polygon')
            if not poly or len(poly) < 3:
                poly = [[b['x1'], b['y1']], [b['x2'], b['y1']], [b['x2'], b['y2']], [b['x1'], b['y2']]]
                stats['box_as_polygon'] += 1
            lines.append('0 ' + norm_poly(poly, w, h))
        name = '%s__%s' % (group, stem)
        for sub in ('images', 'labels'):
            os.makedirs(os.path.join(DST, sub, split), exist_ok=True)
        target = os.path.join(DST, 'images', split, name + '.jpg')
        try:
            os.link(img, target)
        except OSError:
            shutil.copy2(img, target)
        with open(os.path.join(DST, 'labels', split, name + '.txt'), 'w') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        stats[split] += 1
        stats[split + '_objects'] += len(lines)
        if not lines:
            stats[split + '_empty'] += 1
    with open(os.path.join(DST, 'data.yaml'), 'w') as f:
        yaml.safe_dump({'path': DST, 'train': 'images/train', 'val': 'images/val',
                        'names': {0: 'person'}}, f, sort_keys=False)
    print(json.dumps(dict(stats), indent=1))


if __name__ == '__main__':
    main()
