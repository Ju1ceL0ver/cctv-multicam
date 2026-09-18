"""How much identity survives when the teacher looks at fewer frames.

Covering a whole day costs frames: at 2.8 pairs/s the big segmentation model cannot
see 25 fps of two cameras for nine hours. This measures, on the one clip with real
labels, what each frame rate actually costs -- evaluated only on the frames that rate
would have looked at, since the frames it skips are simply not part of its output."""
import sys, os, json, numpy as np
ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from person3d import Camera
from fusion2 import build, fuse
from clipdata import load, FPS
from eval_ids import evaluate

clip = sys.argv[1] if len(sys.argv) > 1 else 'c103700'
strides = [int(x) for x in (sys.argv[2].split(',') if len(sys.argv) > 2 else ['1', '2', '4', '6', '8'])]
GT = os.path.join('data', 'raw_clips', clip, 'gt_identity_yolo26x-seg.json')
dets0, feats0, meta, embs0 = load(clip)
clean0 = meta.get('clean')
gt0 = json.load(open(GT))
cams = {c: Camera(c, json.load(open('data/calib_final.json'))) for c in ('cam1', 'cam2')}

res = {}
for s in strides:
  for off in range(0, s, max(1, s // 3)):
    dets, feats, embs, clean, gt = {}, {}, {}, {}, {}
    for c in ('cam1', 'cam2'):
        f = np.round(dets0[c][:, 0] * FPS).astype(int)
        keep = np.where(f % s == off)[0]
        remap = {int(o): i for i, o in enumerate(keep)}
        dets[c] = dets0[c][keep]; feats[c] = feats0[c][keep]
        if embs0 and c in embs0 and len(embs0[c]):
            embs[c] = embs0[c][keep]
        if clean0 is not None and c in clean0:
            clean[c] = np.asarray(clean0[c])[keep]
        gt[c] = {str(remap[int(k)]): v for k, v in gt0[c].items() if int(k) in remap}
    tmp = os.path.join('data', 'raw_clips', clip, '_gt_stride%d_%d.json' % (s, off))
    json.dump(gt, open(tmp, 'w'))
    per_cam, _ = build(cams, dets, feats, embs or None, clean or None)
    items = per_cam['cam1'] + per_cam['cam2']
    segs, people, _ = fuse(per_cam)
    label = {'cam1': {}, 'cam2': {}}
    for pid, p in enumerate(people):
        for si in p['segments']:
            for m in segs[si]['members']:
                it = items[m]
                for di in it['det']:
                    label[it['cam']][int(di)] = pid
    rows, errors = evaluate(label, tmp, verbose=False)
    tot = sum(r['detections'] for r in rows.values())
    weighted = sum(r['main_id_share'] * r['detections'] for r in rows.values()) / tot
    wrong = sum(sum(c.values()) - max(c.values()) for _, c in errors)
    main = {g: r for g, r in rows.items() if r['detections'] > 400 // s}
    res.setdefault(s, []).append((weighted, 100 * wrong / max(tot, 1)))
    print('stride %d off %d (%.1f fps) | dets %d | pieces %d | people %d | one-id %.3f | wrong %.1f%% | %s'
          % (s, off, FPS / s, tot, len(items), len(people), weighted, 100 * wrong / max(tot, 1),
             ', '.join('%s %.2f/%d' % (g, r['main_id_share'], r['ids']) for g, r in sorted(main.items()))), flush=True)
    os.remove(tmp)
print()
for s in strides:
    v = np.array(res[s])
    print('SUMMARY stride %d (%.1f fps): one-id %.3f +- %.3f   wrong %.1f%% over %d samples'
          % (s, FPS / s, v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), len(v)), flush=True)
