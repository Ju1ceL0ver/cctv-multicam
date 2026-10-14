"""13.10.2026: doorseg_c (step14604) predictions on every /doorside stretch, once, for CPU experiments with the
tracker and the zone rules. Own previous mask at 0.4 as live; detections kept from 0.2 (low ones for continuation).
-> data/door_v2/dsc_cache/<tag>.npz: det rows [tick, score, x1, y1, x2, y2, opx, opy, onx, ony, mx0, my0, mw, mh, off]
and bits (packed masks of each detection's 1/4-res crop)."""
import json, os, sys
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = Path(os.environ.get('RA_CACHE_OUT') or ROOT / 'data/door_v2/dsc_cache')          # 14.10: other checkpoints into other folders
CK = Path(os.environ.get('RA_CACHE_CK') or ROOT / 'data/quality_backups/doorseg_c_kaggle_final_20261010/step14604_ce9b0a8498d3.pt')


def main():
    import torch
    import door_seg as S, door_seg_fast as F
    from door_seg_run import contexts
    from door_live_bench import Net
    OUT.mkdir(parents=True, exist_ok=True)
    net = S.DoorSeg('cuda'); net.load(torch.load(CK, map_location='cpu', weights_only=False)['model'])
    net.yolo.eval(); net.off.eval(); net.yolo.fuse(verbose=False)
    M = Net(net, 'eager')
    F_ = json.load(__import__('gzip').open(ROOT / 'data/door_v2/zone_feats.json.gz', 'rt'))
    for tag in F_['tags']:
        if (OUT / (tag + '.npz')).exists():
            continue
        asm = F.GPUChannels(S.background(tag.split('_')[1], 'cam1'))
        cap = cv2.VideoCapture(str(ROOT / 'data/sam31_door' / tag / 'cam1/video.mp4'))
        prev = np.zeros((S.H, S.W), np.uint8)
        rows, bits = [], []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for tick, ctx in contexts(cap, transform=lambda f: cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA)):
                pred = M.predict(asm(ctx, prev), thr=.2)
                prev = F.previous_union([p for p in pred if p[2] >= .4])
                for m, box, s, op, on in pred:
                    m = m[:S.H // 4]
                    ys, xs = np.nonzero(m)
                    if not len(xs):
                        continue
                    x0, y0, x1, y1 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                    b = np.packbits(m[y0:y1, x0:x1].reshape(-1))
                    rows.append([tick, s, *box, *op, *on, x0, y0, x1 - x0, y1 - y0, sum(len(x) for x in bits)])
                    bits.append(b)
        cap.release()
        np.savez_compressed(OUT / (tag + '.npz'), rows=np.array(rows, np.float32).reshape(-1, 15),
                            bits=np.concatenate(bits) if bits else np.zeros(0, np.uint8))
        print(tag, len(rows), flush=True)
    print('done')


if __name__ == '__main__':
    main()
