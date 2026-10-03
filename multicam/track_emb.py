"""The heavy ReID teachers' vectors for every shop track of a day -- the looks stitch.py joins pieces by.

Per track up to K views: the track cut into K equal stretches of time, from each the detection that is
largest and most sure while no other detection of that moment covers it (IoU < 0.3) when one such exists.
The crops come from the day's light copies (day_proxy, 960 px; day_masks.FRAMES gives the exact frame the
film shows at a time), both cameras read in parallel, in time order. Teachers, as teacher_emb.py:
TransReID ViT MSMT17 (clothes, 3840 = global + 4 local) and CSCI EVA02-L LTCC (body shape, 1024).

usage: track_emb.py DAY [K]   -> data/stitch/DAY_teachers.npz {keys, cloth, shape} (each the normalised mean)"""
import importlib.util
import json
import os
import sys
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
E = Path(r'C:\Users\ArykovAA\cctv_ai\ext')
TAG = 'yolo26x-seg'


def teachers(dev='cuda'):
    """(embed(crops BGR) -> (clothes N x 3840, shape N x 1024), both L2-normalised) -- as teacher_emb.py."""
    import cv2
    import torch
    sys.path.insert(0, str(E / 'pytorch-reid-models')); cwd = os.getcwd(); os.chdir(E)
    from reid_models.modeling import build_reid_model
    tr = build_reid_model('vit_transreid', 'msmt17').to(dev).eval(); os.chdir(cwd)
    pkg = types.ModuleType('model'); pkg.__path__ = [str(E / 'csci' / 'model')]; sys.modules['model'] = pkg
    sys.path.insert(0, str(E / 'csci'))
    cspec = importlib.util.spec_from_file_location('csci_cfg', E / 'csci' / 'config' / 'defaults.py')
    cmod = importlib.util.module_from_spec(cspec); cspec.loader.exec_module(cmod)
    for name, f in (('csci_eva', 'eva_cloth_embed.py'), ('csci_ez', 'ez_eva_custom.py')):
        sp = importlib.util.spec_from_file_location(name, E / 'csci' / 'model' / f)
        mm = importlib.util.module_from_spec(sp); sys.modules[name] = mm; sp.loader.exec_module(mm)
    sd = torch.load(E / 'weights' / 'csci_ltcc_eva02_l_cloth_best.pth', map_location='cpu')
    sd = {k.replace('module.', ''): v for k, v in (sd.get('model', sd) if isinstance(sd, dict) else sd).items()}
    cfg = cmod._C.clone(); cfg.MODEL.EXTRA_DIM = int(sd['mlp.fc2.weight'].shape[0])
    cs = sys.modules['csci_ez'].eva02_img_extra_token(pretrained=False, config=cfg, num_classes=sd['head.weight'].shape[0],
                                                     cloth=sd['cloth_embed'].shape[0] if 'cloth_embed' in sd else 300, cloth_xishu=3, spatial_avg=None)
    cs.load_state_dict(sd, strict=False); cs = cs.to(dev).eval().half()
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)

    def embed(crops):
        with torch.no_grad():
            rgb = [c[:, :, ::-1] for c in crops]
            a = torch.stack([torch.from_numpy(np.ascontiguousarray(cv2.resize(c, (128, 256)))) for c in rgb]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            v1 = torch.cat([tr(a[k:k + 64], cam_label=torch.zeros(len(a[k:k + 64]), dtype=torch.long, device=dev)) for k in range(0, len(a), 64)])
            b = torch.stack([torch.from_numpy(np.ascontiguousarray(cv2.resize(c, (224, 224)))) for c in rgb]).to(dev).permute(0, 3, 1, 2).float() / 255.0
            b = ((b - mean) / std).half()
            v2 = torch.cat([cs(b[k:k + 32]) for k in range(0, len(b), 32)]).float()
        n = lambda v: torch.nn.functional.normalize(v.float(), dim=1).cpu().numpy()
        return n(v1), n(v2)
    return embed


def views(day, root=ROOT, k=4):
    """[(cam, film time, box 2560 px, track key)] -- the chosen views of every shop track."""
    import day_movie
    idx = day_movie.index(day, root)
    cache, out = {}, []
    for name, tr in idx['tracks'].items():
        if not tr['shop'] or len(tr['rows']) < 3:
            continue
        clip, cam = tr['clip'], tr['cam']
        if clip not in cache:
            with np.load(Path(root) / 'data' / 'raw_clips' / clip / ('dets_%s.npz' % TAG)) as a:
                d = {c: a[c] for c in ('cam1', 'cam2')}
            by_t = {c: {} for c in d}
            for c in d:
                for r, t in enumerate(d[c][:, 0]):
                    by_t[c].setdefault(round(float(t), 3), []).append(r)
            cache[clip] = (d, by_t)
        d, by_t = cache[clip]
        rows = np.asarray(tr['rows']); times = np.asarray(tr['times'])
        for chunk in np.array_split(np.arange(len(rows)), min(k, len(rows))):
            best, bs = None, -1.0
            for i in chunk:
                r = rows[i]
                x1, y1, x2, y2, sc = d[cam][r, 1:6]
                others = [q for q in by_t[cam].get(round(float(d[cam][r, 0]), 3), []) if q != r]
                cover = 0.0
                for q in others:
                    b = d[cam][q, 1:5]
                    ix = max(0.0, min(x2, b[2]) - max(x1, b[0])); iy = max(0.0, min(y2, b[3]) - max(y1, b[1]))
                    cover = max(cover, ix * iy / max(1.0, (x2 - x1) * (y2 - y1)))
                s = float(sc) * (y2 - y1) * (0.2 if cover >= 0.3 else 1.0)
                if s > bs:
                    best, bs = i, s
            out.append((cam, float(times[best]), d[cam][rows[best], 1:5].astype(float).tolist(), name))
    return out


def crops_for(day, cam, items, root=ROOT):
    """Crops (BGR) of `items` (sorted by time) of one camera from the light copy of the day."""
    import day_masks
    out = []
    for t, box, key in items:
        frame, info = day_masks.FRAMES.at(day, cam, t, root)
        if frame is None:
            out.append(None); continue
        s = frame.shape[1] / 2560.0
        x1, y1, x2, y2 = [v * s for v in box]
        w, h = x2 - x1, y2 - y1
        X1, Y1 = max(0, int(x1 - 0.05 * w)), max(0, int(y1 - 0.05 * h))
        X2, Y2 = min(frame.shape[1], int(x2 + 0.05 * w) + 1), min(frame.shape[0], int(y2 + 0.05 * h) + 1)
        c = frame[Y1:Y2, X1:X2]
        out.append(c.copy() if c.shape[0] >= 16 and c.shape[1] >= 8 else None)
    return out


def main():
    day = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    t0 = time.time()
    V = views(day, ROOT, k)
    log = {'day': day, 'views': len(V)}
    print('views', len(V), '%.0f s' % (time.time() - t0), flush=True)
    per_cam = {c: sorted([(t, b, key) for cc, t, b, key in V if cc == c]) for c in ('cam1', 'cam2')}
    with ThreadPoolExecutor(2) as ex:                             # both cameras' copies read side by side
        got = dict(zip(per_cam, ex.map(lambda c: crops_for(day, c, per_cam[c]), per_cam)))
    print('crops %.0f s' % (time.time() - t0), flush=True)
    embed = teachers()
    acc = {}
    for cam, items in per_cam.items():
        crops = [(key, c) for (t, b, key), c in zip(items, got[cam]) if c is not None]
        for s in range(0, len(crops), 256):
            chunk = crops[s:s + 256]
            v1, v2 = embed([c for _, c in chunk])
            for (key, _), a, b in zip(chunk, v1, v2):
                x = acc.setdefault(key, [np.zeros(len(a)), np.zeros(len(b)), 0])
                x[0] += a; x[1] += b; x[2] += 1
    keys = sorted(acc)
    cloth = np.stack([acc[kk][0] / np.linalg.norm(acc[kk][0]) for kk in keys]).astype(np.float16)
    shape = np.stack([acc[kk][1] / np.linalg.norm(acc[kk][1]) for kk in keys]).astype(np.float16)
    out = ROOT / 'data' / 'stitch'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / ('%s_teachers.npz' % day), keys=np.array(keys), cloth=cloth, shape=shape)
    log.update(tracks=len(keys), seconds=round(time.time() - t0))
    json.dump(log, open(out / ('%s_teachers.json' % day), 'w'), indent=1)
    print(log, flush=True)


if __name__ == '__main__':
    main()
