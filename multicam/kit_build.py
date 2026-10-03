"""A training kit for a borrowed machine (Kaggle, 2 x T4): the code and a small slice of the data, listed in
data/kit/manifest.json for kit_run.py, which the job server hands out (sam31_jobs: /api/jobs/kit*).

The slice: WINDOWS (both cameras: targets, masks, the ReID teachers' vectors, the recent-hall images, the long
backgrounds of their files) with the video re-encoded lighter (CRF 28 instead of 18, one frame per tick, a key
frame every 12 as the original -- ~6x smaller, for the tunnel), N_DRAFTS rough draft frames with their
backgrounds, the stand's mask, the scene maps. It is for measuring speed, not for training a model for use.

usage: kit_build.py                 (the small kit, served through the tunnel)
       kit_build.py full PWFILE     (every window, the held-out day, all drafts, the owner's exam, the best checkpoint:
                                     a staging tree of hard links, then an encrypted 7-Zip in 4 GB volumes,
                                     data/kit_full/kit_full.7z.001 ..., for a fast host -- kit_run.py --archive)"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
KIT = ROOT / 'data' / 'kit'
WINDOWS = ['20260917_26105', '20260921_08100']
N_DRAFTS = 1200
CODE = ['train_v2.py', 'slot_v2.py', 'slot_model.py', 'deimv2_vit.py', 'train_slots.py', 'v2_data.py', 'v2_fast.py',
        'v2_eval.py', 'sam31_reid.py']


def reencode(src, dst):
    import cv2
    import day_proxy
    if dst.exists():
        return
    tmp = dst.with_name(dst.stem + '.tmp.mp4')
    cmd = [day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-i', str(src), '-an', '-c:v', 'libx264', '-preset', 'veryfast',
           '-crf', '28', '-g', '12', '-keyint_min', '12', '-sc_threshold', '0', '-bf', '0', '-pix_fmt', 'yuv420p']
    for keep in (['-fps_mode', 'passthrough'], ['-vsync', '0']):
        r = subprocess.run(cmd + keep + [str(tmp)], capture_output=True, text=True)
        if r.returncode == 0:
            break
    else:
        raise RuntimeError(r.stderr[-400:])
    n = [int(cv2.VideoCapture(str(p)).get(cv2.CAP_PROP_FRAME_COUNT)) for p in (src, tmp)]
    if n[0] != n[1]:
        raise RuntimeError('frame count %s' % n)
    tmp.replace(dst)


def main():
    import v2_data as VD
    t0 = time.time()
    (KIT / 'video').mkdir(parents=True, exist_ok=True)
    files = [(f, f) for f in CODE]
    bgs = set()
    for tag in WINDOWS:
        for cam in ('cam1', 'cam2'):
            v2 = Path('data') / 'v2' / tag / cam
            for f in ('rows.npz', 'bg_recent.npy', 'bg_ticks.npy', 'meta.json', 'emb.npz'):
                if (ROOT / v2 / f).exists():
                    files.append((str(v2 / f), str(v2 / f)))
            seg = Path('data') / 'sam31_seg' / tag / cam
            files.append((str(seg / 'chunks.npz'), str(seg / 'chunks.npz')))
            light = Path('data') / 'kit' / 'video' / ('%s_%s.mp4' % (tag, cam))
            reencode(ROOT / seg / 'video.mp4', ROOT / light)
            files.append((str(light), str(seg / 'video.mp4')))
            meta = json.load(open(ROOT / v2 / 'meta.json'))
            for _, name in meta['segments']:
                bgs.add('%s_%s_%s.jpg' % (meta['day'], cam, name.replace('.mp4', '')))
            print(tag, cam, 'ready', round(time.time() - t0), 's', flush=True)
    rng = random.Random(0)
    drafts = VD.draft_list()
    rng.shuffle(drafts)
    for lab, img, day, cam, seg in drafts[:N_DRAFTS]:
        for p in (lab, img):
            rel = str(Path(p).resolve().relative_to(ROOT))
            files.append((rel, rel))
        bgs.add('%s_%s_%s.jpg' % (day, cam, seg))
    for b in sorted(bgs):
        rel = str(Path('data') / 'seg_datasets' / 'backgrounds' / b)
        if (ROOT / rel).exists():
            files.append((rel, rel))
    for f in ('depth_cam1_da2_large.npy', 'depth_cam2_da2_large.npy', 'world_cam1.npy', 'world_cam2.npy'):
        files.append((str(Path('data') / 'scene' / f), str(Path('data') / 'scene' / f)))
    files.append((str(Path('data') / 'v2' / 'poster.npz'), str(Path('data') / 'v2' / 'poster.npz')))
    items = [{'src': s.replace('\\', '/'), 'dst': d.replace('\\', '/'), 'size': (ROOT / s).stat().st_size} for s, d in files]
    json.dump({'made': time.strftime('%Y-%m-%d %H:%M'), 'windows': WINDOWS, 'drafts': N_DRAFTS, 'files': items},
              open(KIT / 'manifest.json', 'w'), indent=0)
    print('kit: %d files, %.2f GB, %.0f s' % (len(items), sum(i['size'] for i in items) / 1e9, time.time() - t0), flush=True)


def full(password, workers=3):
    """Everything a borrowed machine needs to train and test like the shop's card does."""
    import concurrent.futures as cf
    import shutil
    import v2_data as VD
    t0 = time.time()
    out = ROOT / 'data' / 'kit_full'
    tree = out / 'tree'
    if tree.exists():
        shutil.rmtree(tree)
    tree.mkdir(parents=True)
    vids = ROOT / 'data' / 'kit' / 'video'
    vids.mkdir(parents=True, exist_ok=True)

    def put(src, dst=None):
        src = ROOT / src
        d = tree / (dst or src.relative_to(ROOT))
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(src, d)                       # same disk: no copy
        except OSError:
            shutil.copy2(src, d)
    for f in CODE + ['gold.py', 'v2_teacher_test.py', 'kit_run.py']:
        put(f)
    tags = sorted(t for t in os.listdir(ROOT / 'data' / 'v2') if (ROOT / 'data' / 'v2' / t).is_dir() and not t.startswith('20260918'))
    jobs = []
    bgs = set()
    for tag in tags:
        for cam in ('cam1', 'cam2'):
            v2 = Path('data') / 'v2' / tag / cam
            if not (ROOT / v2 / 'rows.npz').exists():
                continue
            for f in ('rows.npz', 'bg_recent.npy', 'bg_ticks.npy', 'meta.json', 'emb.npz'):
                if (ROOT / v2 / f).exists():
                    put(v2 / f)
            seg = Path('data') / 'sam31_seg' / tag / cam
            put(seg / 'chunks.npz')
            jobs.append((ROOT / seg / 'video.mp4', vids / ('%s_%s.mp4' % (tag, cam)), seg / 'video.mp4'))
            meta = json.load(open(ROOT / v2 / 'meta.json'))
            for _, name in meta['segments']:
                bgs.add('%s_%s_%s.jpg' % (meta['day'], cam, name.replace('.mp4', '')))
    with cf.ThreadPoolExecutor(workers) as ex:                # ffmpeg does the work, the threads only wait
        for f in cf.as_completed([ex.submit(reencode, a, b) for a, b, _ in jobs]):
            f.result()
    for _, light, dst in jobs:
        put(light.relative_to(ROOT), dst)
    print('windows: %d cameras, %.0f s' % (len(jobs), time.time() - t0), flush=True)
    for lab, img, day, cam, seg in VD.draft_list():
        put(Path(lab).resolve().relative_to(ROOT)); put(Path(img).resolve().relative_to(ROOT))
        bgs.add('%s_%s_%s.jpg' % (day, cam, seg))
    for b in sorted(bgs):
        p = Path('data') / 'seg_datasets' / 'backgrounds' / b
        if (ROOT / p).exists():
            put(p)
    for f in ('depth_cam1_da2_large.npy', 'depth_cam2_da2_large.npy', 'world_cam1.npy', 'world_cam2.npy'):
        put(Path('data') / 'scene' / f)
    for f in ('data/v2/poster.npz', 'data/v2_test/v2b.json', 'data/logs/gold_all.json', 'data/seg_datasets/backgrounds/exam_frames.json',
              'data/weights/deimv2_s/model.safetensors'):
        if (ROOT / f).exists():
            put(f)
    exam = json.load(open(ROOT / 'data' / 'logs' / 'gold_all.json'))['split']['test_ids']
    where = json.load(open(ROOT / 'data' / 'seg_datasets' / 'backgrounds' / 'exam_frames.json'))
    for ident in exam:
        for f in ('data/paint/%s.jpg' % ident, 'data/paint/%s_mask.png' % ident, 'data/v2_exam/%s.jpg' % ident):
            if (ROOT / f).exists():
                put(f)
        w = where.get(ident, {})
        b = 'data/seg_datasets/backgrounds/%s_%s_%s.jpg' % (w.get('day'), w.get('cam', 'cam1'), str(w.get('segment', ''))[:-4])
        if (ROOT / b).exists() and not (tree / b).exists():
            put(b)
    st = json.load(open(ROOT / 'data' / 'paint' / 'state.json', encoding='utf-8'))
    json.dump({k: v for k, v in st.items() if k in exam}, open(tree / 'data' / 'paint' / 'state.json', 'w', encoding='utf-8'))
    for run in ('v2_c_repvit', 'v2_b'):
        for ck in ('best.pt', 'last.pt'):
            if (ROOT / 'runs' / run / ck).exists():
                (tree / 'runs' / run).mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / 'runs' / run / ck, tree / 'runs' / run / ck)       # a copy: the run goes on writing its own
                shutil.copy2(ROOT / 'runs' / run / 'args.json', tree / 'runs' / run / 'args.json')
    n = sum(1 for _ in tree.rglob('*') if _.is_file())
    size = sum(p.stat().st_size for p in tree.rglob('*') if p.is_file())
    print('tree: %d files, %.2f GB, %.0f s' % (n, size / 1e9, time.time() - t0), flush=True)
    for old in out.glob('kit_full.7z*'):
        old.unlink()
    z = r'C:\Program Files\7-Zip\7z.exe'
    r = subprocess.run([z, 'a', '-t7z', '-mx=0', '-mhe=on', '-p' + password, '-v4g', str(out / 'kit_full.7z'), '.'],
                       cwd=str(tree), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout[-500:] + r.stderr[-500:])
    vols = sorted(out.glob('kit_full.7z.*'))
    json.dump({'made': time.strftime('%Y-%m-%d %H:%M'), 'files': n, 'bytes': size, 'windows': tags,
               'volumes': [{'name': v.name, 'size': v.stat().st_size} for v in vols]}, open(out / 'kit_full.json', 'w'), indent=1)
    print('archive: %d volumes, %.2f GB, %.0f s' % (len(vols), sum(v.stat().st_size for v in vols) / 1e9, time.time() - t0), flush=True)


if __name__ == '__main__':
    if sys.argv[1:2] == ['full']:
        pw = sys.argv[2]
        full(Path(pw).read_text(encoding='utf-8').strip() if os.path.exists(pw) else pw)     # a file: not on the command line
    else:
        main()
