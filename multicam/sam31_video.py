"""SAM 3.1 (Object Multiplex, Meta's own code in venv_sam3, ext/sam3) on a short stretch of the raw 2560x1440
recording at 12.5 fps: text "person" on the first frame, propagated forward.

SAM 3.1 keeps every frame's outputs of a session on the card: at 2560x1440 it filled 12 GB by frame ~80 and
crawled. So the stretch goes in chunks of CHUNK frames overlapping by OVERLAP, the session closed after each.
Here every chunk's own answer is kept as it is (its local track numbers); sam31_reid.py links the chunks at
the shared frames, joins the pieces of one person by ReID and draws the video.

usage (venv_sam3):
  sam31_video.py TAG film DAY CAM FILM_SECONDS [LENGTH_S]     (FILM_SECONDS on the day film's clock)
  sam31_video.py TAG raw PATH SECOND [LENGTH_S]               (a raw 15-minute file, from SECOND - 3)
  -> data/logs/sam31/TAG/frames/NNNNN.jpg, chunks.npz, info.json"""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
CKPT = ROOT / 'data' / 'weights' / 'sam3' / 'sam3.1_multiplex.pt'
CHUNK, OVERLAP = 32, 4           # a session slows down after ~40 frames at 2560x1440 (the card fills up)
# The model squeezes every frame to 1008x1008 itself (io_utils: resize((image_size, image_size))) and predicts
# masks on a 288x288 grid, stretched to the frame's size. So a frame given at 2560x1440 only costs card memory:
# RA_S31_SIZE=1008 gives it the 1008x1008 frame it would see anyway; the masks come back 1008x1008 and are
# stretched here, on the CPU, to the full frame.
import os
SIZE = int(os.environ.get('RA_S31_SIZE', '0'))
CHUNK = int(os.environ.get('RA_S31_CHUNK', CHUNK))
STEP = int(os.environ.get('RA_S31_STEP', '2'))     # every STEP-th frame of the 25 fps recording
STORE = os.environ.get('RA_S31_STORE')              # 'WxH': the frames (and so the masks) kept at this size
OUT_ROOT = os.environ.get('RA_S31_ROOT')            # where TAG/ goes (default data/logs/sam31)
# The detector runs its backbone on batches of 16 frames (batched_grounding_batch_size): a 20 GB peak on a
# 12 GB card, spilled to the host -- 3.3 s a frame. At 1 frame a batch: 0.6 s a frame, 6 GB (28.09).
GB = int(os.environ.get('RA_S31_GB', '1'))
BF16_BACKBONE = os.environ.get('RA_S31_BF16BB') == '1'  # the image backbone's weights in bf16


def raw_frames(path, second, seconds, step=2):
    """Every `step`-th frame of a raw file from `second` for `seconds`."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, second * 1000.0))
    out, k = [], 0
    while len(out) < int(seconds * 25 / step):
        if not cap.grab():
            break
        if k % step == 0:
            ok, f = cap.retrieve()
            if ok:
                out.append(f)
        k += 1
    cap.release()
    return out


def pack(m):
    """(x1, y1, x2, y2, packed bits of the mask inside its box) or None for an empty mask."""
    ys, xs = np.nonzero(m)
    if not len(xs):
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
    return (x1, y1, x2, y2), np.packbits(m[y1:y2, x1:x2])


def masks_of(out, shape):
    """[(local id, prob, bool mask HxW)] from one frame's outputs."""
    import cv2
    ids = out.get('out_obj_ids', [])
    probs = out.get('out_probs', [1.0] * len(ids))
    ms = out.get('out_binary_masks', [])
    res = []
    for i, p, m in zip(list(ids), list(probs), list(ms)):
        m = np.asarray(m.detach().cpu() if hasattr(m, 'detach') else m).astype(bool)
        if m.ndim == 3:
            m = m[0]
        if m.shape != shape:
            m = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        res.append((int(i), float(p), m))
    return res


def main():
    import cv2
    import torch
    from sam3.model_builder import build_sam3_predictor
    import sam3.model.decoder as sam3_decoder        # it asks for Flash Attention only; PyTorch on Windows has none
    from torch.nn.attention import sdpa_kernel, SDPBackend
    sam3_decoder.sdpa_kernel = lambda *a, **k: sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    tag, kind = sys.argv[1], sys.argv[2]
    t0 = time.time()
    if kind == 'film':
        import sam3_video as V
        day, cam, film = sys.argv[3], sys.argv[4], float(sys.argv[5])
        length = float(sys.argv[6]) if len(sys.argv) > 6 else 15.0
        frames = V.raw_run(day, cam, film, length, STEP)
        src = {'day': day, 'cam': cam, 'film': film}
    else:
        path, second = sys.argv[3], float(sys.argv[4])
        length = float(sys.argv[5]) if len(sys.argv) > 5 else 15.0
        frames = raw_frames(path, max(0.0, second - 3.0), length, STEP)
        src = {'path': path, 'second': second}
    if STORE:
        sw, sh = map(int, STORE.lower().split('x'))
        frames = [cv2.resize(f, (sw, sh), interpolation=cv2.INTER_AREA) for f in frames]
    H, W = frames[0].shape[:2]
    out_dir = (Path(OUT_ROOT) if OUT_ROOT else ROOT / 'data' / 'logs' / 'sam31') / tag
    folder = out_dir / 'frames'
    shutil.rmtree(folder, ignore_errors=True); folder.mkdir(parents=True)
    for k, f in enumerate(frames):
        cv2.imwrite(str(folder / ('%05d.jpg' % k)), f, [cv2.IMWRITE_JPEG_QUALITY, 93])        # the kept frame
    model_in = folder
    if SIZE:                                  # what the model gets: the frame squeezed as it would do itself
        model_in = out_dir / 'frames_in'
        shutil.rmtree(model_in, ignore_errors=True); model_in.mkdir()
        for k, f in enumerate(frames):
            cv2.imwrite(str(model_in / ('%05d.jpg' % k)), cv2.resize(f, (SIZE, SIZE), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 95])
    pred = build_sam3_predictor(checkpoint_path=str(CKPT), version='sam3.1', use_fa3=False, max_num_objects=16)   # 16: 3.1's own bucket; 32 filled 12 GB
    import inspect                           # the repo's start_session passes init_state() a keyword the multiplex model lacks
    orig = pred.model.init_state
    known = set(inspect.signature(orig).parameters)
    pred.model.init_state = lambda *a, **kw: orig(*a, **{k: v for k, v in kw.items() if k in known})
    for m in pred.model.modules():
        if hasattr(m, 'batched_grounding_batch_size'):
            m.batched_grounding_batch_size = GB
    if BF16_BACKBONE:
        for name, m in pred.model.named_modules():
            if name.endswith('backbone.vision_backbone'):
                m.to(torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    rows, buf, offs, chunks, start = [], [], [0], [], 0
    while start < len(frames):
        stop = min(len(frames), start + CHUNK)
        sub = out_dir / ('chunk_%04d' % start)
        shutil.rmtree(sub, ignore_errors=True); sub.mkdir()
        for k in range(start, stop):
            shutil.copy(model_in / ('%05d.jpg' % k), sub / ('%05d.jpg' % (k - start)))
        local = {}
        with torch.autocast('cuda', dtype=torch.bfloat16):
            sid = pred.handle_request(dict(type='start_session', resource_path=str(sub), offload_video_to_cpu=True))['session_id']
            first = pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
            local[0] = masks_of(first.get('outputs', {}) or {}, (H, W))
            for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
                local[int(resp.get('frame_index', len(local)))] = masks_of(resp.get('outputs', {}) or {}, (H, W))
            pred.handle_request(dict(type='close_session', session_id=sid))
        torch.cuda.empty_cache()
        shutil.rmtree(sub, ignore_errors=True)
        for kl in sorted(local):
            for i, p, m in local[kl]:
                pk = pack(m)
                if pk is None:
                    continue
                (x1, y1, x2, y2), bits = pk
                rows.append((start, start + kl, i, p, x1, y1, x2, y2))
                buf.append(bits); offs.append(offs[-1] + len(bits))
        chunks.append([start, stop, round(time.time() - t1)])
        print('chunk', start, stop, '%.0f s' % (time.time() - t1), flush=True)
        if stop >= len(frames):
            break
        start = stop - OVERLAP
    t2 = time.time()
    if SIZE:
        shutil.rmtree(model_in, ignore_errors=True)
    np.savez_compressed(out_dir / 'chunks.npz', rows=np.array(rows, dtype=np.float64).reshape(-1, 8),
                        buf=np.concatenate(buf) if buf else np.zeros(0, np.uint8), offs=np.array(offs, dtype=np.int64))
    info = dict(src, tag=tag, frames=len(frames), size=[W, H], model_size=SIZE or None, step=STEP, chunk=CHUNK, overlap=OVERLAP, grounding_batch=GB, bf16_backbone=BF16_BACKBONE, chunks=chunks,
                read_s=round(t1 - t0), track_s=round(t2 - t1), s_per_frame=round((t2 - t1) / max(1, len(frames)), 3),
                gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
    json.dump(info, open(out_dir / 'info.json', 'w'), indent=1)
    print(info)


if __name__ == '__main__':
    main()
