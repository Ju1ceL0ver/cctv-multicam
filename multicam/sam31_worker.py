"""A SAM 3.1 worker: takes a job (160-240 frames of one camera, 1008x1008, as an mp4) from the shop machine,
tracks "person" through it with SAM 3.1, sends the masks back, takes the next. The same code runs on the shop
machine itself and in Google Colab (T4) -- there the cell fetches this file from the server and runs it.

usage: sam31_worker.py URL KEY [NAME]        (or exec'd with URL, KEY, NAME set)
  URL  -- the labelling site, e.g. https://....trycloudflare.com ; KEY -- the workers' key

A job's result: npz {rows: (frame in job, SAM's number, prob, x1, y1, x2, y2) at 1008x1008, buf/offs: the
mask bits inside each box}. Nothing of the job stays on the worker."""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

SAM_IN = 1008


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def http(url, data=None, timeout=600):
    req = urllib.request.Request(url, data=data, method='POST' if data is not None else 'GET',
                                 headers={'Content-Type': 'application/octet-stream'} if data is not None else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


# ---------------------------------------------------------------- setting up (Colab: once per session)

def ensure_sam3(weights_url=None):
    """sam3 importable and the SAM 3.1 checkpoint on disk -> its path."""
    home = '/content' if os.path.isdir('/content') else tempfile.gettempdir()
    src = os.path.join(home, 'sam3')
    if os.path.isdir(os.path.join(src, 'sam3')):  # after a restart the repo folder in the working directory would
        sys.path.insert(0, src)                   # import as an empty namespace 'sam3' (no __file__): the package first
        for m in [m for m in sys.modules if m == 'sam3' or m.startswith('sam3.')]:
            del sys.modules[m]
    try:
        import sam3
        if getattr(sam3, '__file__', None) is None:
            raise ImportError('namespace only')
    except ImportError:
        if not os.path.isdir(src):
            log('installing sam3 ...')
            zp = os.path.join(home, 'sam3.zip')
            urllib.request.urlretrieve('https://github.com/facebookresearch/sam3/archive/refs/heads/main.zip', zp)
            shutil.unpack_archive(zp, home)
            os.rename(os.path.join(home, 'sam3-main'), src)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '-e', src], check=True)
        sys.path.insert(0, src)
        for m in [m for m in sys.modules if m == 'sam3' or m.startswith('sam3.')]:
            del sys.modules[m]
    import re                                  # its dependencies, one by one, as the import asks for them -- tried in a
    for _ in range(15):                        # child process: a failed import here leaves torch half-imported and broken
        r = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, %r); '
                            'from sam3.model_builder import build_sam3_predictor' % src], capture_output=True, text=True)
        missing = re.findall(r"No module named '([^']+)'", r.stderr)
        if r.returncode == 0 or not missing:
            break
        top = missing[-1].split('.')[0]          # 'triton.runtime' -> the package triton
        name = {'cv2': 'opencv-python-headless', 'yaml': 'pyyaml', 'PIL': 'pillow', 'sklearn': 'scikit-learn'}.get(top, top)
        log('pip install', name)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', name], check=True)
    from sam3.model_builder import build_sam3_predictor  # noqa: F401
    local = os.environ.get('SAM31_CKPT')
    if local and os.path.exists(local):
        return local
    try:                                       # the worker's own Hugging Face login (licence accepted on facebook/sam3.1)
        from huggingface_hub import hf_hub_download
        return hf_hub_download('facebook/sam3.1', 'sam3.1_multiplex.pt')
    except Exception as e:
        if not weights_url:
            raise
        log('no Hugging Face access (%s): the weights from the shop machine, ~3.5 GB' % str(e)[:80])
        path = os.path.join('/content' if os.path.isdir('/content') else tempfile.gettempdir(), 'sam3.1_multiplex.pt')
        if not os.path.exists(path):
            urllib.request.urlretrieve(weights_url, path + '.part'); os.rename(path + '.part', path)
        return path


def build(ckpt):
    import inspect
    import torch
    from sam3.model_builder import build_sam3_predictor
    import sam3.model.decoder as sam3_decoder        # it asks for Flash Attention only: not on Windows, not on a T4
    from torch.nn.attention import sdpa_kernel, SDPBackend
    sam3_decoder.sdpa_kernel = lambda *a, **k: sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH])
    pred = build_sam3_predictor(checkpoint_path=ckpt, version='sam3.1', use_fa3=False, max_num_objects=16)
    orig = pred.model.init_state
    known = set(inspect.signature(orig).parameters)
    pred.model.init_state = lambda *a, **kw: orig(*a, **{k: v for k, v in kw.items() if k in known})
    for m in pred.model.modules():           # the detector's backbone one frame at a time: 16 overflowed the card
        if hasattr(m, 'batched_grounding_batch_size'):
            m.batched_grounding_batch_size = 1
    return pred


# ---------------------------------------------------------------- one job

def masks_of(out):
    import numpy as np
    ids = out.get('out_obj_ids', [])
    probs = out.get('out_probs', [1.0] * len(ids))
    res = []
    for i, p, m in zip(list(ids), list(probs), list(out.get('out_binary_masks', []))):
        m = np.asarray(m.detach().cpu() if hasattr(m, 'detach') else m).astype(bool)
        if m.ndim == 3:
            m = m[0]
        res.append((int(i), float(p), m))
    return res


def run_job(pred, mp4, dtype, stills=False):
    """stills: the job's frames are unrelated single frames (the varied drafts, the owner's painted frames) --
    each is its own one-frame session, nothing is tracked from one to the next."""
    import cv2
    import numpy as np
    import torch
    folder = tempfile.mkdtemp(prefix='sam31job_')
    try:
        cap, k = cv2.VideoCapture(mp4), 0
        while True:
            ok, f = cap.read()
            if not ok:
                break
            if f.shape[:2] != (SAM_IN, SAM_IN):
                f = cv2.resize(f, (SAM_IN, SAM_IN), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(folder, '%05d.jpg' % k), f, [cv2.IMWRITE_JPEG_QUALITY, 95]); k += 1
        cap.release()
        local, sid = {}, None
        parts = [(folder, 0)]
        if stills:                             # one folder per frame, each named 00000.jpg
            parts = []
            for i in range(k):
                sub = os.path.join(folder, 's%05d' % i)
                os.mkdir(sub)
                os.replace(os.path.join(folder, '%05d.jpg' % i), os.path.join(sub, '00000.jpg'))
                parts.append((sub, i))
        try:
            for path, base in parts:
                with torch.autocast('cuda', dtype=dtype):
                    sid = pred.handle_request(dict(type='start_session', resource_path=path, offload_video_to_cpu=True))['session_id']
                    first = pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
                    local[base] = masks_of(first.get('outputs', {}) or {})
                    if not stills:
                        for resp in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
                            local[int(resp.get('frame_index', len(local)))] = masks_of(resp.get('outputs', {}) or {})
                    pred.handle_request(dict(type='close_session', session_id=sid))
                    sid = None
        finally:                               # a session left open after a failure keeps its memory on the card,
            if sid is not None:                # and every next job then runs out of it
                try:
                    pred.handle_request(dict(type='close_session', session_id=sid))
                except Exception:
                    pass
            import gc
            gc.collect(); torch.cuda.empty_cache()
        rows, buf, offs = [], [], [0]
        for kl in sorted(local):
            for i, p, m in local[kl]:
                ys, xs = np.nonzero(m)
                if not len(xs):
                    continue
                x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
                bits = np.packbits(m[y1:y2, x1:x2])
                rows.append((kl, i, p, x1, y1, x2, y2)); buf.append(bits); offs.append(offs[-1] + len(bits))
        out = io.BytesIO()
        np.savez_compressed(out, rows=np.array(rows, dtype=np.float64).reshape(-1, 7),
                            buf=np.concatenate(buf) if buf else np.zeros(0, np.uint8), offs=np.array(offs, dtype=np.int64),
                            frames=np.array(k))
        return out.getvalue(), k
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main(url, key, name=None, restart_fp16=False):
    import torch
    url = url.rstrip('/')
    name = name or os.environ.get('SAM31_WORKER') or ('colab-%s' % os.urandom(3).hex() if os.path.isdir('/content') else 'worker')
    from urllib.parse import quote
    q = lambda path: '%s%s%skey=%s&worker=%s' % (url, path, '&' if '?' in path else '?', quote(key), quote(name))   # a name may be Cyrillic
    fp16 = os.environ.get('SAM31_FP16') == '1'
    real_bf16 = torch.bfloat16
    if fp16:                                   # SAM's own bf16 -> fp16, for SAM's code only: its decorators read
        torch.bfloat16 = torch.float16         # torch.bfloat16 while it is imported ...
    try:
        ckpt = ensure_sam3(q('/api/jobs/weights'))
        if fp16:
            import importlib, pkgutil, sam3.model
            for m in pkgutil.walk_packages(sam3.model.__path__, 'sam3.model.'):
                try:
                    importlib.import_module(m.name)
                except Exception:
                    pass
    finally:
        torch.bfloat16 = real_bf16             # ... torch itself (loading the weights) needs the real one back
    if fp16:
        import types

        class _Torch(types.ModuleType):        # ... and at run time SAM sees a torch whose bfloat16 is float16
            def __getattr__(self, n):
                return torch.float16 if n == 'bfloat16' else getattr(torch, n)
        proxy = _Torch('torch')
        for n, m in list(sys.modules.items()):
            if (n == 'sam3' or n.startswith('sam3.')) and getattr(m, 'torch', None) is torch:
                m.torch = proxy
        log('fp16 everywhere (SAM\'s bf16 turned into fp16)')
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16      # a T4 (7.5) has no bf16
    log('worker', name, torch.cuda.get_device_name(0), str(dtype), 'torch', torch.__version__,
        'free %.1f of %.1f GB' % tuple(x / 2**30 for x in torch.cuda.mem_get_info()))
    import gc                                  # an interrupted earlier run keeps its model alive through the traceback
    sys.last_traceback = sys.last_value = sys.last_type = None
    gc.collect(); torch.cuda.empty_cache()
    held = torch.cuda.memory_allocated() / 2**30
    if held > 1.0:
        log('the card already holds %.1f GB from an earlier run in this session. '
            'Runtime -> Restart session, then run only this cell.' % held)
        return
    pred = build(ckpt)
    idle, ooms = 0, 0
    while True:
        try:
            st, body = http(q('/api/jobs/claim?stills=1'), timeout=120)      # stills=1: this worker knows single-frame jobs
            job = json.loads(body) if st == 200 and body else None
        except Exception as e:
            log('server not answering:', str(e)[:120]); time.sleep(60); continue
        if not job or not job.get('id'):
            idle += 1
            if idle % 10 == 1:
                log('no jobs, waiting')
            time.sleep(60); continue
        idle = 0
        t0 = time.time()
        mp4 = os.path.join(tempfile.gettempdir(), 'job_%s.mp4' % job['id'])
        try:
            urllib.request.urlretrieve(q('/api/jobs/%s/input' % job['id']), mp4)
            t1 = time.time()
            try:
                data, n = run_job(pred, mp4, dtype, bool(job.get('stills')))
            except Exception as e:             # on a T4 SAM's own bf16 (or bf16 mixed with our fp16) pushes attention
                clash = 'BFloat16' in str(e) or 'OutOfMemory' in type(e).__name__     # onto the slow kernel that
                if clash and restart_fp16 and dtype == torch.float16 and not os.environ.get('SAM31_FP16'):     # fills the card: all fp16
                    log('bf16 on this card: restarting with fp16 everywhere')
                    try:
                        http(q('/api/jobs/%s/fail' % job['id']), data=b'restarting in fp16')
                    except Exception:
                        pass
                    os.execve(sys.executable, [sys.executable] + sys.argv, dict(os.environ, SAM31_FP16='1'))
                raise
            t2 = time.time()
            http(q('/api/jobs/%s/result?frames=%d&sam_s=%.1f' % (job['id'], n, t2 - t1)), data=data)
            log('job', job['id'], '%d frames: download %.0f s, SAM %.0f s (%.2f s/frame), %.1f MB back'
                % (n, t1 - t0, t2 - t1, (t2 - t1) / max(1, n), len(data) / 1e6))
            ooms = 0
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            log('job', job.get('id'), 'failed:', repr(e)[:300])
            print(tb[-3000:], flush=True)
            try:
                http(q('/api/jobs/%s/fail' % job['id']), data=tb[-4000:].encode())
            except Exception:
                pass
            ooms = ooms + 1 if 'OutOfMemory' in type(e).__name__ else 0
            if ooms >= 2:                       # the card is full for a reason a retry won't fix
                log('out of GPU memory twice in a row: stopping. Runtime -> Restart session, then run only this cell.')
                return
            gc.collect(); torch.cuda.empty_cache()
            time.sleep(10)
        finally:
            if os.path.exists(mp4):
                os.remove(mp4)


if 'URL' in globals() and 'KEY' in globals():                        # exec'd from the Colab cell
    main(globals()['URL'], globals()['KEY'], globals().get('NAME'))
elif __name__ == '__main__' and len(sys.argv) >= 3:
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, restart_fp16=True)
