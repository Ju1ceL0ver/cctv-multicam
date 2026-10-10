"""Continue a staged door student on Colab, with its full frozen data manifest.

Credentials are supplied in a private JSON file, never in a command or log.
Run this script as a detached process on the allocated VM. Checkpoints and
progress are copied back to the existing Windows run through Jupyter's API.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests


def training_space_plan(root, manifest, reserve_bytes=3 * 1024 ** 3):
    """Count remaining files AND packed-mask mmap expansion, without inflating it.

    Small NPZ files must already be present. Checkpoint/temporary-write reserve
    is additional to the exact dataset; compressed NPZ size is not mmap size.
    """
    remaining = sum(item['size'] for item in manifest['files']
                    if not (root / item['path']).exists() or
                    (root / item['path']).stat().st_size != item['size'])
    mmap_bytes = 0
    for source in dict.fromkeys(manifest['train'] + manifest['val']):
        npz = root / source / 'chunks.npz'
        npy = npz.with_name('chunks_buf.npy')
        if npy.exists() and npy.stat().st_mtime >= npz.stat().st_mtime:
            continue
        with zipfile.ZipFile(npz) as archive:
            mmap_bytes += archive.getinfo('buf.npy').file_size
    free = shutil.disk_usage(root).free
    required = remaining + mmap_bytes + reserve_bytes
    if free < required:
        raise RuntimeError('Insufficient disk for dataset + mask mmap + checkpoint reserve: '
                           'free=%d required=%d remaining=%d mmap=%d reserve=%d' %
                           (free, required, remaining, mmap_bytes, reserve_bytes))
    return dict(free_bytes=free, remaining_bytes=remaining, mmap_bytes=mmap_bytes,
                reserve_bytes=reserve_bytes, required_bytes=required)


def ensure_training_space(root, manifest, run):
    try:
        return training_space_plan(root, manifest)
    except RuntimeError:
        # Colab's base-image libraries often occupy lower filesystem layers:
        # uninstalling them does not reclaim actual writable-layer capacity.
        # Cache identical packed masks on shared RAM, bounded to 3 GiB total
        # and leaving at least 6 GiB available for workers/sequence buffers.
        if not Path('/dev/shm').is_dir():
            raise
        from door_seg_colab_cache import prepare
        cached = prepare(root, manifest, Path('/dev/shm/doorseg_masks'))
        (run / 'ram_cache_manifest.json').write_text(json.dumps(cached))
        return training_space_plan(root, manifest)  # Fail safely if still short.


class Transfer:
    def __init__(self, access, root):
        credentials = json.loads(Path(access).read_text())
        self.url = credentials['JUP_URL'].rstrip('/')
        self.headers = {'Authorization': 'token ' + credentials['JUP_TOKEN']}
        if credentials.get('JUP_CF_ID'):
            self.headers.update({'CF-Access-Client-Id': credentials['JUP_CF_ID'],
                                 'CF-Access-Client-Secret': credentials['JUP_CF_SECRET']})
        self.root = root

    def fetch(self, item):
        path = self.root / item['path']
        size = item['size']
        if path.exists() and path.stat().st_size == size:
            return size
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + '.part')
        for attempt in range(4):
            try:
                url = self.url + '/files/' + quote('cctv_ai/multicam/' + item['path'], safe='/')
                with requests.get(url, headers=self.headers, stream=True, timeout=(30, 90)) as r:
                    r.raise_for_status()
                    with partial.open('wb') as out:
                        for block in r.iter_content(4 * 1024 * 1024):
                            out.write(block)
                if partial.stat().st_size != size:
                    raise ValueError('download length mismatch: ' + item['path'])
                partial.replace(path)
                return size
            except Exception as exc:
                if attempt == 3:
                    raise
                print('retry', item['path'], type(exc).__name__, flush=True)
                time.sleep(3 * (attempt + 1))

    def put(self, path, remote=None):
        remote = remote or str(path.relative_to(self.root))
        if path.stat().st_size > 32 * 1024 ** 2:
            # Cloudflare limits request bodies. Upload chunks, then a completion
            # manifest; the CPU-only Windows receiver verifies and assembles it.
            parts = []
            folder = str(Path(remote).parent / 'colab_upload')
            with path.open('rb') as src:
                sha = hashlib.sha256()
                size = 0
                while block := src.read(4 * 1024 ** 2):
                    sha.update(block)
                    size += len(block)
                digest = sha.hexdigest()
                src.seek(0)  # Keep the same inode even if training replaces last.pt.
                index = 0
                while block := src.read(24 * 1024 ** 2):
                    name = '%s.%s.%03d.part' % (path.name, digest, index)
                    self.put_bytes(block, folder + '/' + name)
                    parts.append(name)
                    index += 1
            meta = json.dumps({'target': path.name, 'sha256': digest,
                               'size': size, 'parts': parts}).encode()
            meta_remote = folder + '/' + path.name + '.json'
            self.put_bytes(meta, meta_remote)
            # Receiver deletes this manifest only after checksum + finite state
            # validation and atomic restoration on Windows.
            url = self.url + '/api/contents/' + quote('cctv_ai/multicam/' + meta_remote, safe='/')
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
                r = requests.get(url, headers=self.headers, params={'content': '0'}, timeout=30)
                if r.status_code == 404:
                    return
                r.raise_for_status()
                time.sleep(5)
            raise RuntimeError('Windows checkpoint receiver did not acknowledge ' + path.name)
            return
        self.put_bytes(path.read_bytes(), remote)

    def put_bytes(self, value, remote):
        url = self.url + '/api/contents/' + quote('cctv_ai/multicam/' + remote, safe='/')
        body = {'type': 'file', 'format': 'base64',
                'content': base64.b64encode(value).decode('ascii')}
        r = requests.put(url, headers=self.headers, json=body, timeout=(30, 180))
        r.raise_for_status()


def main(args):
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.manifest).read_text())
    transfer = Transfer(args.access, root)
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=True)
    status = run / 'colab_status.json'
    ownership = run / 'COLAB_ACTIVE.json'
    phase, child = 'preparing', None
    synced, stop_backup = {}, threading.Event()

    def record(**extra):
        nonlocal phase
        phase = extra.pop('phase', phase)
        value = dict(phase=phase, session=args.session, pid=os.getpid(),
                     child_pid=child.pid if child else None, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)
        status.write_text(json.dumps(value, indent=2))
        print(json.dumps(value), flush=True)

    def sync_once():
        for path in sorted(run.iterdir()):
            if path.suffix not in ('.pt', '.md', '.json', '.jsonl', '.log') and path.name != 'STOP':
                continue
            if path.name.startswith('rejected_') or not path.is_file():
                continue  # The rejected batch can be large; preserve it on the VM for diagnosis.
            fingerprint = (path.stat().st_size, path.stat().st_mtime_ns)
            if synced.get(path.name) != fingerprint:
                transfer.put(path)
                synced[path.name] = fingerprint

    def backup_loop():
        while not stop_backup.wait(120):
            try:
                sync_once()
            except Exception as exc:
                print('backup retry needed:', type(exc).__name__, flush=True)

    ownership.write_text(json.dumps({'active': True, 'session': args.session, 'run': args.run}))
    transfer.put(ownership)
    backup = threading.Thread(target=backup_loop, daemon=True)
    backup.start()
    try:
        files = manifest['files']
        total = sum(x['size'] for x in files)
        missing = sum(x['size'] for x in files if not (root / x['path']).exists())
        if shutil.disk_usage(root).free < missing + 3 * 1024 ** 3:
            raise RuntimeError('Insufficient disk for the full unchanged dataset')
        record(phase='downloading', files=len(files), total_bytes=total)
        if manifest.get('small_bundle'):
            small = [x for x in files if not x['path'].endswith('video.mp4')]
            if any(not (root / x['path']).exists() or (root / x['path']).stat().st_size != x['size'] for x in small):
                transfer.fetch(manifest['small_bundle'])
                bundle = root / manifest['small_bundle']['path']
                with zipfile.ZipFile(bundle) as z:
                    for member in z.infolist():
                        target = root / member.filename
                        if not target.exists() or target.stat().st_size != member.file_size:
                            z.extract(member, root)
                bundle.unlink()
            # Check the full-size training update early, before copying all videos.
            first_video = manifest['train'][0] + '/video.mp4'
            transfer.fetch(next(x for x in files if x['path'] == first_video))
            env = dict(os.environ, RA_DS_GLUED='1', RA_DS_EVAL_N='400')
            smoke = subprocess.run([sys.executable, '-u', str(root / 'door_seg_colab_smoke.py')],
                                   cwd=root, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
            (run / 'colab_smoke.log').write_text(smoke.stdout)
            if smoke.returncode:
                raise RuntimeError('Early BF16 batch6 compatibility smoke failed; inspect colab_smoke.log')
            record(phase='downloading', early_smoke='finite batch6 forward/backward')
            ensure_training_space(root, manifest, run)
        completed = 0
        # Small labels/checkpoint first; two streams avoid saturating the shop's upload.
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(transfer.fetch, item) for item in sorted(files, key=lambda x: x['size'])]
            last_report = time.monotonic()
            for task in concurrent.futures.as_completed(tasks):
                completed += task.result()
                if time.monotonic() - last_report > 45:
                    record(completed_bytes=completed, total_bytes=total)
                    last_report = time.monotonic()
        os.chdir(root)
        sys.path.insert(0, str(root))
        os.environ.update(RA_DS_GLUED='1', RA_DS_LR='1e-4', RA_DS_EVAL_N='400',
                          RA_DS_EVAL_STEPS='2000', PYTHONIOENCODING='utf-8')
        import torch
        import cv2
        import door_seg as S
        import sam31_reid as R
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        actual = lambda ds: [str(p.relative_to(root)) for p in ds]
        if actual(S.sources()) != manifest['train'] or actual(S.sources(True)) != manifest['val']:
            raise RuntimeError('Source manifest mismatch; training not started')
        checkpoint = run / 'last.pt'
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        expected = manifest.get('checkpoint_sha256',
                                '18b4fab92332e26bcae520bddb1771c947cb21fe407e30aafd5350e6cc8b4b4f')
        initial_step = int(manifest.get('checkpoint_step', 8604))
        if digest != expected:
            raise RuntimeError('Initial model/optimizer checksum mismatch')
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        S.checkpoint_finite(state)
        if state['step'] != initial_step or 'opt' not in state:
            raise RuntimeError('Expected the manifest-verified weights AND optimizer')
        del state
        record(phase='preparing_masks', space_plan=ensure_training_space(root, manifest, run))
        for d in S.sources() + S.sources(True):
            R.Masks.to_npy(d / 'chunks.npz')
        record(phase='compatibility_check', checkpoint_sha256=digest,
               train_sources=len(manifest['train']), val_sources=len(manifest['val']),
               torch=torch.__version__, gpu=torch.cuda.get_device_name(),
               bf16=torch.cuda.is_bf16_supported())
        # Original six-example update and BF16, including backward and finite gradients.
        # A failed smoke test blocks training rather than silently changing precision/batch.
        if not manifest.get('small_bundle'):
            smoke = subprocess.run([sys.executable, '-u', str(root / 'door_seg_colab_smoke.py')],
                                   cwd=root, env=os.environ.copy(), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
            (run / 'colab_smoke.log').write_text(smoke.stdout)
            if smoke.returncode:
                raise RuntimeError('Compatibility smoke failed; inspect colab_smoke.log')
        launcher = root / 'door_seg_colab_train.py'
        launcher.write_text("import os,sys,torch,cv2\nfrom pathlib import Path\n"
                            "torch.set_num_threads(2);cv2.setNumThreads(2)\n"
                            "import door_seg as S\n"
                            "S.train(sys.argv[1],sys.argv[2],batch=6,workers=2)\n")
        # Colab has no shop-hours constraint; twelve hours is a bounded allocation.
        stop = time.strftime('%H:%M', time.localtime(time.time() + 12 * 3600))
        with (run / 'colab_train.log').open('a') as out:
            child = subprocess.Popen([sys.executable, '-u', str(launcher), args.run, stop],
                                     cwd=root, env=os.environ.copy(), stdout=out, stderr=subprocess.STDOUT)
            record(phase='training', start_step=initial_step, stop_local=stop, workers=2, batch=6)
            sync_once()
            code = child.wait()
        record(phase='finished' if code == 0 else 'failed', returncode=code)
    except Exception as exc:
        record(phase='failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stop_backup.set()
        backup.join()  # Do not race the final backup against a still-running upload.
        # Explicit ownership survives errors until recovery; Windows must not start a duplicate.
        sync_once()
        if phase == 'finished':
            ownership.write_text(json.dumps({'active': False, 'session': args.session, 'run': args.run}))
            transfer.put(ownership)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--access', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', default='/content/doorseg')
    parser.add_argument('--run', default='doorseg_c_background')
    parser.add_argument('--session', default='doorseg-c')
    main(parser.parse_args())
