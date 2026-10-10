"""Put a bounded identical packed-mask mmap cache on owned Colab shared RAM.

The dataset/teacher labels are unchanged. Adjacent chunks_buf.npy symlinks
are read by the existing Masks API; the Colab process needs no new recipe.
"""
import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import psutil


def prepare(root, manifest, cache_root, budget_bytes=3 * 1024 ** 3,
            ram_reserve_bytes=6 * 1024 ** 3):
    cache_root.mkdir(parents=True, exist_ok=True)
    existing = sum(p.stat().st_size for p in cache_root.glob('*.npy'))
    budget = min(max(0, budget_bytes - existing), shutil.disk_usage(cache_root).free,
                 max(0, psutil.virtual_memory().available - ram_reserve_bytes))
    candidates = []
    for source in dict.fromkeys(manifest['train'] + manifest['val']):
        npz = root / source / 'chunks.npz'
        npy = npz.with_name('chunks_buf.npy')
        if npy.exists() and npy.stat().st_mtime >= npz.stat().st_mtime:
            continue
        if npy.exists() or npy.is_symlink():
            raise RuntimeError('Refuse to replace an existing/stale mask cache: ' + str(npy))
        with zipfile.ZipFile(npz) as archive:
            size = archive.getinfo('buf.npy').file_size
        candidates.append((size, source))
    # Large buffers first, bounded by the RAM budget; never allocate all masks.
    used, selected = 0, []
    for size, source in sorted(candidates, reverse=True):
        if size > budget - used:
            continue
        npz = root / source / 'chunks.npz'
        npy = npz.with_name('chunks_buf.npy')
        target = cache_root / (hashlib.sha256(source.encode()).hexdigest() + '.npy')
        if target.exists():
            raise RuntimeError('Unexpected orphaned RAM cache: ' + str(target))
        # Decompression briefly needs another buffer of the same size. Recheck
        # available RAM after each allocation, not just at process startup.
        if psutil.virtual_memory().available < size + ram_reserve_bytes:
            continue
        temp = target.with_suffix('.tmp.npy')
        try:
            with np.load(npz) as archive:
                buf = archive['buf']
                np.save(temp, buf)
                mapped = np.load(temp, mmap_mode='r')
                if not np.array_equal(buf, mapped):
                    raise ValueError('RAM cache differs from teacher mask buffer')
                del mapped, buf
            if temp.stat().st_size != size:
                raise ValueError('RAM cache size differs from packed mask NPY')
            temp.replace(target)
            os.utime(target, (max(target.stat().st_mtime, npz.stat().st_mtime),) * 2)
            npy.symlink_to(target)
        except BaseException:
            temp.unlink(missing_ok=True)
            if not npy.exists():
                target.unlink(missing_ok=True)
            raise
        used += size
        selected.append(source)
    return dict(ram_cache_bytes=used, sources=selected, budget_bytes=budget,
                ram_available_bytes=psutil.virtual_memory().available,
                disk_free_bytes=shutil.disk_usage(root).free)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/content/doorseg'))
    parser.add_argument('--manifest', type=Path, default=Path('/content/doorseg_manifest.json'))
    parser.add_argument('--cache-root', type=Path, default=Path('/dev/shm/doorseg_masks'))
    args = parser.parse_args()
    result = prepare(args.root, json.loads(args.manifest.read_text()), args.cache_root)
    (args.root / 'runs/doorseg_c_background/ram_cache_manifest.json').write_text(json.dumps(result))
    print(json.dumps(result), flush=True)
