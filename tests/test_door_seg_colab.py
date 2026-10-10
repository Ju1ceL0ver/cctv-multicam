"""Transport/ownership safety, without a GPU, network or credentials."""
import json
import sys
from pathlib import Path

import pytest
import zipfile
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import door_seg_colab as C
import door_seg_supervise as supervisor


def test_disk_gate_counts_uncompressed_mmap_and_checkpoint_reserve(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    npz = source / 'chunks.npz'
    with zipfile.ZipFile(npz, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('buf.npy', b'0' * 10000)
    manifest = {'train': ['source'], 'val': [],
                'files': [{'path': 'source/chunks.npz', 'size': npz.stat().st_size},
                          {'path': 'source/video.mp4', 'size': 1000}]}
    # Compressed dataset alone fits, mmap plus atomic-write reserve does not.
    monkeypatch.setattr(C.shutil, 'disk_usage', lambda root: SimpleNamespace(free=12000))
    with pytest.raises(RuntimeError, match='mask mmap'):
        C.training_space_plan(tmp_path, manifest, reserve_bytes=2000)
    monkeypatch.setattr(C.shutil, 'disk_usage', lambda root: SimpleNamespace(free=14000))
    plan = C.training_space_plan(tmp_path, manifest, reserve_bytes=2000)
    assert plan['mmap_bytes'] == 10000 and plan['remaining_bytes'] == 1000
    assert plan['required_bytes'] == 13000
    (source / 'chunks_buf.npy').write_bytes(b'0' * 10000)
    assert C.training_space_plan(tmp_path, manifest, reserve_bytes=2000)['mmap_bytes'] == 0
    # An incomplete final-name download must still count as needing replacement.
    (source / 'video.mp4').write_bytes(b'bad')
    assert C.training_space_plan(tmp_path, manifest, reserve_bytes=2000)['remaining_bytes'] == 1000


def test_ram_cache_is_bounded_identical_and_reused_by_masks(tmp_path, monkeypatch):
    import numpy as np
    import door_seg_colab_cache as cache
    from sam31_reid import Masks
    source = tmp_path / 'source'
    source.mkdir()
    buf = np.arange(10000, dtype=np.uint8)
    npz = source / 'chunks.npz'
    np.savez_compressed(npz, buf=buf, rows=np.zeros((1,8)), offs=np.array([0,10000]))
    manifest = {'train':['source'], 'val':[],
                'files':[{'path':'source/chunks.npz','size':npz.stat().st_size}]}
    monkeypatch.setattr(cache.psutil, 'virtual_memory', lambda: SimpleNamespace(available=100000))
    monkeypatch.setattr(cache.shutil, 'disk_usage', lambda root: SimpleNamespace(free=100000))
    result = cache.prepare(tmp_path, manifest, tmp_path/'ram', budget_bytes=15000, ram_reserve_bytes=50000)
    path = source/'chunks_buf.npy'
    assert path.is_symlink() and np.array_equal(np.load(path), buf)
    assert result['ram_cache_bytes'] <= 15000
    assert Masks.to_npy(npz) == path and path.is_symlink()
    assert np.array_equal(Masks(npz,mmap=True).buf, buf)
    assert cache.prepare(tmp_path, manifest, tmp_path/'ram', budget_bytes=15000,
                         ram_reserve_bytes=50000)['ram_cache_bytes'] == 0
    assert C.training_space_plan(tmp_path, manifest, reserve_bytes=0)['mmap_bytes'] == 0
    # A later source cannot grow the already-owned cache above the total budget.
    source2=tmp_path/'source2';source2.mkdir()
    np.savez_compressed(source2/'chunks.npz',buf=buf)
    manifest['train'].append('source2')
    assert cache.prepare(tmp_path,manifest,tmp_path/'ram',budget_bytes=15000,
                         ram_reserve_bytes=50000)['ram_cache_bytes']==0


def test_ram_cache_preserves_memory_reserve(tmp_path, monkeypatch):
    import door_seg_colab_cache as cache
    import numpy as np
    source=tmp_path/'source'; source.mkdir()
    np.savez_compressed(source/'chunks.npz',buf=np.zeros(10000,np.uint8))
    monkeypatch.setattr(cache.psutil,'virtual_memory',lambda:SimpleNamespace(available=20000))
    result=cache.prepare(tmp_path,{'train':['source'],'val':[]},tmp_path/'ram',
                         budget_bytes=20000,ram_reserve_bytes=15000)
    assert result['ram_cache_bytes']==0 and not (source/'chunks_buf.npy').exists()


def transport(tmp_path):
    access = tmp_path / 'access.json'
    access.write_text(json.dumps({'JUP_URL': 'https://example.invalid', 'JUP_TOKEN': 'test-only'}))
    return C.Transfer(access, tmp_path / 'data')


def test_incomplete_download_does_not_replace_checkpoint(tmp_path, monkeypatch):
    t = transport(tmp_path)
    path = t.root / 'last.pt'
    path.parent.mkdir()
    path.write_bytes(b'good')
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
        def iter_content(self, size): return iter([b'bad'])
    monkeypatch.setattr(C.requests, 'get', lambda *a, **k: Response())
    monkeypatch.setattr(C.time, 'sleep', lambda *a: None)
    with pytest.raises(ValueError, match='length mismatch'):
        t.fetch({'path': 'last.pt', 'size': 8})
    assert path.read_bytes() == b'good'


def test_completed_video_not_downloaded_again(tmp_path, monkeypatch):
    t = transport(tmp_path)
    t.root.mkdir()
    (t.root / 'video.mp4').write_bytes(b'complete')
    monkeypatch.setattr(C.requests, 'get', lambda *a, **k: pytest.fail('Unexpected download'))
    assert t.fetch({'path': 'video.mp4', 'size': 8}) == 8


def test_windows_refuses_duplicate_colab_training(tmp_path, monkeypatch):
    monkeypatch.setattr(supervisor, 'ROOT', tmp_path)
    run = tmp_path / 'runs' / 'candidate'
    run.mkdir(parents=True)
    (run / 'COLAB_ACTIVE.json').write_text('{"active":true}')
    with pytest.raises(RuntimeError, match='owned by Colab'):
        supervisor.main('candidate')


def test_checkpoint_receiver_checks_transport_and_preserves_newer_state(tmp_path):
    import hashlib
    import torch
    from door_seg_colab_restore import restore
    run = tmp_path / 'run'
    folder = run / 'colab_upload'
    folder.mkdir(parents=True)
    torch.save({'step': 12, 'model': {}, 'opt': {}}, run / 'last.pt')
    incoming = tmp_path / 'incoming.pt'
    torch.save({'step': 10, 'model': {}, 'opt': {}}, incoming)
    block = incoming.read_bytes()
    (folder / 'chunk.part').write_bytes(block)
    manifest = folder / 'last.pt.json'
    manifest.write_text(json.dumps({'target': 'last.pt', 'size': len(block),
                                    'sha256': hashlib.sha256(block).hexdigest(), 'parts': ['chunk.part']}))
    with pytest.raises(ValueError, match='newer checkpoint'):
        restore(run)
    assert torch.load(run / 'last.pt', weights_only=False)['step'] == 12
    meta = json.loads(manifest.read_text())
    meta['sha256'] = '0' * 64
    manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='checksum mismatch'):
        restore(run)
    assert torch.load(run / 'last.pt', weights_only=False)['step'] == 12


def test_checkpoint_receiver_acknowledges_only_finite_state(tmp_path):
    import hashlib
    import torch
    from door_seg_colab_restore import restore
    run = tmp_path / 'run'
    folder = run / 'colab_upload'
    folder.mkdir(parents=True)
    part = folder / 'chunk.part'
    torch.save({'step': 20, 'model': {'x': torch.ones(2)}, 'opt': {}}, part)
    blob = part.read_bytes()
    manifest = folder / 'last.pt.json'
    manifest.write_text(json.dumps({'target': 'last.pt', 'size': len(blob),
                                    'sha256': hashlib.sha256(blob).hexdigest(), 'parts': [part.name]}))
    restore(run)
    assert torch.load(run / 'last.pt', weights_only=False)['step'] == 20
    assert not manifest.exists() and not part.exists()
