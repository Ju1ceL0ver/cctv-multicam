"""Transport/ownership safety, without a GPU, network or credentials."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import door_seg_colab as C
import door_seg_supervise as supervisor


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
