"""CPU-only scheduling and immutable input guards for the bounded probe."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import door_seg_sequence_probe as Q


def test_manifest_detects_owner_edits_and_rejects_test(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, 'ROOT', tmp_path)
    original = Q.P.cohort
    monkeypatch.setattr(Q.P, 'cohort', lambda: original(tmp_path))
    state = {tag: {'done': True, 'role': 'train', 'cuts': {}} for tag in Q.P.TAGS}
    labels = tmp_path / 'data/door_side/20260918.json'
    labels.parent.mkdir(parents=True); labels.write_text(json.dumps(state))
    for tag in Q.P.TAGS:
        d = tmp_path / 'data/sam31_door' / tag / 'cam1'; d.mkdir(parents=True)
        for name in ('video.mp4', 'chunks.npz', 'info.json'): (d/name).write_bytes(b'fixture')
    before = Q.inputs()
    state[Q.P.TAGS[0]]['cuts'] = {'piece': [1.2]}; labels.write_text(json.dumps(state))
    assert Q.inputs() != before
    state[Q.P.TAGS[0]]['role'] = 'test'; labels.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='TRAIN'): Q.inputs()


def test_worker_cannot_write_c_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(Q, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='separate'):
        Q.worker(SimpleNamespace(out='runs/doorseg_c_background', seconds=120))
    assert not (tmp_path/'runs').exists()


def test_scheduler_excludes_own_ancestry_and_blocks_another_sequence(monkeypatch):
    import psutil
    monkeypatch.setattr(Q.P, 'blockers', lambda: ['foreign GPU app PID 70'])
    monkeypatch.setattr(psutil, 'Process', lambda: SimpleNamespace(parents=lambda: [SimpleNamespace(pid=71)]))
    rows = [{'pid': os.getpid(), 'name': 'python.exe', 'cmdline': ['door_seg_sequence_probe.py']},
            {'pid': 71, 'name': 'python.exe', 'cmdline': ['door_seg_sequence_probe.py']},
            {'pid': 72, 'name': 'python.exe', 'cmdline': ['door_seg_sequence_probe.py', '--wait']}]
    monkeypatch.setattr(psutil, 'process_iter', lambda keys: [SimpleNamespace(info=row) for row in rows])
    assert Q.blockers() == ['additional GPU job PID 72', 'foreign GPU app PID 70']
