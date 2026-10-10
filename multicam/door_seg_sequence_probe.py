"""Two ordered TRAIN head updates, bounded and yielding to existing GPU jobs.

--check and --wait are CPU-only until all guards pass. The worker is a separate
process with a finite deadline. This never resumes C, edits labels or changes
production; two updates establish train/restore mechanics, not accuracy.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import door_seg_night_probe as P

ROOT = Path(__file__).resolve().parent
SOURCE = 'data/quality_backups/doorseg_c_kaggle_final_20261010/last_085f995430fb.pt'
SOURCE_SHA = '085f995430fbe3a973c9a9f460460be9e61f41cda6b02bc432abb4f03ac24ae8'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def blockers():
    found = P.blockers()
    import psutil
    try:
        own = {os.getpid(), *(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid', 'name', 'cmdline']):
            if p.info['pid'] in own:
                continue
            cmd = ' '.join(p.info['cmdline'] or []).lower()
            if 'python' in (p.info['name'] or '').lower() and any(x in cmd for x in
                    ('door_seg_sequence_probe.py', 'auto_night.py', 'train_student_seg.py',
                     'staff_learn.py', 'staff_train.py')):
                found.append('additional GPU job PID %d' % p.info['pid'])
    except (psutil.Error, OSError):
        found.append('additional process check failed')
    return sorted(set(found))


def inputs():
    tags = P.cohort()
    labels = json.loads((ROOT / 'data/door_side/20260918.json').read_text(encoding='utf-8'))
    chosen = {tag: labels[tag] for tag in tags}
    paths = []
    for tag in tags:
        for name in ('video.mp4', 'chunks.npz', 'info.json'):
            p = ROOT / 'data/sam31_door' / tag / 'cam1' / name
            if not p.is_file():
                raise FileNotFoundError(str(p))
            paths.append(p)
    backgrounds = sorted((ROOT / 'data/seg_datasets/backgrounds').glob('20260918_cam1_*.jpg'))
    if not backgrounds:
        backgrounds = sorted((ROOT / 'data/seg_datasets/backgrounds').glob('*_cam1_*.jpg'))
    paths += backgrounds
    return {'owner': chosen, 'files': [{'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
                                       'mtime_ns': p.stat().st_mtime_ns} for p in paths]}


def write(out, report):
    p = out / 'report.tmp.json'
    p.write_text(json.dumps(report, indent=2), encoding='utf-8')
    p.replace(out / 'report.json')


def worker(args):
    out = ROOT / args.out
    if not out.name.startswith('doorseg_sequence_'):
        raise ValueError('separate sequence output required')
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {'complete': False, 'heads_trained': False, 'quality_improvement_proven': False,
              'production_changed': False, 'updates': []}
    frozen = inputs()
    def guard():
        if time.monotonic() - started > args.seconds:
            raise TimeoutError('bounded sequence time exhausted')
        busy = blockers()
        if busy:
            raise RuntimeError('yielding GPU: ' + ', '.join(busy))
        if inputs() != frozen:
            raise RuntimeError('frozen TRAIN inputs changed')
    try:
        guard()
        from door_seg_supervise import gpu_busy
        if gpu_busy():
            raise RuntimeError('GPU utilization gate busy')
        source = ROOT / SOURCE
        if sha(source) != SOURCE_SHA:
            raise ValueError('finite final source SHA differs')
        manifest = {**frozen, 'file_sha256': {x['path']: sha(ROOT / x['path']) for x in frozen['files']}}
        manifest_sha = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        (out / 'train_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        # No CUDA/torch import on check/wait paths; only the guarded worker imports.
        import cv2
        import torch
        import door_seg as S
        from door_seg_sequence import DoorSegTemporal, sample_sequence, sequence_starts, sequence_loss
        from door_seg_sequence_state import restore, save, finite, named_groups
        from door_seg_sequence_update import update
        torch.set_num_threads(2); cv2.setNumThreads(2); torch.manual_seed(7)
        os.environ['RA_DS_GLUED'] = '1'
        sequences, descriptions = [], []
        allowed = [ROOT / 'data/sam31_door' / tag / 'cam1' for tag in P.TAGS]
        for d in allowed:
            guard(); src = S.Source(d)
            try:
                starts = sequence_starts(src, 4)
                at = next((at for at in starts[:200] if len(src.people(*at)) >= 2), None)
                if at is None:
                    raise ValueError('no bounded co-visible TRAIN sequence: ' + d.parent.name)
                seq = sample_sequence(src, 4, seed=7, at=at, allowed_sources=allowed)
                sequences.append(seq)
                descriptions.append({'source': str(d.relative_to(ROOT)), 'session': at[0],
                                     'first_tick': at[1], 'frames': 4,
                                     'people': [len(f['person_keys']) for f in seq]})
            finally:
                src.close()
        guard()
        state = torch.load(source, map_location='cpu', weights_only=False)
        finite(state)
        if state['step'] != 20604 or state['precision'] != 'fp32':
            raise ValueError('expected final finite FP32 source, never implicit BF16')
        weights = ROOT / 'yolo26s-seg.pt'
        if not weights.is_file():
            raise FileNotFoundError('offline constructor weights absent')
        net = DoorSegTemporal('cuda', weights=str(weights))
        optimizer, resume = restore(net, state)
        base, heads = named_groups(net)
        for _, param in base:
            param.requires_grad_(False)
        net.yolo.train(); net.off.eval(); net.temporal.train()
        # Frozen base includes BN running statistics, while the segmentation
        # head keeps its TRAIN output format for the original criterion.
        for module in net.yolo.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        criterion = net.yolo.init_criterion()
        for _ in range(resume['criterion_updates']):
            criterion.update()
        recipe = {'version': 1, 'train_manifest_sha256': manifest_sha, 'source_sha256': SOURCE_SHA,
                  'source_step': 20604, 'scope': 'two head-only diagnostic updates; base and BN frozen',
                  'sequence_length': 4, 'identity_weight': .1, 'head_lr': 1e-4,
                  'sample_seed': 7, 'torch_seed': 7,
                  'precision': 'fp32', 'sequences': descriptions}
        report.update(recipe=recipe, source_resume=resume, updates=[])
        def loss_for(seq):
            def compute():
                loss, stats = sequence_loss(net, [seq], criterion, 'cuda', before_frame=guard)
                if not stats['identity_anchors']:
                    raise ValueError('no real co-visible identity anchors')
                return loss, stats
            return compute
        before_heads = {k: v.clone() for k, v in net.temporal.state_dict().items()}
        save(net, optimizer, out / 'initial.pt', step=20604, precision='fp32', recipe=recipe)
        for seq in sequences:
            guard()
            result = update(net, optimizer, loss_for(seq), guard=guard)
            report['updates'].append(result)
            report.update(heads_trained=True, successful_updates=len(report['updates']))
            for block in ('yolo', 'off'):
                if not all(torch.equal(v.cpu(), state['model'][block][name])
                           for name, v in getattr(net, block).state_dict().items()):
                    raise RuntimeError('frozen base parameter/buffer changed')
            save(net, optimizer, out / 'last.pt', step=20604+len(report['updates']),
                 precision='fp32', recipe=recipe)
            write(out, report)
        changed = [k for k, v in net.temporal.state_dict().items() if not torch.equal(v, before_heads[k])]
        if not changed:
            raise RuntimeError('heads received no effective update')
        guard()
        saved = torch.load(out / 'last.pt', map_location='cpu', weights_only=False)
        # Reuse the same model to bound memory; restore both named Adam groups.
        restored, info = restore(net, saved)
        for block in ('yolo', 'off', 'temporal'):
            assert all(torch.equal(v.cpu(), saved['model'][block][name])
                       for name, v in getattr(net, block).state_dict().items())
        for group, stored_group in zip(restored.param_groups, saved['opt']['param_groups']):
            for param, key in zip(group['params'], stored_group['params']):
                for name, value in saved['opt']['state'].get(key, {}).items():
                    got = restored.state[param][name]
                    assert torch.equal(got.cpu(), value) if torch.is_tensor(value) else got == value
        assert sha(source) == SOURCE_SHA
        assert manifest['file_sha256'] == {x['path']: sha(ROOT / x['path']) for x in frozen['files']}
        guard()
        report.update(complete=True, heads_trained=True, changed_head_tensors=changed,
                      successful_updates=len(report['updates']), restored=info,
                      restored_model_optimizer_exact=True, frozen_base_exact=True,
                      source_unchanged=True, checkpoint_sha256=sha(out / 'last.pt'),
                      peak_gpu_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic()-started)
        write(out, report); print(json.dumps(report), flush=True)
    except Exception as exc:
        if 'torch' in locals() and 'sequences' in locals():
            torch.save({'sequences': sequences, 'updates_completed': len(report['updates']),
                        'reason': str(exc)}, out / 'rejected_sequences.pt')
        report.update(error_type=type(exc).__name__, error=str(exc), seconds=time.monotonic()-started)
        write(out, report)
        print(json.dumps(report), flush=True)
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true'); parser.add_argument('--wait', action='store_true')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--out', default='data/quality_backups/doorseg_sequence_probe_20261010')
    parser.add_argument('--seconds', type=int, default=120)
    args = parser.parse_args()
    if not 30 <= args.seconds <= 180:
        raise ValueError('bounded seconds must be30..180')
    if args.worker:
        return worker(args)
    inputs()
    if not (ROOT / SOURCE).is_file():
        raise FileNotFoundError(SOURCE)
    if args.check:
        print(json.dumps({'ready': not blockers(), 'blockers': blockers(), 'torch_imported': 'torch' in sys.modules,
                          'source': SOURCE, 'source_sha256_expected': SOURCE_SHA}), flush=True)
        return 0
    end = datetime.datetime.now().replace(hour=9, minute=35, second=0, microsecond=0)
    if end <= datetime.datetime.now():
        end += datetime.timedelta(days=1)
    while blockers():
        if not args.wait or datetime.datetime.now() >= end:
            print('deferred sequence probe; no GPU opened', flush=True); return 2
        time.sleep(20)
    # Parent stays CPU-only; its finite child deadline also covers imports and
    # blocked library calls that cannot execute the per-frame yield callback.
    log = ROOT / 'data/logs/doorseg_sequence_probe_worker_20261010.log'
    with log.open('a', encoding='utf-8') as output:
        cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--out', args.out,
               '--seconds', str(args.seconds)]
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT)
        print('own finite sequence worker PID', p.pid, flush=True)
        try:
            return p.wait(timeout=args.seconds+45)
        except subprocess.TimeoutExpired:
            import psutil
            child_cmd = psutil.Process(p.pid).cmdline()
            if str(Path(__file__).resolve()) not in child_cmd or '--worker' not in child_cmd:
                raise RuntimeError('child command differs; refusing PID termination')
            subprocess.run(['taskkill', '/PID', str(p.pid), '/T', '/F'], check=True, capture_output=True)
            p.wait(timeout=15)
            print('own bounded worker timed out; inspect last completed checkpoint', flush=True)
            return 3


if __name__ == '__main__':
    sys.exit(main())
