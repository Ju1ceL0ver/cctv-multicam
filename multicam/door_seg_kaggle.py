"""Foreground private Kaggle job: isolated CUDA environment, one two-GPU student.

Credentials and frozen manifest are supplied separately, outside this source.
Dataset lives on /kaggle/temp; only model/log artifacts use /kaggle/working.
The committed kernel must wait for this process, not detach it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main(args):
    root = Path(args.root)
    run = root / 'runs' / args.run
    run.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    os.chmod(args.access, 0o600)
    sys.path.insert(0, str(root))
    from door_seg_colab import Transfer
    transfer = Transfer(args.access, root)

    def status(phase, **extra):
        path = run / 'colab_status.json'
        path.write_text(json.dumps(dict(phase=phase, session='kaggle-c',
            pid=os.getpid(), utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **extra)))
        transfer.put(path)
        print('Kaggle phase:', phase, flush=True)

    # Keep the shop supervisor blocked throughout preparation and training.
    marker = run / 'COLAB_ACTIVE.json'
    marker.write_text(json.dumps(dict(active=True, session='kaggle-c', run=args.run)))
    transfer.put(marker)
    env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONIOENCODING='utf-8',
               OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    env.pop('PYTHONPATH', None)
    env.pop('LD_LIBRARY_PATH', None)
    try:
        status('installing', global_batch=12, gpus=2, checkpoint_step=8996)
        venv = Path('/kaggle/temp/doorseg_venv')
        if not (venv / 'bin/pip').exists():
            # Kaggle's system Python lacks ensurepip. virtualenv ships its own
            # pip seed wheels; installing it to temp leaves the base image intact.
            tools = Path('/kaggle/temp/doorseg_envtools')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
                                   '--target', str(tools), 'virtualenv'], env=env)
            subprocess.check_call([sys.executable, '-m', 'virtualenv', '--no-download', str(venv)],
                                  env=dict(env, PYTHONPATH=str(tools)))
        python = str(venv / 'bin/python')
        # Fresh venv avoids overlapping CUDA12/CUDA13 cuDNN distributions.
        manifest = json.loads(Path(args.manifest).read_text())
        packages = [name + '==' + version.split('+')[0]
                    for name, version in manifest['versions'].items()]
        packages += ['requests', 'psutil', 'joblib', 'scikit-learn']
        with (run / 'kaggle_install.log').open('a') as log:
            installer = subprocess.Popen([python, '-m', 'pip', 'install', '--no-cache-dir', *packages],
                                         env=env, stdout=log, stderr=subprocess.STDOUT)
            while True:
                try:
                    code = installer.wait(timeout=120)
                    transfer.put(run / 'kaggle_install.log')
                    if code:
                        raise RuntimeError('Pinned dependency installation failed; inspect kaggle_install.log')
                    break
                except subprocess.TimeoutExpired:
                    transfer.put(run / 'kaggle_install.log')
                    status('installing', installer_pid=installer.pid, gpus=2, global_batch=12)
        gate = """import json, torch, torchvision, ultralytics
assert torch.__version__ == '2.13.0+cu130', torch.__version__
assert torchvision.__version__ == '0.28.0+cu130', torchvision.__version__
assert ultralytics.__version__ == '8.4.108', ultralytics.__version__
assert torch.cuda.device_count() == 2
assert torch.distributed.is_nccl_available()
for i in range(2):
    torch.cuda.set_device(i)
    assert torch.cuda.is_bf16_supported()
    conv = torch.nn.Conv2d(17, 8, 3).cuda(i)
    x = torch.randn(6, 17, 64, 64, device='cuda:%d' % i)
    with torch.autocast('cuda', dtype=torch.bfloat16): y=conv(x).square().mean()
    y.backward()
    assert torch.isfinite(y) and all(torch.isfinite(p.grad).all() for p in conv.parameters())
print(json.dumps(dict(torch=torch.__version__, GPUs=[torch.cuda.get_device_name(i) for i in range(2)],
                     status='finite BF16 convolution/backward on both GPUs')))
"""
        result = subprocess.run([python, '-c', gate], env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (run / 'kaggle_gpu_gate.log').write_text(result.stdout)
        transfer.put(run / 'kaggle_gpu_gate.log')
        if result.returncode:
            raise RuntimeError('Pinned environment GPU gate failed; inspect kaggle_gpu_gate.log')
        status('starting_runner', checkpoint_step=manifest['checkpoint_step'], gpus=2,
               batch_per_gpu=6, global_batch=12)
        command = [python, '-u', str(root / 'door_seg_colab.py'), '--access', args.access,
                   '--manifest', args.manifest, '--root', str(root), '--run', args.run,
                   '--session', 'kaggle-c', '--ddp', '--deadline-epoch', str(args.deadline_epoch)]
        with (run / 'kaggle_runner.log').open('a') as log:
            subprocess.check_call(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
    except Exception as exc:
        # Retain cloud ownership for explicit recovery; never start a duplicate.
        status('kaggle_failed', error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        # Kernel output excludes access JSON, code bootstrap and the video dataset.
        for path in run.iterdir():
            if path.is_file() and (path.suffix in ('.pt', '.log', '.md', '.json', '.jsonl')
                                   or path.name == 'STOP'):
                shutil.copy2(path, output / path.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--access', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', default='/kaggle/temp/doorseg')
    parser.add_argument('--run', default='doorseg_c_background')
    parser.add_argument('--output', default='/kaggle/working/doorseg_c_background')
    parser.add_argument('--deadline-epoch', required=True, type=float)
    main(parser.parse_args())
