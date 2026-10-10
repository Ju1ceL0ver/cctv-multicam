"""Run explicitly selected pytest checks with CUDA initialization prohibited.

Usage: python door_seg_cpu_test.py tests/test_door_seg_sequence_state.py
Set the explicit -1 sentinel BEFORE torch import. On this Windows CUDA build,
an empty CUDA_VISIBLE_DEVICES did not prevent CPU AdamW from opening CUDA.
This is for authorized CPU Torch tests, not the GPU waiter's daytime check.
"""
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import torch


def forbidden(*args, **kwargs):
    raise RuntimeError('CPU-only test forbids CUDA initialization')


def main():
    if not sys.argv[1:]:
        raise ValueError('explicit CPU test paths/pytest options required')
    if torch.cuda.is_initialized():
        raise RuntimeError('CUDA already initialized before CPU test guard')
    torch.cuda._lazy_init = forbidden
    torch._C._cuda_init = forbidden
    import pytest
    code = pytest.main(['-q', '--color=no', *sys.argv[1:]])
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU test unexpectedly initialized CUDA')
    print('CPU guard: CUDA remained uninitialized', flush=True)
    return code


if __name__ == '__main__':
    sys.exit(main())
