"""Resume/checkpoint tools for a SEPARATE temporal experiment, never C.

Legacy AdamW has one positional group (YOLO then offset). Validate that order
against saved model keys and shapes before appending new temporal parameters.
New checkpoints record names explicitly and reject architecture/order drift.
This module does not start a trainer, sample data or open CUDA.
"""
import math
import os
from pathlib import Path
import tempfile

import torch


def named_groups(net):
    base = [('yolo.'+name, p) for name, p in net.yolo.named_parameters()]
    base += [('off.'+name, p) for name, p in net.off.named_parameters()]
    heads = [('temporal.'+name, p) for name, p in net.temporal.named_parameters()]
    all_params = base + heads
    if len({name for name, _ in all_params}) != len(all_params) or len({id(p) for _, p in all_params}) != len(all_params):
        raise ValueError('duplicate parameter name or shared parameter')
    if [id(p) for _, p in all_params] != [id(p) for p in net.parameters()]:
        raise ValueError('unexpected temporal parameter order')
    return base, heads


def finite(value, path='checkpoint'):
    if torch.is_tensor(value):
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError('nonfinite '+path)
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, path+'.'+str(key))
    elif isinstance(value, (tuple, list)):
        for i, item in enumerate(value):
            finite(item, path+'.'+str(i))
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError('nonfinite '+path)


def metadata(state):
    if type(state.get('step')) is not int or state['step'] < 0:
        raise ValueError('nonnegative integer source step required')
    if state.get('precision') not in ('bf16', 'fp32'):
        raise ValueError('saved BF16/FP32 precision required; no implicit reset')
    if not isinstance(state.get('model'), dict) or not isinstance(state.get('opt'), dict):
        raise ValueError('model AND optimizer required')
    finite(state)


def validate_adam(saved, names, parameters):
    if len(saved['param_groups']) != len(names):
        raise ValueError('optimizer group count differs')
    used = set()
    for group, expected, params in zip(saved['param_groups'], names, parameters):
        keys = group['params']
        if len(keys) != len(expected):
            raise ValueError('optimizer parameter count differs')
        for key, param in zip(keys, params):
            if key in used:
                raise ValueError('duplicate optimizer parameter ID')
            used.add(key)
            moment = saved['state'].get(key)
            if moment is None:  # Frozen/unused parameters legitimately have no Adam state.
                continue
            for name in ('exp_avg', 'exp_avg_sq'):
                if name not in moment or moment[name].shape != param.shape:
                    raise ValueError('optimizer moment shape differs: '+name)
            if 'max_exp_avg_sq' in moment and moment['max_exp_avg_sq'].shape != param.shape:
                raise ValueError('AMSGrad moment shape differs')
            step = moment.get('step')
            if not torch.is_tensor(step) or step.numel() != 1 or float(step) < 0:
                raise ValueError('invalid Adam successful-step counter')
    if set(saved['state']) - used:
        raise ValueError('orphan optimizer state')


def validate_model(net, model, legacy):
    blocks = ('yolo', 'off') if legacy else ('yolo', 'off', 'temporal')
    for block in blocks:
        current = getattr(net, block).state_dict()
        if set(current) != set(model[block]):
            raise ValueError('model parameter/buffer keys differ: '+block)
        for name, value in current.items():
            other = model[block][name]
            if not torch.is_tensor(other) or value.shape != other.shape:
                raise ValueError('model parameter/buffer shape differs: '+block+'.'+name)


def restore(net, state, *, head_lr=1e-4, weight_decay=1e-4):
    """Return AdamW and resume metadata; do not execute an optimizer update.

    Preserve the source precision and global successful step. Criterion decay
    remains caller-owned (step//2000). A source without optimizer is rejected.
    Legacy order is accepted ONLY when model parameter key order agrees.
    """
    if not math.isfinite(head_lr) or head_lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError('finite positive head LR and nonnegative decay required')
    metadata(state)
    base, heads = named_groups(net)
    base_names, head_names = [n for n, _ in base], [n for n, _ in heads]
    base_params, head_params = [p for _, p in base], [p for _, p in heads]
    legacy = 'temporal' not in state['model']
    names = [base_names] if legacy else [base_names, head_names]
    parameters = [base_params] if legacy else [base_params, head_params]
    if legacy:
        known = set(base_names)
        saved_order = [block+'.'+name for block in ('yolo', 'off')
                       for name in state['model'][block] if block+'.'+name in known]
        if saved_order != base_names:
            raise ValueError('legacy model parameter order differs; unsafe positional resume')
        for name, p in base:
            block, key = name.split('.', 1)
            if state['model'][block][key].shape != p.shape:
                raise ValueError('legacy model parameter shape differs: '+name)
        if state.get('optimizer_names') is not None and state['optimizer_names'] != names:
            raise ValueError('legacy optimizer names differ')
    elif state.get('optimizer_names') != names:
        raise ValueError('temporal optimizer parameter names/order differ')
    validate_adam(state['opt'], names, parameters)
    if not legacy and state['model'].get('temporal_arch') != net.temporal.arch:
        raise ValueError('temporal architecture differs')
    validate_model(net, state['model'], legacy)
    groups = [dict(params=base_params, **{k: v for k, v in state['opt']['param_groups'][0].items() if k != 'params'})]
    if legacy:
        opt = torch.optim.AdamW(groups)
        opt.load_state_dict(state['opt'])
        opt.add_param_group(dict(params=head_params, lr=head_lr, weight_decay=weight_decay))
    else:
        groups.append(dict(params=head_params, **{k: v for k, v in state['opt']['param_groups'][1].items() if k != 'params'}))
        opt = torch.optim.AdamW(groups)
        opt.load_state_dict(state['opt'])
    # Loading model happens only after metadata, moments, buffers and optimizer
    # hyperparameters pass gates; malformed buffers cannot partially change it.
    net.load(state['model'])
    return opt, {'step': state['step'], 'precision': state['precision'], 'legacy_migration': legacy,
                 'criterion_updates': state['step']//2000, 'base_parameters': len(base), 'new_parameters': len(heads)}


def save(net, opt, path, *, step, precision, recipe):
    """Atomic finite checkpoint, restricted to runs named doorseg_sequence_*.

    The explicit recipe must document data split/loss; storing this artifact
    does not enable temporal heads in C or demonstrate training quality.
    """
    path = Path(path)
    if not path.parent.name.startswith('doorseg_sequence_'):
        raise ValueError('separate doorseg_sequence_* output directory required')
    if not isinstance(recipe, dict) or not recipe.get('train_manifest_sha256') or not recipe.get('version'):
        raise ValueError('versioned sequence recipe and TRAIN manifest SHA required')
    groups = named_groups(net)
    if len(opt.param_groups) != 2 or any([id(p) for p in group['params']] != [id(p) for _, p in expected]
                                       for group, expected in zip(opt.param_groups, groups)):
        raise ValueError('optimizer does not match named model groups')
    state = {'model': net.state(), 'opt': opt.state_dict(), 'step': step, 'precision': precision,
             'optimizer_names': [[n for n, _ in group] for group in groups], 'sequence_recipe': recipe}
    metadata(state)
    validate_adam(state['opt'], state['optimizer_names'], [[p for _, p in group] for group in groups])
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name+'.', suffix='.tmp', delete=False) as handle:
            tmp = Path(handle.name)
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        tmp = None
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
