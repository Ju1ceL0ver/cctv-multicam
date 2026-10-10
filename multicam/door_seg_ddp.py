"""One door student across GPUs; preserve legacy model/optimizer checkpoint format.

torchrun --standalone --nproc_per_node=2 door_seg_ddp.py train RUN HH:MM
torchrun --standalone --nproc_per_node=2 door_seg_ddp.py smoke RUN
The local loss has the old batch-six scale; DDP averages it across GPUs.
"""
import argparse
import datetime
import json
import math
import os
import queue
import random
import re
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

import door_seg as S


class LossModule(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.yolo = net.yolo
        self.off = net.off

    def forward(self, b, offsets, valid):
        predictions = self.yolo(b['img'].float() / 255)
        losses, items = self.yolo.loss(b, predictions)
        lo = S.offset_loss(self.off(self.net.feat), b, offsets, valid)
        total = losses.sum() + 2 * len(b['img']) * lo
        return total, {**items, 'off': lo.detach()}


def all_finite(value, device):
    flag = torch.tensor(int(bool(value)), dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def shared_flags(flags, device):
    value = torch.tensor(flags, dtype=torch.int64, device=device)
    dist.broadcast(value, src=0)
    return value.tolist()


def train_update(model, net, optimizer, samples, device, precision, on_rejected):
    b, offsets, valid = S.collate(samples, device)
    buffers = {k: v.clone() for k, v in net.yolo.named_buffers()}
    failed = False
    for attempt in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16,
                            enabled=precision == 'bf16' and attempt == 0):
            loss, items = model(b, offsets, valid)
        loss_ok = all_finite(torch.isfinite(loss), device)
        # Both ranks finish DDP's backward, including a rejected graph. Neither
        # updates parameters until every rank agrees that loss AND grads are finite.
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10., error_if_nonfinite=False)
        grads_ok = all_finite(torch.isfinite(norm), device)
        if loss_ok and grads_ok:
            optimizer.step()
            return True, 'fp32' if failed else precision, items
        with torch.no_grad():
            for k, v in net.yolo.named_buffers():
                v.copy_(buffers[k])
        if not failed:
            on_rejected('nonfinite loss/gradient on at least one rank')
        failed = True
        loss = items = None
        net.feat = None
        if attempt == 1 or precision == 'fp32':
            optimizer.zero_grad(set_to_none=True)
            return False, precision, {}
    raise AssertionError('unreachable')


def save(net, optimizer, step, precision, path, evaluation=None, world=2, per_gpu=6):
    state = {'model': net.state(), 'opt': optimizer.state_dict(), 'step': step,
             'precision': precision, 'parallel': {'strategy': 'ddp', 'world_size': world,
             'batch_per_gpu': per_gpu, 'global_batch': per_gpu * world,
             'loss_scale': 'mean of legacy local batches; unchanged lr'}}
    if evaluation is not None:
        state['eval'] = evaluation
    S.checkpoint_finite(state)
    tmp = path.with_suffix('.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def samples_from(q, workers):
    while True:
        try:
            return q.get(timeout=30)
        except queue.Empty:
            if not all(p.is_alive() for p in workers):
                raise RuntimeError('A DDP data worker died; refusing to continue')


def main(args):
    rank, world = int(os.environ['RANK']), int(os.environ['WORLD_SIZE'])
    local = int(os.environ['LOCAL_RANK'])
    if world != 2:
        raise ValueError('This candidate is approved for exactly two GPUs')
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    torch.set_num_threads(2)
    S.cv2.setNumThreads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=30))
    run = S.ROOT / 'runs' / args.run
    workers = []
    try:
        net = S.DoorSeg(device)
        state = torch.load(run / 'last.pt', map_location='cpu', weights_only=False)
        S.checkpoint_finite(state)
        assert 'opt' in state, 'Finite optimizer required for resume'
        net.load(state['model'])
        step = int(state['step'])
        optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        optimizer.load_state_dict(state['opt'])
        precision = state.get('precision', 'bf16')
        if precision not in ('bf16', 'fp32'):
            raise ValueError('Unsupported checkpoint precision')
        del state
        net.yolo.train()
        net.off.train()
        net.yolo.criterion = net.yolo.init_criterion()
        for _ in range(step // 2000):
            net.yolo.criterion.update()
        model = DistributedDataParallel(LossModule(net), device_ids=[local],
                                        find_unused_parameters=True, broadcast_buffers=True)
        if args.command == 'smoke':
            rng = random.Random(7 + rank)
            src = S.Source(S.sources()[0])
            xs = []
            while len(xs) < args.per_gpu:
                x = S.sample(src, rng)
                if x is not None:
                    xs.append(x)
            b, off, valid = S.collate(xs, device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=precision == 'bf16'):
                loss, items = model(b, off, valid)
            assert all_finite(torch.isfinite(loss), device), 'Nonfinite DDP smoke loss'
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 10., error_if_nonfinite=False)
            assert all_finite(torch.isfinite(norm), device), 'Nonfinite DDP smoke gradients'
            norms = [torch.zeros_like(norm) for _ in range(world)]
            dist.all_gather(norms, norm)
            assert all(torch.allclose(norms[0], n, rtol=1e-5, atol=1e-6) for n in norms)
            print(json.dumps({'ddp_smoke': 'finite backward; no optimizer update',
                              'rank': rank, 'gpu': torch.cuda.get_device_name(local),
                              'per_gpu': args.per_gpu, 'global_batch': args.per_gpu * world,
                              'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                              'peak_gb': torch.cuda.max_memory_allocated() / 1e9}), flush=True)
            src.close()
            return
        if rank == 0:
            S.say(run, 'DDP resume step%d: weights AND optimizer restored; %d GPUs, batch%d/%d global; lr1e-4' %
                  (step, world, args.per_gpu, args.per_gpu * world))
        text = (run / 'progress.md').read_text(encoding='utf-8')
        evals = [(int(m[1]), json.loads(m[2])['f1']) for m in
                 re.finditer(r'шаг (\d+): против SAM 3.1 на 23.09 (\{.*\})', text) if int(m[1]) <= step]
        hh, mm = map(int, args.stop.split(':'))
        now = time.localtime()
        end = time.time() + ((hh * 60 + mm) - (now.tm_hour * 60 + now.tm_min)) % 1440 * 60
        context = torch.multiprocessing.get_context('spawn')
        q = context.Queue(maxsize=64)
        workers = [context.Process(target=S.stream, args=(1000 * step + 100 * rank + i, q),
                   daemon=True) for i in range(args.workers)]
        for worker in workers:
            worker.start()
        pool = []
        acc = {}
        saved_at = time.time()
        last_eval = evals[-1][0] if evals else step
        criterion_updates = step // 2000
        stop_reason = None
        while True:
            stop = int(time.time() >= end or (run / 'STOP').exists() or (run / 'PAUSE').exists())
            if shared_flags([stop if rank == 0 else 0], device)[0]:
                break
            while len(pool) < 96:
                pool.append(samples_from(q, workers))
            xs = []
            for _ in range(args.per_gpu):
                i = random.randrange(len(pool))
                xs.append(pool[i])
                pool[i] = samples_from(q, workers)
            lr_factor = min(1., (step + 1) / 500) * (.1 + .9 * .5 *
                        (1 + math.cos(math.pi * min(1., step / 60000))))
            for group in optimizer.param_groups:
                group['lr'] = 1e-4 * lr_factor

            def rejected(reason):
                torch.save({'samples': xs, 'step': step, 'rank': rank,
                            'world_size': world, 'reason': reason},
                           run / ('rejected_%d_rank%d.pt' % (step, rank)))
                if rank == 0:
                    S.say(run, 'шаг %d rejected on DDP before optimizer: %s' % (step, reason))
                    save(net, optimizer, step, precision, run / ('before_rejected_%d.pt' % step),
                         world=world, per_gpu=args.per_gpu)
                dist.barrier()

            ok, new_precision, items = train_update(model, net, optimizer, xs, device, precision, rejected)
            if not ok:
                stop_reason = 'nonfinite DDP batch in FP32 at step%d' % step
                if rank == 0:
                    (run / 'STOP').write_text(stop_reason, encoding='utf-8')
                break
            if new_precision != precision and rank == 0:
                S.say(run, 'DDP same batch finite in FP32; subsequent steps use FP32')
            precision = new_precision
            step += 1
            while step // 2000 > criterion_updates:
                net.yolo.criterion.update()
                criterion_updates += 1
            keys = sorted(items)
            values = torch.stack([items[k].detach().float() for k in keys])
            dist.all_reduce(values)
            if rank == 0:
                for k, v in zip(keys, (values / world).tolist()):
                    acc[k] = acc.get(k, 0.) + v
                acc['n'] = acc.get('n', 0) + 1
                if step % 100 == 0:
                    result = {k: round(v / acc['n'], 4) for k, v in acc.items() if k != 'n'}
                    result.update(step=step, lr=round(1e-4 * lr_factor, 7), t=round(time.time()),
                                  global_batch=args.per_gpu * world)
                    with (run / 'log.jsonl').open('a') as f:
                        f.write(json.dumps(result) + '\n')
                    acc = {}
            do_save, do_eval = shared_flags([int(time.time() - saved_at > 600),
                                           int(step - last_eval >= 2000)] if rank == 0 else [0, 0], device)
            if do_eval or do_save:
                dist.barrier()
                if rank == 0:
                    evaluation = S.evaluate(net) if do_eval else None
                    save(net, optimizer, step, precision, run / 'last.pt',
                         world=world, per_gpu=args.per_gpu)
                    if do_eval:
                        S.say(run, 'шаг %d: против SAM 3.1 на 23.09 %s' % (step, json.dumps(evaluation)))
                        save(net, optimizer, step, precision, run / ('step%d.pt' % step), evaluation,
                             world=world, per_gpu=args.per_gpu)
                        evals.append((step, evaluation['f1']))
                        if S.plateau(evals):
                            (run / 'STOP').write_text('plateau: %s' % evals, encoding='utf-8')
                dist.barrier()
                saved_at = time.time()
                if do_eval:
                    last_eval = step
        if rank == 0:
            save(net, optimizer, step, precision, run / 'last.pt', world=world, per_gpu=args.per_gpu)
            if not (run / 'STOP').exists() and not (run / 'PAUSE').exists():
                evaluation = S.evaluate(net)
                S.say(run, 'стоп на шаге %d: против SAM 3.1 на 23.09 %s' % (step, json.dumps(evaluation)))
            else:
                S.say(run, 'DDP stopped safely at step%d; %s' % (step, stop_reason or 'STOP/PAUSE'))
        dist.barrier()
    finally:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=10)
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train', 'smoke'])
    parser.add_argument('run')
    parser.add_argument('stop', nargs='?', default='20:50')
    parser.add_argument('--per-gpu', type=int, default=6)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.per_gpu != 6:
        raise ValueError('Start with the approved local batch6; measure before increasing it')
    main(args)
