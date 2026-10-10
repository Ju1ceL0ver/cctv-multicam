"""Two real CPU ranks test averaging and cross-rank nonfinite rejection."""
import json
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'multicam'))
import door_seg_ddp as D
from door_seg_ddp import all_finite, shared_flags


def worker(rank, init, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=init, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad(): model.weight.fill_(1.)
        ddp = DistributedDataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        x = torch.tensor([[1. + rank], [3. + rank]])
        loss = ddp(x).square().mean()
        loss.backward()
        # DDP averages the mean local losses. A shared batch is the reference.
        reference = torch.tensor([[1.], [3.], [2.], [4.]])
        expected = 2 * reference.square().mean()
        assert torch.allclose(model.weight.grad.squeeze(), expected)
        assert all_finite(torch.isfinite(model.weight.grad).all(), 'cpu')
        optimizer.step()
        good = model.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        loss = ddp(x).sum() * (float('nan') if rank == 1 else 1.)
        loss.backward()
        finite = all_finite(torch.isfinite(loss), 'cpu')
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=False)
        grads = all_finite(torch.isfinite(norm), 'cpu')
        if finite and grads: optimizer.step()
        assert torch.equal(good, model.weight)
        assert shared_flags([1, 0] if rank == 0 else [0, 1], 'cpu') == [1, 0]
        # Exercise the actual update loop: rank one's bad loss must leave both
        # models, Adam state and pre-forward BN running statistics unchanged.
        guarded = GuardedModel(rank)
        wrapped = DistributedDataParallel(guarded)
        net = GuardedNet(guarded)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
        before = {k: v.clone() for k, v in guarded.state_dict().items()}
        D.S.collate = lambda samples, device: ({'img': x}, None, None)
        rejections = []
        ok, _, _ = D.train_update(wrapped, net, opt, [], 'cpu', 'fp32', rejections.append)
        assert not ok and len(rejections) == 1 and not opt.state
        assert all(torch.equal(v, before[k]) for k, v in guarded.state_dict().items())
        Path(output, str(rank)+'.json').write_text(json.dumps(dict(gradient=float(expected),
            rejected=not finite and not grads, optimizer_step=float(optimizer.state[model.weight]['step']))))
    finally:
        dist.destroy_process_group()


class GuardedModel(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.bn = torch.nn.BatchNorm1d(1)
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, batch, offsets, valid):
        loss = self.linear(self.bn(batch['img'])).square().mean()
        loss = loss * (float('nan') if self.rank == 1 else 1.)
        return loss, {'loss': loss.detach()}


class GuardedNet:
    def __init__(self, model):
        self.yolo = model
        self.feat = None

    def parameters(self):
        return list(self.yolo.parameters())


def test_ddp_means_gradients_and_rejects_bad_rank_before_optimizer(tmp_path):
    mp.spawn(worker, args=('file://' + str(tmp_path/'init'), str(tmp_path)), nprocs=2, join=True)
    results = [json.loads((tmp_path/(str(i)+'.json')).read_text()) for i in range(2)]
    assert results[0] == results[1] == dict(gradient=15., rejected=True, optimizer_step=1.)
