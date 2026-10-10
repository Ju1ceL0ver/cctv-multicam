"""CPU AdamW continuation and failure isolation for separate sequence runs."""
import copy
import hashlib
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'multicam'))
from door_seg_temporal import TemporalHeads
import door_seg_sequence_state as Q


class Mini:
    def __init__(self):
        self.yolo = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3))
        self.off = nn.Linear(3, 2)
        self.temporal = TemporalHeads(3, hidden=2, embedding=2)
        self.memory = None

    def parameters(self):
        return list(self.yolo.parameters())+list(self.off.parameters())+list(self.temporal.parameters())

    def state(self):
        return {'yolo': self.yolo.state_dict(), 'off': self.off.state_dict(),
                'temporal': self.temporal.state_dict(), 'temporal_arch': self.temporal.arch}

    def load(self, state):
        self.yolo.load_state_dict(state['yolo'])
        self.off.load_state_dict(state['off'])
        if 'temporal' in state:
            self.temporal.load_state_dict(state['temporal'])
        self.memory = None


def update(opt):
    opt.zero_grad()
    # Nonconstant gradients depend on previous updates, exposing a lost/misbound Adam history.
    loss = sum((p.square()+.13*p).sum() for group in opt.param_groups for p in group['params'])
    loss.backward()
    opt.step()


def legacy():
    net = Mini()
    base = list(net.yolo.parameters())+list(net.off.parameters())
    opt = torch.optim.AdamW(base, lr=3e-4, betas=(.81,.93), eps=1e-7, weight_decay=.017)
    for _ in range(4):
        update(opt)
    state = {'model': {k:v for k,v in net.state().items() if k in ('yolo','off')},
             'opt': opt.state_dict(), 'step': 17787, 'precision': 'fp32'}
    return net, opt, copy.deepcopy(state)


def same_parameters(a, b, base_only=False):
    aa = Q.named_groups(a)
    bb = Q.named_groups(b)
    groups = 1 if base_only else 2
    for ga,gb in zip(aa[:groups],bb[:groups]):
        assert [n for n,_ in ga] == [n for n,_ in gb]
        assert all(torch.equal(pa,pb) for (_,pa),(_,pb) in zip(ga,gb))


def test_legacy_migration_preserves_adam_moments_and_next_base_update():
    torch.manual_seed(8)
    reference, control, state = legacy()
    candidate = Mini()
    opt, info = Q.restore(candidate, state, head_lr=2e-5)
    assert info['precision']=='fp32' and info['step']==17787 and info['criterion_updates']==8
    assert info['legacy_migration'] and len(opt.param_groups)==2
    assert opt.param_groups[0]['lr']==3e-4 and opt.param_groups[0]['betas']==(.81,.93)
    assert opt.param_groups[0]['weight_decay']==.017 and opt.param_groups[1]['lr']==2e-5
    for pa,pb in zip(control.param_groups[0]['params'],opt.param_groups[0]['params']):
        assert torch.equal(control.state[pa]['exp_avg'],opt.state[pb]['exp_avg'])
        assert torch.equal(control.state[pa]['exp_avg_sq'],opt.state[pb]['exp_avg_sq'])
        assert float(opt.state[pb]['step'])==4
    assert all(p not in opt.state for p in opt.param_groups[1]['params'])
    update(control); update(opt)
    same_parameters(reference,candidate,base_only=True)
    assert all(float(opt.state[p]['step'])==1 for p in opt.param_groups[1]['params'])
    assert not torch.cuda.is_initialized()


def test_temporal_named_checkpoint_restores_both_groups_next_update(tmp_path):
    _,_,old = legacy()
    net = Mini(); opt,_ = Q.restore(net,old)
    update(opt)
    net.memory = torch.randn(2,3)  # Ephemeral recurrent state must not enter a checkpoint.
    path=tmp_path/'doorseg_sequence_cpu_test'/'last.pt'
    Q.save(net,opt,path,step=17788,precision='fp32',recipe={'version':1,'train_manifest_sha256':'a'*64})
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert 'memory' not in state['model'] and state['sequence_recipe']['version']==1
    resumed=Mini(); restored,info=Q.restore(resumed,state)
    assert not info['legacy_migration'] and resumed.memory is None
    update(opt);update(restored)
    same_parameters(net,resumed)
    assert not list(path.parent.glob('*.tmp'))


def test_nonfinite_rejection_does_not_mutate_target_or_replace_good_file(tmp_path):
    _,_,old=legacy()
    target=Mini(); before=copy.deepcopy(target.state())
    key=old['opt']['param_groups'][0]['params'][0]
    old['opt']['state'][key]['exp_avg'].flatten()[0]=float('inf')
    with pytest.raises(ValueError,match='nonfinite'):
        Q.restore(target,old)
    assert all(torch.equal(v,before['yolo'][k]) for k,v in target.yolo.state_dict().items())
    _,_,old=legacy();opt,_=Q.restore(target,old)
    path=tmp_path/'doorseg_sequence_cpu_test'/'last.pt'
    recipe={'version':1,'train_manifest_sha256':'a'*64}
    Q.save(target,opt,path,step=17787,precision='fp32',recipe=recipe)
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    with torch.no_grad():
        target.temporal.residual.bias[0]=float('nan')
    with pytest.raises(ValueError,match='nonfinite'):
        Q.save(target,opt,path,step=17788,precision='fp32',recipe=recipe)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    assert not list(path.parent.glob('*.tmp'))


def test_name_order_shape_precision_and_output_guards(tmp_path):
    _,_,old=legacy()
    bad=copy.deepcopy(old);bad['model']['yolo']=dict(reversed(list(bad['model']['yolo'].items())))
    with pytest.raises(ValueError,match='order'):
        Q.restore(Mini(),bad)
    bad=copy.deepcopy(old);key=bad['opt']['param_groups'][0]['params'][0]
    bad['opt']['state'][key]['exp_avg']=torch.ones(1)
    with pytest.raises(ValueError,match='shape'):
        Q.restore(Mini(),bad)
    bad=copy.deepcopy(old);bad['model']['yolo']['1.running_mean']=torch.ones(1)
    target=Mini();before=copy.deepcopy(target.state())
    with pytest.raises(ValueError,match='buffer shape'):
        Q.restore(target,bad)
    assert all(torch.equal(v,before['yolo'][k]) for k,v in target.yolo.state_dict().items())
    bad=copy.deepcopy(old);bad.pop('precision')
    with pytest.raises(ValueError,match='precision'):
        Q.restore(Mini(),bad)
    net=Mini();opt,_=Q.restore(net,old)
    with pytest.raises(ValueError,match='separate'):
        Q.save(net,opt,tmp_path/'doorseg_c_background'/'last.pt',step=17787,precision='fp32',recipe={})
    path=tmp_path/'doorseg_sequence_cpu_test'/'last.pt'
    Q.save(net,opt,path,step=17787,precision='fp32',recipe={'version':1,'train_manifest_sha256':'a'*64})
    bad=torch.load(path,map_location='cpu',weights_only=False)
    bad['optimizer_names'][1][0]='temporal.renamed'
    with pytest.raises(ValueError,match='names'):
        Q.restore(Mini(),bad)
