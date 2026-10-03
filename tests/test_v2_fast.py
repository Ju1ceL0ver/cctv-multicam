"""v2_fast.py against the code it replaces (tests/ref_train_v2.py, ref_train_slots.py = the files before the rewrite)."""
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'multicam'
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip('torch')
import torch.nn as nn  # noqa: E402

import slot_v2 as V  # noqa: E402
import train_slots as TS  # noqa: E402
import train_v2  # noqa: E402
import v2_fast as VF  # noqa: E402


def load_ref(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / (name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


REF = load_ref('ref_train_v2')
REFS = load_ref('ref_train_slots')
DEV = 'cpu'


def make_masks(n, h=48, w=80, seed=0):
    g = torch.Generator().manual_seed(seed)
    m = torch.zeros(n, h, w)
    boxes = []
    for i in range(n):
        y0 = int(torch.randint(2, h - 20, (1,), generator=g)); x0 = int(torch.randint(2, w - 25, (1,), generator=g))
        hh = int(torch.randint(8, 18, (1,), generator=g)); ww = int(torch.randint(6, 20, (1,), generator=g))
        m[i, y0:y0 + hh, x0:x0 + ww] = 1
        boxes.append([(x0 + ww / 2) / w, (y0 + hh / 2) / h, ww / w, hh / h])
    return m, torch.tensor(boxes, dtype=torch.float32)


def test_centre_targets_match_the_loop():
    for n in (0, 1, 5):
        m, boxes = make_masks(max(n, 1), seed=n)
        m, boxes = m[:n], boxes[:n]
        tg = {'masks': m, 'boxes': boxes}
        a = REF.centre_targets(tg, 12, 20, DEV)
        b = VF.centre_targets(tg, 12, 20, DEV)
        for x, y in zip(a, b):
            assert torch.allclose(x, y, atol=1e-5), (n, (x - y).abs().max())


def test_centre_targets_skip_empty_masks():
    m, boxes = make_masks(3)
    m[1] = 0
    tg = {'masks': m, 'boxes': boxes}
    a = REF.centre_targets(tg, 12, 20, DEV)
    b = VF.centre_targets(tg, 12, 20, DEV)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5)


def rand_out(Ly, S, C, seed=0, dn=False):
    g = torch.Generator().manual_seed(seed)
    return [{'obj': torch.randn(1, S, generator=g), 'box': torch.rand(1, S, 4, generator=g) * 0.5 + 0.25,
             'mask_vec': torch.randn(1, S, C, generator=g)} for _ in range(Ly)]


@pytest.mark.parametrize('n', [0, 1, 4])
def test_stacked_set_losses_equal_the_mean_of_the_layers(n):
    C, S, H, W = 16, 20, 40, 60
    pix = torch.randn(1, C, H, W)
    m, boxes = make_masks(max(n, 1), H, W, seed=3)
    tg = {'masks': m[:n], 'boxes': boxes[:n]}
    outs = rand_out(5, S, C, seed=n)
    rows = torch.tensor(random.Random(n).sample(range(S), n), dtype=torch.long)
    cols = torch.arange(n)
    TS.LOSS_POINTS = REFS.LOSS_POINTS = 4096
    per = [REFS.set_losses(o, pix, [tg], [(rows, cols)]) for o in outs]
    mean = {k: sum(p[k] for p in per) / len(per) for k in per[0]}
    got = VF.set_losses_stack(outs, pix, tg, rows, cols, TS)
    for k in ('l1', 'giou'):                     # no randomness in them: exact
        assert torch.allclose(mean[k], got[k], atol=1e-5), k
    for k in ('obj', 'bce', 'dice'):             # random points: close
        assert torch.allclose(mean[k], got[k], rtol=0.06, atol=0.02), (k, float(mean[k]), float(got[k]))


def test_stacked_set_losses_backward_reaches_every_layer():
    C, S, H, W = 16, 20, 40, 60
    pix = torch.randn(1, C, H, W, requires_grad=True)
    m, boxes = make_masks(3, H, W)
    tg = {'masks': m, 'boxes': boxes}
    outs = rand_out(4, S, C)
    for o in outs:
        for k in o:
            o[k].requires_grad_(True)
    TS.LOSS_POINTS = 1024
    tot = sum(VF.set_losses_stack(outs, pix, tg, torch.tensor([2, 5, 9]), torch.arange(3), TS).values())
    tot.backward()
    assert all(o['mask_vec'].grad is not None and o['mask_vec'].grad.abs().sum() > 0 for o in outs)
    assert pix.grad.abs().sum() > 0


def test_giou_pair_is_the_diagonal_of_the_pairwise():
    a = torch.rand(6, 4); b = torch.rand(6, 4)
    a = TS.box_xyxy(a); b = TS.box_xyxy(b)
    assert torch.allclose(torch.diag(TS.giou(a, b)), VF.giou_pair(a, b), atol=1e-6)


def test_make_dn_has_the_same_layout():
    m, boxes = make_masks(3)
    tgs = [{'boxes': boxes}, {'boxes': boxes[:2]}]
    torch.manual_seed(0)
    a, am = REFS.make_dn(tgs, DEV)
    torch.manual_seed(0)
    b, bm = VF.make_dn(tgs, DEV, TS.DN_MAX)
    assert a['box'].shape == b['box'].shape and torch.equal(a['group'], b['group'])
    for (ar, ac), (br, bc) in zip(am, bm):
        assert torch.equal(ar, br) and torch.equal(ac, bc)
    n_max = 3
    groups = TS.DN_MAX // (2 * n_max)
    bx = b['box'].view(2, groups, 2 * n_max, 4)
    # positives are near their person (within 20 % of the size), negatives 40-100 % away
    for bi, tb in ((0, boxes), (1, boxes[:2])):
        n = len(tb)
        pos = bx[bi, :, :n]
        assert ((pos[..., :2] - tb[None, :, :2]).abs() <= 0.2 * tb[None, :, 2:] + 1e-4).all()
        neg = bx[bi, :, n_max:n_max + n]
        assert ((neg[..., :2] - tb[None, :, :2]).abs() >= 0.39 * tb[None, :, 2:] - 1e-4).any()
    assert (b['box'][..., :2] >= 0.001).all() and (b['box'][..., 2:] <= 1.0).all()


def np_targets(n, seed=0):
    rng = np.random.default_rng(seed)
    return {'person': np.arange(10, 10 + n), 'xy': rng.uniform(0, 8, (n, 2)).astype(np.float32), 'height': rng.uniform(1.4, 1.9, n).astype(np.float32),
            'place_w': np.where(rng.random(n) > 0.2, 1.0, 0.0).astype(np.float32), 'zone': rng.integers(-1, 3, n).astype(np.int64),
            'hidden': np.array([100, 101]), 'hidden_xy': np.array([[1.0, 2.0], [np.nan, np.nan]], np.float32), 'has_reid': rng.random(n) > 0.3}


def ref_state_place(r, tg_np, rows, cols, states, dev):
    """The state / place part of the old frame_losses, verbatim (loops and all)."""
    import torch.nn.functional as F
    L = {}
    add = lambda k, v: L.__setitem__(k, L.get(k, 0.0) + v)
    st_l, st_t, pl_l = [], [], []
    tg = {k: torch.as_tensor(v) for k, v in tg_np.items()}
    seen = {int(s): int(c) for s, c in zip(rows, cols)}
    for s, c in seen.items():
        st_l.append(r['state'][0, s]); st_t.append(0)
        pw = float(tg['place_w'][c])
        if pw > 0 and torch.isfinite(tg['xy'][c]).all():
            xy = (tg['xy'][c] - torch.tensor(V.XY_C)) / V.XY_S
            z = tg['height'][c] / 2 / V.Z_S
            pl_l.append((r['place'][0, s], xy, z if torch.isfinite(z) else None, pw))
    for s, (k, hi) in states.items():
        if k == 0:
            continue
        st_l.append(r['state'][0, s]); st_t.append(k)
        if k == 1 and hi is not None and torch.isfinite(tg['hidden_xy'][hi]).all():
            xy = (tg['hidden_xy'][hi] - torch.tensor(V.XY_C)) / V.XY_S
            pl_l.append((r['place'][0, s], xy, None, 0.5))
    if st_l:
        add('state', F.cross_entropy(torch.stack(st_l).float(), torch.as_tensor(st_t), weight=torch.tensor([1.0, 2.0, 2.0])))
    if pl_l:
        tot = 0.0
        for p, xy, z, wgt in pl_l:
            p = p.float()
            lv = p[3:6].clamp(-6, 3)
            nll = 0.5 * (((p[:2] - xy) ** 2) / lv[:2].exp() + lv[:2]).sum()
            if z is not None:
                nll = nll + 0.5 * ((p[2] - z) ** 2 / lv[2].exp() + lv[2])
            tot = tot + wgt * nll
        add('place', tot / len(pl_l))
    return L


def test_state_and_place_losses_equal_the_loops():
    S = 12
    g = torch.Generator().manual_seed(1)
    r = {'state': torch.randn(1, S, 3, generator=g), 'place': torch.randn(1, S, 8, generator=g)}
    tg_np = np_targets(5)
    tg = {'np': tg_np}
    rows, cols = [3, 4, 7, 9], [0, 2, 3, 4]
    states = {1: (0, None), 2: (1, 0), 5: (1, 1), 6: (2, None), 8: (2, None), 0: (1, 0)}
    mt = {'rows': rows, 'cols': cols, 'states': states}
    want = ref_state_place(r, tg_np, rows, cols, states, DEV)
    got = VF.state_place_losses(r, tg, mt, DEV, V)
    assert set(want) == set(got)
    for k in want:
        assert torch.allclose(torch.as_tensor(want[k], dtype=torch.float32), got[k], atol=1e-5), k


def test_state_and_place_with_nothing_to_score():
    r = {'state': torch.randn(1, 4, 3), 'place': torch.randn(1, 4, 8)}
    mt = {'rows': [], 'cols': [], 'states': {}}
    assert VF.state_place_losses(r, {'np': np_targets(1)}, mt, DEV, V) == {}


class Heads(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.to_teacher = nn.ModuleList([nn.Linear(V.REID, 20), nn.Linear(V.REID, 12)])
        self.same_net = nn.Sequential(nn.Linear(2 * 2 * V.REID + 8, 8), nn.ReLU(), nn.Linear(8, 1))

    def same(self, a, b, ctx):
        return self.same_net(torch.cat([a, b, ctx], 1))[:, 0]


def make_pooled(n_now, n_bank, seed=0):
    g = torch.Generator().manual_seed(seed)
    def one(i, tick):
        person = i % 3
        return (torch.randn(2 * V.REID, generator=g), torch.randn(20, generator=g), torch.randn(12, generator=g), i % 4 != 3,
                ('tag', 'cam1' if i % 2 == 0 else 'cam2', person), tick, torch.randn(8, generator=g) * 0.1, torch.rand((), generator=g), torch.softmax(torch.randn(3, generator=g), -1))
    now = [one(i, 10) for i in range(n_now)]
    bank = [one(i + 1, 7) for i in range(n_bank)]
    bank = [(p[0].detach(),) + p[1:] for p in bank]
    return now, bank


def test_identity_losses_equal_the_original_when_no_negative_is_dropped():
    now, bank = make_pooled(5, 3)
    model = Heads()
    Lr, Lf = {}, {}
    old_obj = [(p[:7] + (float(p[7]), p[8])) for p in now]
    old_bank = [(p[:7] + (float(p[7]), p[8])) for p in bank]
    REF.ident_losses(model, old_obj, Lr, DEV, old_bank)
    VF.ident_losses(model, now, Lf, DEV, bank, V, random.Random(0))
    assert set(Lr) == set(Lf) and Lr
    for k in Lr:
        assert torch.allclose(Lr[k], Lf[k], atol=1e-4), (k, float(Lr[k]), float(Lf[k]))


def test_identity_losses_with_a_single_person_do_not_break():
    now, bank = make_pooled(1, 0)
    L = {}
    VF.ident_losses(Heads(), now, L, DEV, bank, V, random.Random(0))
    assert 'reid' in L and 'rel' not in L


def test_matching_finds_the_same_pairs_when_the_costs_are_clear():
    # slots whose boxes sit on the people's boxes; padding slots are ignored
    S, C, H, W = 10, 8, 32, 48
    m, boxes = make_masks(3, H, W, seed=5)
    g = torch.Generator().manual_seed(0)
    box = torch.rand(1, S, 4, generator=g) * 0.1 + 0.9
    for k, slot in enumerate((4, 6, 8)):
        box[0, slot] = boxes[k]
    obj = torch.full((1, S), -3.0); obj[0, [4, 6, 8]] = 4.0
    r = {'obj': obj, 'box': box, 'mask_vec': torch.randn(1, S, C, generator=g), 'pix': torch.randn(1, C, H, W, generator=g),
         'pad': torch.tensor([[False] * 2 + [False] * 6 + [True] * 2]), 'tracks': 2}
    tg = {'boxes': boxes, 'masks': m, 'np': {'person': np.array([1, 2, 3]), 'hidden': np.array([], np.int64)}}
    tg_old = {'boxes': boxes, 'masks': m, 'person': torch.tensor([1, 2, 3]), 'hidden': torch.tensor([], dtype=torch.long)}
    torch.manual_seed(0)
    old = REF.match_frame(r, 0, tg_old, {})
    torch.manual_seed(0)
    h = VF.match_launch(r, 0, tg, TS)
    new = VF.match_finish(h, tg, {}, DEV)
    assert sorted(zip(old[0].tolist(), old[1].tolist())) == sorted(zip(new['rows'], new['cols']))
    assert sorted(zip(old[3][0].tolist(), old[3][1].tolist())) == sorted(zip(new['arows'], new['acols']))
    assert new['states'] == old[2]
    assert not new['pad'][:8].any() and new['pad'][8:].all()


def test_matching_keeps_tracks_and_gives_the_rest_to_proposals():
    S, C, H, W = 10, 8, 32, 48
    m, boxes = make_masks(3, H, W, seed=6)
    g = torch.Generator().manual_seed(1)
    box = torch.rand(1, S, 4, generator=g) * 0.1 + 0.9
    for k, slot in enumerate((5, 6, 7)):
        box[0, slot] = boxes[k]
    obj = torch.full((1, S), -3.0); obj[0, [5, 6, 7]] = 4.0
    r = {'obj': obj, 'box': box, 'mask_vec': torch.randn(1, S, C, generator=g), 'pix': torch.randn(1, C, H, W, generator=g),
         'pad': torch.zeros(1, S, dtype=torch.bool), 'tracks': 2}
    tg = {'boxes': boxes, 'masks': m, 'np': {'person': np.array([1, 2, 3]), 'hidden': np.array([9])}}
    h = VF.match_launch(r, 0, tg, TS)
    was = VF.FINAL_ALL
    try:
        VF.FINAL_ALL = False                                        # the MOTR way: proposals only for the free people
        new = VF.match_finish(h, tg, {0: 2, 1: 9}, DEV)             # track 0 holds person 2, track 1 is person 9 (hidden)
        assert (0, 1) in list(zip(new['rows'], new['cols']))
        assert new['states'][1] == (1, 0)
        assert {c for _, c in zip(new['rows'], new['cols'])} == {0, 1, 2}
        assert len(new['rows']) == 3
        VF.FINAL_ALL = True                                         # 29.09: the proposals find everybody, tracked people too
        new = VF.match_finish(h, tg, {0: 2, 1: 9}, DEV)
        pairs = list(zip(new['rows'], new['cols']))
        assert pairs[0] == (0, 1)                                   # the track first, then the proposals
        assert sorted(c for s, c in pairs if s >= 2) == [0, 1, 2]
        assert new['rows'] == new['arows'] and new['cols'] == new['acols']
    finally:
        VF.FINAL_ALL = was


def test_next_tracks_with_the_host_copy_of_keep():
    torch.manual_seed(0)
    model = V.SlotV2()
    B, S = 1, 9
    r = {'q': torch.randn(B, S, V.D), 'box': torch.rand(B, S, 4), 'tracks': 0}
    keep = torch.zeros(B, S, dtype=torch.bool); keep[0, [1, 4, 6]] = True
    a, ia = model.next_tracks(r, keep)
    b, ib = model.next_tracks(r, keep, keep_cpu=keep.clone())
    for k in a:
        assert torch.allclose(a[k].float(), b[k].float()), k
    assert ia[0].tolist() == ib[0].tolist() == [1, 4, 6]


def test_ema_foreach_equals_the_loop():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))
    a = TS.EMA(model, 0.99)
    b = REFS.EMA(model, 0.99)
    for _ in range(4):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * 0.1)
            model[1].running_mean.add_(1.0); model[1].num_batches_tracked += 1
        a.update(model); b.update(model)
    for (k, x), (_, y) in zip(a.model.state_dict().items(), b.model.state_dict().items()):
        assert torch.allclose(x.float(), y.float(), atol=1e-6), k


def test_to_dev_on_cpu_is_a_plain_copy():
    x = np.arange(6, dtype=np.int64)
    t = VF.to_dev(x, 'cpu')
    assert t.tolist() == x.tolist()
    c = VF.const('k', [1.0, 2.0], 'cpu')
    assert VF.const('k', [9.0], 'cpu') is c
