"""The slot model end to end on a synthetic day: loader -> model -> matching and losses -> backward."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1] / 'multicam'
sys.path.insert(0, str(ROOT))
torch = pytest.importorskip('torch')
cv2 = pytest.importorskip('cv2')


def make_root(tmp):
    import slot_data
    d = tmp / 'data'
    seg = d / 'seg_datasets'
    ident = '20260919_cam1_100003_0005_120'
    for sub in ('rf_20260925/train/images', 'pseudo_x/drafts', 'backgrounds'):
        (seg / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    bg = (rng.integers(60, 90, (720, 1280, 3))).astype(np.uint8)
    img = bg.copy()
    lab = np.zeros((720, 1280), np.uint8)
    for v, (x, y) in enumerate(((300, 200), (800, 300)), 1):
        img[y:y + 300, x:x + 100] = (200, 40, 40 * v)
        lab[y:y + 300, x:x + 100] = v
    cv2.imwrite(str(seg / 'rf_20260925/train/images' / (ident + '.jpg')), img)
    cv2.imwrite(str(seg / 'pseudo_x/drafts' / (ident + '.png')), lab)
    cv2.imwrite(str(seg / 'backgrounds' / ('20260919_cam1_100003_0005.jpg')), bg)
    (d / 'slot_prep' / 'multiday').mkdir(parents=True)
    L = cv2.cvtColor(cv2.resize(bg, (544, 306)), cv2.COLOR_BGR2LAB).astype(np.float16)
    np.savez(d / 'slot_prep' / 'multiday' / 'cam1_all.npz', mu=L, sd=np.full_like(L, 4.0), n=10)
    json.dump({'mean': [0.4, 0.4, 0.4, 3.0, 1.7], 'std': [0.25, 0.25, 0.25, 5.3, 2.5]}, open(d / 'slot_prep' / 'norm.json', 'w'))
    (d / 'teacher_emb').mkdir()
    np.savez(d / 'teacher_emb' / (ident + '.npz'), labels=np.array([1, 2], np.int16), boxes=np.zeros((2, 4), np.int16),
             cloth=rng.normal(size=(2, 3840)).astype(np.float16), shape=rng.normal(size=(2, 1024)).astype(np.float16))
    (d / 'radio_feats').mkdir()
    np.save(d / 'radio_feats' / (ident + '.npy'), rng.normal(size=(72, 128, 512)).astype(np.float16))
    (d / 'inout').mkdir()
    json.dump({'labels': {ident + '_p1': {'label': 1}, ident + '_p2': {'label': 3}}}, open(d / 'inout' / 'labels.json', 'w'))
    return d, ident


def test_sample_targets(tmp_path):
    import slot_data
    d, ident = make_root(tmp_path)
    fr = slot_data.Frames(d)
    s = slot_data.load(fr, ident, train=False)
    assert s['x'].shape == (5, 608, 1088)
    assert s['masks'].shape == (2, 152, 272) and s['boxes'].shape == (2, 4)
    assert s['has_reid'].all() and list(s['inout']) == [1, 2]
    assert s['radio'].shape == (512, 38, 68) and s['radio_valid'].mean() > 0.9
    # the person's heat map is high, the empty floor's low
    assert s['x'][3, 300, 290] > s['x'][3, 50, 50] + 3      # person at x 255-340, y 170-425 of the 1088 frame
    t = slot_data.load(fr, ident, train=True, rng=np.random.default_rng(3))
    assert t['masks'].shape[1:] == (152, 272) and len(t['boxes']) >= 1


@pytest.mark.parametrize('backbone', ['deimv2_vit_tiny', 'convnext_atto'])
def test_forward_backward(tmp_path, backbone):
    import slot_data
    import train_slots
    from slot_model import SlotModel
    d, ident = make_root(tmp_path)
    fr = slot_data.Frames(d)
    x, radio, valid, targets = slot_data.collate([slot_data.load(fr, ident, train=True, rng=np.random.default_rng(1))])
    model = SlotModel(backbone, pretrained=False, mean=fr.mean, std=fr.std, layers=2, teacher_dims=fr.teacher_dims)
    out = model(x)
    assert model.full_masks(out).shape == (1, 32, 152, 272) and out['radio'].shape == (1, 512, 38, 68)
    loss, parts = train_slots.step_losses(model, out, radio, valid, targets)
    assert np.isfinite(float(loss))
    for k in ('obj', 'l1', 'giou', 'bce', 'dice', 'radio', 'reid', 'rel', 'inout'):
        assert k in parts, k
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.body.parameters())
    people = train_slots.predict(model.eval(), x, thr=0.0)[0]
    assert len(people) == 32 and people[0][1].shape == (152, 272)


def test_deimv2_checkpoint_layout():
    """Our DEIMv2 backbone has DEIMv2-S's parameter names, and the stem takes the extra channels."""
    from slot_model import DEIMv2Tiny
    ours = DEIMv2Tiny(5)
    ref = DEIMv2Tiny(3)
    sd = {'backbone.' + k: v for k, v in ref.state_dict().items()}
    missing, unexpected = ours.load_deimv2(sd)
    assert not unexpected and not [k for k in missing if 'num_batches' not in k]
    w = ours.sta.stem[0].weight
    assert torch.equal(w[:, :3], ref.sta.stem[0].weight) and not w[:, 3:].any()


def test_point_sampling_reads_the_right_pixels():
    """Targets sampled at points equal the mask at those points; the uniform quarter gives the IoU."""
    import train_slots
    m = torch.zeros(1, 152, 272); m[0, 40:120, 100:150] = 1
    pix = torch.randn(8, 152, 272)
    logits, target, k_imp = train_slots.mask_points(torch.randn(1, 8), pix, m)
    assert logits.shape == target.shape == (1, train_slots.LOSS_POINTS)
    share = float(target[0, k_imp:].mean())
    assert abs(share - (80 * 50) / (152 * 272)) < 0.03          # the uniform points see the mask's real area
    pts = torch.tensor([[125 / 272, 80 / 152], [10 / 272, 10 / 152]])
    v = train_slots.sample(m[:, None], pts[None])[..., 0]
    assert v[0, 0] > 0.99 and v[0, 1] < 0.01


def test_denoising_and_one_to_many(tmp_path):
    """Hints run beside the slots, cannot be seen by them, and add their own losses; the aux layers may
    give one person several slots, the last layer one."""
    import slot_data
    import train_slots
    from slot_model import SlotModel
    d, ident = make_root(tmp_path)
    fr = slot_data.Frames(d)
    x, radio, valid, targets = slot_data.collate([slot_data.load(fr, ident, train=False)])
    model = SlotModel('convnext_atto', pretrained=False, mean=fr.mean, std=fr.std, layers=2, teacher_dims=fr.teacher_dims).eval()
    dn, dn_matches = train_slots.make_dn(targets, 'cpu')
    assert dn['box'].shape[1] == train_slots.DN_MAX // 4 * 4 and len(dn_matches[0][0]) == len(dn_matches[0][1])
    with torch.no_grad():
        plain = model(x)
        hinted = model(x, dn=dn)
    # the ordinary slots cannot see the hints: their outputs do not change
    assert torch.allclose(plain['obj'], hinted['obj'], atol=1e-4)
    assert len(hinted['dn']) == 2 and hinted['dn'][-1]['obj'].shape[1] == dn['box'].shape[1]
    model.train()
    out = model(x, dn=dn); out['dn_matches'] = dn_matches
    loss, parts = train_slots.step_losses(model, out, radio, valid, targets)
    assert 'dn_dice' in parts and np.isfinite(float(loss))
    r, c = train_slots.match(out['aux'][0]['obj'][0], out['aux'][0]['box'][0], out['aux'][0]['mask_vec'][0], out['pix'][0], targets[0], k=4)
    assert len(r) == 8 and sorted(c.tolist()) == [0] * 4 + [1] * 4
