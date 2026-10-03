import numpy as np

from multicam import rate_features, train_rate


def test_split_keeps_files_whole_and_hits_the_fraction():
    ids = ['2026091%d_cam1_100000_%04d_%03d' % (d, f, s) for d in range(3, 6) for f in range(20) for s in (1, 2)]
    scores = {i: 1 + (hash(i) % 5) for i in ids}
    tr, te = train_rate.split(ids, scores, 0.2, 0)
    assert not set(tr) & set(te) and len(tr) + len(te) == len(ids)
    assert {train_rate.group_of(i) for i in tr}.isdisjoint({train_rate.group_of(i) for i in te})
    assert 0.15 * len(ids) <= len(te) <= 0.25 * len(ids)
    assert train_rate.split(ids, scores, 0.2, 0) == (tr, te)


def test_draft_channels_outline_the_border_between_two_people():
    lab = np.zeros((20, 20), np.uint8)
    lab[5:15, 2:10], lab[5:15, 10:18] = 1, 2
    m, e = train_rate.draft_channels(lab)
    assert m[10, 10] == 255 and m[0, 0] == 0
    assert e[10, 9] == 255 and e[10, 10] == 255 and e[10, 5] == 0


def test_augment_keeps_shapes_and_the_model_takes_eight_channels():
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 255, (train_rate.H, train_rate.W, 5), dtype=np.uint8)
    d = rng.random((4, train_rate.H, train_rate.W)).astype(np.float16)
    img, md, dd = train_rate.augment(u8, d, rng)
    assert img.shape == (train_rate.H, train_rate.W, 3) and md.shape == (train_rate.H, train_rate.W, 2)
    assert dd.shape == (4, train_rate.H, train_rate.W) and dd.min() >= 0
    dstat = np.array([[0, 0, 0, 0], [1, 1, 1, 1]], np.float32)
    x = train_rate.to_tensor(img, md, dd, dstat, ['rgb', 'draft', 'diff'])
    assert x.shape == (11, train_rate.H, train_rate.W)
    import torch
    model = train_rate.build_model(11, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.from_numpy(np.stack([x, x]))).shape == (2,)


def test_background_maps_light_up_a_new_object_only():
    rng = np.random.default_rng(1)
    bg = rng.integers(60, 200, (rate_features.H, rate_features.W, 3), dtype=np.uint8)
    past = [np.clip(bg.astype(int) + rng.integers(-3, 4, bg.shape), 0, 255).astype(np.uint8) for _ in range(20)]
    cur = bg.copy()
    cur[100:200, 300:350] = (20, 20, 230)
    L = rate_features.lab
    gmu, gsd = L(bg), np.full(bg.shape, 3.0, np.float32)
    m = rate_features.maps(L(cur), past, past, gmu, gsd)
    for c in m:
        assert c[150, 320] > 5 * np.median(c)
