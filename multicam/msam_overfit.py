"""Can the tracker learn one clip? 200 steps on a single clip; slot masks vs the teacher's, and whether each slot
draws its own person or all of them (dice against its own vs against the union of all)."""
import sys, numpy as np, torch, torch.nn.functional as F
from pathlib import Path
R = Path(__file__).resolve().parent
sys.path.insert(0, str(R))
import micro_sam as MS
dev = 'cuda'
torch.manual_seed(0)
m = MS.MicroSAM().to(dev)
sd = torch.load(R / 'runs' / 'msam_a' / 'step11500.pt', map_location='cpu', weights_only=False)['model']
own = m.state_dict(); m.load_state_dict({k: v for k, v in sd.items() if k in own and own[k].shape == v.shape}, strict=False)
data = MS.Clips(seed=3)
while True:
    imgs, masks, boxes, vis = data.sample()
    if vis[0].sum() >= 2:
        break
print('people in clip', masks.shape[1], 'visible on frame 0', int(vis[0].sum()), flush=True)
opt = torch.optim.AdamW([p for n, p in m.named_parameters() if not n.startswith('enc_net.')], lr=2e-4)
for it in range(201):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, st = MS.run_clip(m, imgs, masks, boxes, vis, dev)
    opt.zero_grad(); loss.backward(); opt.step()
    if it % 25 == 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            x = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255
            g16, g8, g4 = m.features(x)
            slot = torch.zeros(1, MS.SLOTS, 184, 320, device=dev)
            N = masks.shape[1]
            slot[0, :N] = torch.from_numpy(masks[0]).to(dev).float()
            mem = [m.memory(g16[:1], slot)]
            pm, pv, _ = m.track(g16[1:2], g8[1:2], g4[1:2], mem, torch.zeros(1, MS.SLOTS, MS.D, device=dev), slot)
            p = (pm[0, :N].float().sigmoid() > 0.5).float().cpu().numpy()
        gt = masks[1].astype(float)
        uni = gt.max(0)
        d = lambda a, b: 2 * (a * b).sum() / max(1, a.sum() + b.sum())
        own_ = [round(d(p[i], gt[i]), 2) for i in range(N) if vis[1, i]]
        all_ = [round(d(p[i], uni), 2) for i in range(N) if vis[1, i]]
        print(it, 'trk_dice %.3f' % st.get('trk_dice', -1), 'slot vs own person', own_, 'slot vs all people', all_, flush=True)
