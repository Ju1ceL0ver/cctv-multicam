"""Read-only compatibility gate for the original six-example Colab update."""
import random

import cv2
import torch

import door_seg as S

torch.set_num_threads(2)
cv2.setNumThreads(2)
rng = random.Random(7)
src = S.Source(S.sources()[0])
xs = []
while len(xs) < 6:
    x = S.sample(src, rng)
    if x is not None:
        xs.append(x)
net = S.DoorSeg('cuda')
st = torch.load(S.ROOT / 'runs/doorseg_c_background/last.pt', map_location='cpu', weights_only=False)
S.checkpoint_finite(st)
net.load(st['model'])
step = st['step']
del st
net.yolo.train()
net.off.train()
net.yolo.criterion = net.yolo.init_criterion()
for _ in range(step // 2000):
    net.yolo.criterion.update()
batch, off, offv = S.collate(xs, 'cuda')
with torch.autocast('cuda', dtype=torch.bfloat16):
    pred = net.yolo(batch['img'].float() / 255)
    loss, items = net.yolo.loss(batch, pred)
    lo = S.offset_loss(net.off(net.feat), batch, off, offv)
    total = loss.sum() + 2 * 6 * lo
if not bool(torch.isfinite(total)):
    raise ValueError('nonfinite smoke loss')
total.backward()
torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0, error_if_nonfinite=True)
print('finite BF16 batch6 forward/backward; loss', float(total.detach()),
      'peak GB', torch.cuda.max_memory_allocated() / 1e9, flush=True)
