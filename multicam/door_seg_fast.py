"""Equivalent inference preparation/postprocessing without per-person CUDA syncs.

Experimental helpers; never modify training mode or production configuration.
"""
import numpy as np


def predict(net, img, thr=0.4):
    import torch
    import door_seg as S
    out = net.yolo(img)
    det, proto = out[0][0][0], out[0][1][0]
    off = net.off(net.feat)[0].float() * S.OFF_SCALE
    det = det[det[:, 4] >= thr].float()
    if not len(det):
        return []
    logits = (det[:, 6:] @ proto.float().flatten(1)).view(-1, S.MH, S.MW)
    boxes = det[:, :4]
    x = torch.arange(S.MW, device=img.device)[None, None, :]
    y = torch.arange(S.MH, device=img.device)[None, :, None]
    lo = torch.trunc(boxes[:, :2] / 4).clamp(min=0)
    hi = torch.ceil(boxes[:, 2:] / 4)
    masks = (logits.sigmoid() > .5) & (x >= lo[:, 0, None, None]) & (x < hi[:, 0, None, None]) & \
        (y >= lo[:, 1, None, None]) & (y < hi[:, 1, None, None])
    cells = masks[:, ::2, ::2].flatten(1).float()
    n = cells.sum(1)
    with torch.autocast(img.device.type, enabled=False):
        offsets = cells @ off.flatten(1).T / n[:, None].clamp(min=1)
    cy = ((boxes[:, 1] + boxes[:, 3]) / 16).long().clamp(0, S.GH - 1)
    cx = ((boxes[:, 0] + boxes[:, 2]) / 16).long().clamp(0, S.GW - 1)
    offsets = torch.where(n[:, None] > 0, offsets, off[:, cy, cx].T)
    # Three bulk copies instead of several syncs for every detected person.
    md, dd, od = masks.cpu().numpy(), det[:, :5].cpu().numpy(), offsets.cpu().numpy()
    return [(md[i], dd[i, :4].astype(float), float(dd[i, 4]), od[i, :2], od[i, 2:]) for i in range(len(dd))]


class Channels:
    """Bounded grayscale cache; exact existing 17 uint8 channels."""
    def __init__(self, background):
        import door_seg as S
        self.background = np.zeros((3, S.HP, S.W), np.uint8)
        self.background[:, :S.H] = background[:, :, ::-1].transpose(2, 0, 1)
        self.grey = {}

    def __call__(self, frames, prev):
        import door_seg as S
        # Use object identity with references to protect against recycled ids.
        current = {id(f): f for f in frames.values()}
        self.grey = {k: v for k, v in self.grey.items() if k in current}
        for k, f in current.items():
            if k not in self.grey:
                self.grey[k] = (f, S._grey(f).astype(np.int16))
        out = np.zeros((S.C, S.HP, S.W), np.uint8)
        out[:3, :S.H] = frames[0][:, :, ::-1].transpose(2, 0, 1)
        out[3:6] = self.background
        g0 = self.grey[id(frames[0])][1]
        for c, offset in enumerate(tuple(-k for k in S.PAST) + S.FUT, 6):
            d = np.abs(g0 - self.grey[id(frames[offset])][1])
            out[c, :S.H] = np.minimum(d * S.DIFF_GAIN, 255)
        out[-1, :S.H] = prev * 255
        return out


def previous_union(predictions):
    import cv2
    import door_seg as S
    small = np.zeros((S.H // 4, S.W // 4), np.uint8)
    for p in predictions:
        small |= p[0][:S.H // 4].astype(np.uint8)
    return cv2.resize(small, (S.W, S.H), interpolation=cv2.INTER_NEAREST)


class GPUChannels:
    """Copy each decoded BGR+OpenCV grey once; build identical channels on GPU."""
    def __init__(self, background, device='cuda', channels_last=False):
        import torch
        import door_seg as S
        self.device = device
        self.channels_last = channels_last
        self.background = torch.from_numpy(background[:, :, ::-1].copy().transpose(2, 0, 1)).to(device)
        self.frames = {}
        self.image = torch.zeros((1, S.C, S.HP, S.W), dtype=torch.uint8, device=device)

    def __call__(self, frames, prev):
        import torch
        import door_seg as S
        current = {id(f): f for f in frames.values()}
        self.frames = {k: v for k, v in self.frames.items() if k in current}
        for k, f in current.items():
            if k not in self.frames:
                packed = np.concatenate([f, S._grey(f)[:, :, None]], axis=2)
                gpu = torch.from_numpy(packed).to(self.device)
                self.frames[k] = (f, gpu[:, :, :3], gpu[:, :, 3].to(torch.int16))
        out = self.image[0]
        bgr, grey = self.frames[id(frames[0])][1:]
        out[:3, :S.H] = bgr.flip(2).permute(2, 0, 1)
        out[3:6, :S.H] = self.background
        for c, offset in enumerate(tuple(-k for k in S.PAST) + S.FUT, 6):
            d = (grey - self.frames[id(frames[offset])][2]).abs() * S.DIFF_GAIN
            out[c, :S.H] = d.clamp(max=255).to(torch.uint8)
        if isinstance(prev, np.ndarray):
            prev = torch.from_numpy(prev).to(self.device)
        out[-1, :S.H] = prev.to(torch.uint8) * 255
        image = self.image.float().div_(255)
        return image.contiguous(memory_format=torch.channels_last) if self.channels_last else image
