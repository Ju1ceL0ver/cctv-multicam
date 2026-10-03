"""Inference speed of the slot model: a pair of frames (the two cameras) at 1088x608, fp16 and bf16,
PyTorch eager (TensorRT not yet). Parts: backbone, neck+heads+decoder, full masks."""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from slot_model import SlotModel
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    res = {}
    for layers in (4,):
        m = SlotModel('deimv2_vit_tiny', pretrained=False, layers=layers, teacher_dims=(3840, 1024)).to(dev).eval().to(memory_format=torch.channels_last)
        x = torch.randn(2, 5, 608, 1088, device=dev).to(memory_format=torch.channels_last)

        def run():
            o = m(x, aux=False, radio=False)                       # as predict(): no radio head
            w = torch.zeros(2, 32, *o['pix'].shape[-2:], device=dev, dtype=o['pix'].dtype)
            for b in range(2):                                      # masks for 5 people per frame
                w[b, :5] = torch.einsum('sc,chw->shw', o['mask_vec'][b, :5], o['pix'][b]).sigmoid().to(w.dtype)
            return m.pooled(o, w)
        for dt in (torch.float16,):
            with torch.no_grad(), torch.autocast('cuda', dt):
                for _ in range(10):
                    run()
                torch.cuda.synchronize(); t = time.time()
                for _ in range(30):
                    run()
                torch.cuda.synchronize(); full = (time.time() - t) / 30 * 1000
                t = time.time()
                for _ in range(30):
                    m.body(x)
                torch.cuda.synchronize(); body = (time.time() - t) / 30 * 1000
            res['layers%d_%s' % (layers, str(dt)[6:])] = {'pair_ms': round(full, 1), 'backbone_ms': round(body, 1)}
        res['params_M_inference'] = round(sum(p.numel() for n, p in m.named_parameters() if not n.startswith(('to_teacher', 'radio'))) / 1e6, 2)
        del m; torch.cuda.empty_cache()
    print(json.dumps(res)); json.dump(res, open(ROOT / 'data' / 'logs' / 'speed_slots.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
