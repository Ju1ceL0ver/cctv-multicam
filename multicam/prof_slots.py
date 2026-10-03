"""Where a training step of the slot model spends its time: loading, forward, matching, losses, backward."""
import json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import slot_data, train_slots
    from slot_model import SlotModel
    dev = 'cuda'
    torch.backends.cudnn.benchmark = True
    fr = slot_data.Frames()
    ids = slot_data.train_ids(fr, 3)[::500][:24]
    t = time.time(); samples = [slot_data.load(fr, i, True) for i in ids]; load_s = (time.time() - t) / len(ids)
    model = SlotModel('deimv2_vit_tiny', pretrained=False, mean=fr.mean, std=fr.std, teacher_dims=fr.teacher_dims).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), 1e-4)
    T = {'forward': 0.0, 'match': 0.0, 'losses_rest': 0.0, 'backward': 0.0}
    orig_match = train_slots.match
    def timed_match(*a, **k):
        torch.cuda.synchronize(); t0 = time.time(); r = orig_match(*a, **k); torch.cuda.synchronize(); T['match'] += time.time() - t0; return r
    train_slots.match = timed_match
    n = 0
    for rep in range(3):
        for k in range(0, len(samples), 4):
            x, radio, valid, targets = slot_data.collate(samples[k:k + 4])
            x = x.to(dev).to(memory_format=torch.channels_last); radio = radio.to(dev)
            targets = [{kk: v.to(dev) for kk, v in tt.items()} for tt in targets]
            torch.cuda.synchronize(); t0 = time.time()
            with torch.autocast('cuda', torch.bfloat16):
                out = model(x)
            torch.cuda.synchronize(); t1 = time.time()
            m0 = T['match']
            loss, _ = train_slots.step_losses(model, out, radio, valid, targets)
            torch.cuda.synchronize(); t2 = time.time()
            loss.backward(); opt.step(); opt.zero_grad()
            torch.cuda.synchronize(); t3 = time.time()
            if rep:
                T['forward'] += t1 - t0; T['losses_rest'] += (t2 - t1) - (T['match'] - m0); T['backward'] += t3 - t2; n += 1
            else:
                T['match'] = m0
    res = {k: round(v / n * 1000) for k, v in T.items()}
    res['load_one_frame_ms_one_core'] = round(load_s * 1000)
    res['mem_GB'] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    print(json.dumps(res)); json.dump(res, open(ROOT / 'data' / 'logs' / 'prof_slots.json', 'w'))


if __name__ == '__main__':
    main()
