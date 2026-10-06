"""One step of sam31_distill_det.py taken apart (06.10.2026): where do 6 s a frame go."""
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import sam31_distill as SD
    import sam31_distill_det as DD
    dev = 'cuda'
    D = DD.Detector()
    ck = torch.load(str(ROOT / 'runs' / 's31micro_a' / 'last.pt'), map_location='cpu', weights_only=False)
    s = SD.Student(**ck['arch'], pretrained=False)
    s.load_state_dict(ck['model'])
    s = s.to(dev)
    x = SD.norm_input(SD.test_frames(2)[:1], dev)
    ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    T = {}

    def tm(name, f, n=3):
        f(); torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            r = f()
        torch.cuda.synchronize()
        T[name] = round((time.time() - t) / n * 1000, 1)
        return r
    with torch.no_grad(), ac():
        tl = tm('teacher_trunk', lambda: D.teacher_trunk(x))
        o_t = tm('teacher_detector', lambda: D.run(x, DD._Fixed(tl)))
    def student_fwd():
        with ac():
            return s(x)
    y = tm('student_fwd', student_fwd)

    def det_grad():
        with ac():
            y_ = s(x)
            o = D.run(x, DD._Fixed(y_))
            loss, _, _ = DD.answer_losses(o, o_t)
        return loss
    tm('student+detector fwd (grad)', det_grad)

    def full():
        loss = det_grad()
        loss.backward()
    torch.cuda.reset_peak_memory_stats()
    tm('student+detector fwd+bwd', full)
    T['peak_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    with torch.autograd.profiler.profile(use_cuda=True) as prof:
        full(); torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=25), flush=True)
    print(json.dumps(T), flush=True)
    json.dump(T, open(ROOT / 'data' / 'logs' / 'sam31_det_prof.json', 'w'))
    import os
    os._exit(0)


if __name__ == '__main__':
    main()
