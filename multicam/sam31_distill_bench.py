"""Where a distillation step's time goes (05.10.2026): frames from the loader, the teacher, the student with and
without the necks' high-resolution maps; peak card memory. usage (venv_sam3): sam31_distill_bench.py"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import sam31_distill as SD
    out = {}
    dl = torch.utils.data.DataLoader(SD.Frames(SD.sources()), batch_size=2, num_workers=4, prefetch_factor=4)
    it = iter(dl)
    next(it)
    t = time.time()
    for _ in range(20):
        next(it)
    out['loader_frames_per_s'] = round(40 / (time.time() - t), 2)
    dev = 'cuda'
    neck = SD.teacher_neck(dev)
    s = SD.Student().to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(s.parameters(), 1e-4)
    ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    x = SD.norm_input(next(it), dev)

    def timed(f, n=5):
        f(); torch.cuda.synchronize()
        t = time.time()
        for _ in range(n):
            f()
        torch.cuda.synchronize()
        return round((time.time() - t) / n * 1000, 1)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), ac():
        out['teacher_trunk_ms_b2'] = timed(lambda: neck.trunk(x))
        tt = neck.trunk(x)[-1]
        out['teacher_necks_ms_b2'] = timed(lambda: SD.neck_maps(neck, tt))
    out['mem_teacher_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)

    def step(full):
        with ac():
            y = s(x)[-1]
            if full:
                loss = sum(m.float().pow(2).mean() for h in SD.neck_maps(neck, y) for m in h)
            else:
                loss = y.float().pow(2).mean() + sum(h[2].float().pow(2).mean() for h in SD.neck_maps(neck, y))
        opt.zero_grad(); loss.backward(); opt.step()
    torch.cuda.reset_peak_memory_stats()
    out['student_step_ms_b2_all_necks'] = timed(lambda: step(True))
    out['mem_student_all_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    torch.cuda.reset_peak_memory_stats()
    out['student_step_ms_b2_1x_only'] = timed(lambda: step(False))
    out['mem_student_1x_gb'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    with torch.no_grad(), ac():
        out['student_fwd_ms_b1'] = timed(lambda: s(x[:1]), 10)
    print(json.dumps(out), flush=True)
    json.dump(out, open(ROOT / 'data' / 'logs' / 'sam31_distill_bench.json', 'w'))
    import os
    os._exit(0)


if __name__ == '__main__':
    main()
