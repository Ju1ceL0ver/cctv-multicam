"""Why a saved student finds nobody (06.10.2026): the same weights, BatchNorm in eval mode (running statistics) and
in train mode (each frame's own statistics), on the 48 held-out frames of 23.09."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def main():
    import sam31_distill as SD
    import sam31_distill_det as DD
    dev = 'cuda'
    D = DD.Detector()
    test = SD.test_frames(48)
    ac = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    cache, rep = {}, {}
    for name in ('runs/s31micro_a/last.pt', 'runs/s31det_c/eval.pt', 'runs/s31det_c/last.pt'):
        ck = torch.load(str(ROOT / name), map_location='cpu', weights_only=False)
        for key in ('model', 'ema'):
            if key not in ck:
                continue
            s = SD.Student(**ck['arch'], pretrained=False)
            s.load_state_dict(ck[key])
            s = s.to(dev)
            bn = [m for m in s.modules() if isinstance(m, torch.nn.BatchNorm2d)]
            stats = {'bn_mean_abs': round(float(torch.stack([m.running_mean.abs().mean() for m in bn]).mean()), 4),
                     'bn_var_mean': round(float(torch.stack([m.running_var.mean() for m in bn]).mean()), 4),
                     'nan_params': int(sum((~torch.isfinite(p)).sum() for p in s.parameters()))}
            h_eval = DD.heldout(D, s, test, cache, dev, ac)
            s.train()
            for m in s.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.momentum = 0.0                       # train-mode normalisation without touching the stored stats
            h_train = DD.heldout.__wrapped__(D, s, test, cache, dev, ac) if hasattr(DD.heldout, '__wrapped__') else None
            if h_train is None:
                with torch.no_grad():
                    import types
                    s.eval = types.MethodType(lambda self: self, s)    # keep train-mode BN inside heldout
                    h_train = DD.heldout(D, s, test, cache, dev, ac)
            rep['%s:%s' % (name, key)] = {'eval_bn': h_eval, 'train_bn': h_train, **stats}
            print(name, key, json.dumps(rep['%s:%s' % (name, key)]), flush=True)
    json.dump(rep, open(ROOT / 'data' / 'logs' / 'sam31_diag_bn.json', 'w'), indent=1)
    import os
    os._exit(0)


if __name__ == '__main__':
    main()
