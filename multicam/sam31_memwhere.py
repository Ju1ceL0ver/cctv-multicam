"""09.10: what of a 240-frame SAM 3.1 session lies on the card at its end -- every CUDA tensor reachable from the
session's state, summed by its key path (frame numbers folded). usage (venv_sam3): sam31_memwhere.py [TAG] [N]"""
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch


def walk(x, path, out, seen):
    if id(x) in seen:
        return
    seen.add(id(x))
    if torch.is_tensor(x):
        if x.is_cuda:
            out[re.sub(r'\[\d+\]', '[k]', path)] += x.element_size() * x.nelement()
        return
    if isinstance(x, dict):
        for k, v in x.items():
            walk(v, '%s[%s]' % (path, k if not isinstance(k, int) else k), out, seen)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x[:4096]):
            walk(v, '%s[%d]' % (path, i), out, seen)
    elif hasattr(x, '__dict__') and not isinstance(x, torch.nn.Module) and len(path) < 200:
        for k, v in vars(x).items():
            walk(v, '%s.%s' % (path, k), out, seen)


def main(tag='door_20260919_32350', n='240'):
    import door_micro as DM
    import sam_speed as SP
    dst = ROOT / 'data' / 'logs' / 'memwhere'
    SP.frames(tag, int(n), 3, dst)
    pred = DM.build(str(ROOT / 'runs' / 's31micro_a' / 'last.pt'), None)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        sid = pred.handle_request(dict(type='start_session', resource_path=str(dst / 'sam_in'), offload_video_to_cpu=True))['session_id']
        pred.handle_request(dict(type='add_prompt', session_id=sid, frame_index=0, text='person'))
        for _ in pred.handle_stream_request(dict(type='propagate_in_video', session_id=sid, propagation_direction='forward')):
            pass
    torch.cuda.synchronize()
    print('allocated %.2f GB, peak %.2f GB' % (torch.cuda.memory_allocated() / 2 ** 30, torch.cuda.max_memory_allocated() / 2 ** 30), flush=True)
    out = defaultdict(int)
    walk(pred._all_inference_states[sid]['state'], 'state', out, set())
    tot = sum(out.values())
    print('reachable from the session: %.2f GB' % (tot / 2 ** 30), flush=True)
    for k, v in sorted(out.items(), key=lambda z: -z[1])[:25]:
        print('%8.1f MB  %s' % (v / 2 ** 20, re.sub(r"\[[0-9]+\]", "[k]", k)[:200]), flush=True)
    print('done', flush=True)


if __name__ == '__main__':
    main(*sys.argv[1:])
