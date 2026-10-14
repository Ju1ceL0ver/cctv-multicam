"""13.10.2026, the owner: the new door system as if from the camera, everything sped up, measured.
doorseg_c (step14604) segmentation + its tracker -> the owner's zone rules (zone_rules_engine, settings as evaluated)
-> events -> staff/customer by the role worker (CPU, async). The video is fed at 12.5 fps of wall clock by a reader
thread (as a camera would), the pipeline runs as fast as it can; per frame: arrival -> done = lag.
Model variants: eager (fused, bf16 autocast) / compile (torch.compile max-autotune, MSVC env) / trt (TensorRT FP16 engine
from ONNX). Each variant is checked against eager on the same frames (detections and masks).
usage: venv_sam3/python door_live_bench.py CKPT TAG[,TAG] VARIANT[,VARIANT] -> data/demo/bench_<variant>.json"""
import json, os, queue, sys, threading, time
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'demo'
RULES = dict(shop=1, first=0, occl_legs=.6, occl_h=.6, conf=1, pair_s=3)


class Net:
    """the model behind one call: img (1, 17, 736, 1280) float -> (det, proto, off) as door_seg_fast.predict wants"""
    def __init__(self, net, variant):
        import torch
        self.net, self.variant = net, variant
        net_ = net

        class Wrap(torch.nn.Module):
            def __init__(s):
                super().__init__(); s.yolo, s.off = net_.yolo, net_.off
            def forward(s, x):
                o = s.yolo(x)
                return o[0][0], o[0][1], s.off(net_.feat)
        self.wrap = Wrap().eval()
        if variant == 'compile':
            self.fn = torch.compile(self.wrap, mode='max-autotune-no-cudagraphs')
        elif variant == 'trt':
            self.fn = self._trt()
        else:
            self.fn = self.wrap

    def _trt(self):
        from door_trt import Engine
        return Engine()

    def predict(self, img, thr=.4):
        """door_seg_fast.predict on (det, proto, off) from any variant"""
        import torch
        import door_seg as S
        det, proto, off = self.fn(img)
        det, proto, off = det[0].float(), proto[0].float(), off[0].float() * S.OFF_SCALE
        det = det[det[:, 4] >= thr]
        if not len(det):
            return []
        logits = (det[:, 6:] @ proto.flatten(1)).view(-1, S.MH, S.MW)
        boxes = det[:, :4]
        x = torch.arange(S.MW, device=img.device)[None, None, :]; y = torch.arange(S.MH, device=img.device)[None, :, None]
        lo = torch.trunc(boxes[:, :2] / 4).clamp(min=0); hi = torch.ceil(boxes[:, 2:] / 4)
        masks = (logits.sigmoid() > .5) & (x >= lo[:, 0, None, None]) & (x < hi[:, 0, None, None]) & (y >= lo[:, 1, None, None]) & (y < hi[:, 1, None, None])
        cells = masks[:, ::2, ::2].flatten(1).float(); n = cells.sum(1)
        offsets = cells @ off.flatten(1).T / n[:, None].clamp(min=1)
        cy = ((boxes[:, 1] + boxes[:, 3]) / 16).long().clamp(0, S.GH - 1); cx = ((boxes[:, 0] + boxes[:, 2]) / 16).long().clamp(0, S.GW - 1)
        offsets = torch.where(n[:, None] > 0, offsets, off[:, cy, cx].T)
        md, dd, od = masks.cpu().numpy(), det[:, :5].cpu().numpy(), offsets.cpu().numpy()
        return [(md[i], dd[i, :4].astype(float), float(dd[i, 4]), od[i, :2], od[i, 2:]) for i in range(len(dd))]


class Zones:
    def __init__(self):
        Z = json.load(open(ROOT / 'data/door_v2/door_lines_owner.json', encoding='utf-8'))['lines']
        self.names = ['band', 'shop', 'door', 'gallery']
        self.m, self.poly = {}, {}
        for n, z in zip(self.names, Z):
            m = np.zeros((180, 320), np.uint8)
            cv2.fillPoly(m, [(np.array(z['pts']) / 4).astype(np.int32)], 1)
            self.m[n] = m.astype(bool); self.poly[n] = np.array(z['pts'], np.float32)

    def rows(self, k, people):
        """people: [(pid, mask 180x320)] -> zone_rules_engine rows (pixel counts in full-frame units)"""
        out, boxes = [], []
        for pid, m in people:
            ys, xs = np.nonzero(m)
            boxes.append((xs.min() * 4, ys.min() * 4, xs.max() * 4 + 4, ys.max() * 4 + 4))
        for i, (pid, m) in enumerate(people):
            x1, y1, x2, y2 = boxes[i]
            ys, xs = np.nonzero(m); j = np.argmax(ys); fx, fy = int(xs[j] * 4 + 2), int(ys[j] * 4 + 3)
            zc = [int((m & self.m[z]).sum()) * 16 for z in self.names]
            h = y2 - y1
            L = (x1 - 5, y2 - .2 * h, x2 + 5, y2 + .5 * h)
            cov = 0.0
            for jj, (q, mm) in enumerate(people):
                if jj == i:
                    continue
                a = mm[max(0, int(L[1] / 4)):max(0, int(L[3] / 4)) + 1, max(0, int(L[0] / 4)):max(0, int(L[2] / 4)) + 1]
                if a.size:
                    cov = max(cov, float(a.mean()))
            foot = next((n for n in ('door', 'shop', 'band', 'gallery') if cv2.pointPolygonTest(self.poly[n], (float(fx), float(fy)), False) >= 0), 'none')
            out.append([k, str(pid), None, None, x1, y1, x2, y2, fx, fy, int(m.sum()) * 16] + zc + [round(cov, 3), foot])
        return out


def run(ckpt, tag, variant, check=None):
    from door_trt import gpu_guard
    gpu_guard()
    import torch
    import door_seg as S
    import door_seg_fast as F
    import zone_rules_engine as Z
    import door_role as DR
    import staff_masked as SM
    from door_seg_track import Config, Tracker
    from door_seg_run import contexts
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    net = S.DoorSeg('cuda')
    net.load(torch.load(ckpt, map_location='cpu', weights_only=False)['model'])
    net.yolo.eval(); net.off.eval(); net.yolo.fuse(verbose=False)
    torch.set_num_threads(2); cv2.setNumThreads(2)
    M = Net(net, variant)
    day = tag.split('_')[1]
    asm = F.GPUChannels(S.background(day, 'cam1'))
    prev = np.zeros((S.H, S.W), np.uint8)
    # warm-up / compile on real frames of the stretch
    cap = cv2.VideoCapture(str(ROOT / 'data/sam31_door' / tag / 'cam1/video.mp4'))
    warm = [cv2.resize(cap.read()[1], (S.W, S.H), interpolation=cv2.INTER_AREA) for _ in range(11)]
    cap.release()
    t = time.time()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=variant != 'trt'):
        for _ in range(20):
            M.predict(asm({o: warm[5 + o] for o in range(-5, 6)}, prev))
        torch.cuda.synchronize()
    warm_s = time.time() - t
    # equivalence on the warm frames against eager
    eq = None
    if check is not None:
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=variant != 'trt'):
            a = M.predict(asm({o: warm[5 + o] for o in range(-5, 6)}, prev))
        b_ = check
        ious = []
        for p in a:
            best = max((float((p[0] & q[0]).sum() / max(1, (p[0] | q[0]).sum())) for q in b_), default=0)
            ious.append(best)
        eq = dict(n=len(a), n_eager=len(b_), mask_iou_min=round(min(ious), 3) if ious else None, mask_iou_mean=round(float(np.mean(ious)), 3) if ious else None)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        ref = Net(net, 'eager').predict(asm({o: warm[5 + o] for o in range(-5, 6)}, prev)) if variant == 'eager' else None

    # the camera: frames at 12.5 fps of wall clock
    q = queue.Queue()
    cap = cv2.VideoCapture(str(ROOT / 'data/sam31_door' / tag / 'cam1/video.mp4'))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    def camera():
        t0 = time.perf_counter(); k = 0
        while True:
            ok, f = cap.read()
            if not ok:
                q.put(None); return
            target = t0 + k * .08
            dt = target - time.perf_counter()
            if dt > 0:
                time.sleep(dt)
            q.put((k, time.perf_counter(), f)); k += 1
    arrive = {}
    class Cam:
        def read(self):
            it = q.get()
            if it is None:
                return False, None
            k, ta, f = it
            arrive[k] = ta
            return True, cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA)
    threading.Thread(target=camera, daemon=True).start()
    tracker = Tracker(Config()); zones = Zones()
    bank = DR.Bank(OUT / ('bench_views_%s_%s' % (variant, tag)))
    rows, done, prof = [], {}, {k: [] for k in ('channels', 'model', 'tracker', 'zones', 'views', 'events')}
    first_seen, events = {}, []
    tstart = time.perf_counter()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=variant != 'trt'):
        for tick, ctx in contexts(Cam()):
            t = time.perf_counter()
            img = asm(ctx, prev); torch.cuda.synchronize(); t1 = time.perf_counter(); prof['channels'].append(t1 - t)
            pred = M.predict(img); t2 = time.perf_counter(); prof['model'].append(t2 - t1)
            prev = F.previous_union([p for p in pred if p[2] >= .4])
            out = [(pid, p[0][:S.H // 4]) for pid, p in tracker.update(tick, pred) if p[0][:S.H // 4].any()]
            t3 = time.perf_counter(); prof['tracker'].append(t3 - t2)
            if tick % 3 == 0 and out:
                rows += zones.rows(tick, out)
                t4 = time.perf_counter(); prof['zones'].append(t4 - t3)
                for pid, m in out:                                    # views for the role
                    ys, xs = np.nonzero(m)
                    bx = [xs.min() * 4, ys.min() * 4, xs.max() * 4 + 4, ys.max() * 4 + 4]
                    sc = DR.score(int(m.sum()) * 16 * (2176 / 1280) ** 2, True, bx[1] <= 2 or bx[3] >= 716)
                    if bank.wants(pid, sc):
                        full = cv2.resize(m.astype(np.uint8), (S.W, S.H), interpolation=cv2.INTER_NEAREST) > 0
                        bank.offer(pid, sc, lambda: SM.crop_masked(ctx[0], full, bx), bx)
                prof['views'].append(time.perf_counter() - t4)
            if tick % 12 == 11:                                       # events once a second, as live
                t5 = time.perf_counter()
                ev, _ = Z.run_tag({'rows': rows}, dict(Z.BASE, **RULES))
                for e in ev:
                    key = (e['kind'], str(e['w']), round(e['t'], 2))
                    if key not in first_seen:
                        first_seen[key] = time.perf_counter(); events.append(dict(e, t_found=time.perf_counter()))
                prof['events'].append(time.perf_counter() - t5)
            done[tick] = time.perf_counter()
    wall = time.perf_counter() - tstart
    # role: asynchronously in live; here after, timed per person
    t = time.perf_counter(); roles = DR.Roles(bank, DR.Worker()); role_start = time.perf_counter() - t
    role_t = []
    for e in events:
        t = time.perf_counter(); e['role'] = roles.role([int(e['w']) if str(e['w']).isdigit() else e['w']]).get('role'); role_t.append(time.perf_counter() - t)
    roles.worker.p.kill()
    lag = [done[k] - arrive[k] for k in done if k in arrive]
    # an event is final once its frame's +5 context arrived and the next once-a-second pass ran
    ev_delay = [e['t_found'] - (tstart + e['t'] + 0.0) for e in events]
    ms = {k: round(1000 * float(np.mean(v)), 2) if v else 0 for k, v in prof.items()}
    per = {k: round(1000 * float(np.sum(v)) / max(1, len(done)), 2) for k, v in prof.items()}
    rep = dict(variant=variant, tag=tag, frames=len(done), video_s=round(n_total * .08, 1), wall_s=round(wall, 1), warm_s=round(warm_s, 1),
               ms_mean_per_call=ms, ms_per_frame=per, ms_per_frame_total=round(sum(per.values()), 1),
               fps_capacity=round(1000 / max(1e-6, sum(per.values())), 1),
               lag_s=dict(mean=round(float(np.mean(lag)), 3), p95=round(float(np.percentile(lag, 95)), 3), max=round(float(np.max(lag)), 3)),
               event_found_after_s=[round(x, 2) for x in ev_delay],
               events=[dict(kind=e['kind'], t=round(e['t'], 1), role=e.get('role')) for e in events],
               role_worker_start_s=round(role_start, 1), role_per_person_s=[round(x, 2) for x in role_t],
               gpu_peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), equivalence_vs_eager=eq)
    return rep, ref


if __name__ == '__main__':
    if '--equivalence-only' in sys.argv:
        from door_trt_check import CKPT,TRAIN,TEST,gpu_check
        if Path(sys.argv[1]).resolve() != CKPT.resolve() or set(sys.argv[2].split(',')) != {TRAIN,TEST} or sys.argv[3] != 'eager,trt':
            raise ValueError('Equivalence CLI requires frozen C checkpoint, both declared tags, eager,trt')
        gpu_check()
        sys.exit(0)
    ckpt, tags, variants = sys.argv[1], sys.argv[2].split(','), sys.argv[3].split(',')
    OUT.mkdir(parents=True, exist_ok=True)
    res = []
    ref = {}
    for tag in tags:
        for v in variants:
            try:
                r, rf = run(ckpt, tag, v, ref.get(tag))
                if rf is not None:
                    ref[tag] = rf
            except Exception as exc:
                import traceback; traceback.print_exc()
                r = dict(variant=v, tag=tag, error=str(exc)[:400])
            res.append(r)
            print(json.dumps(r, ensure_ascii=False), flush=True)
            json.dump(res, open(OUT / 'bench.json', 'w'), ensure_ascii=False, indent=1)
    print('done')
