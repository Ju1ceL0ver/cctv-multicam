"""13.10.2026, the owner: a video of the NEW door system before it goes live -- doorseg_c (step14604) + its tracker,
sides and events by the owner's zone rules (zone_rules_engine, settings as evaluated), staff/customer by the role
model on each person's best views. venv_sam3/python door_demo2.py TAG -> data/demo/<tag>_new.mp4"""
import json, sys, time
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'demo'
CK = ROOT / 'data/quality_backups/doorseg_c_kaggle_final_20261010/step14604_ce9b0a8498d3.pt'


def main(tag):
    import torch
    import door_seg as S, door_seg_fast as F, zone_rules_engine as Z, door_role as DR, staff_masked as SM
    from door_seg_track import Config, Tracker
    from door_seg_run import contexts
    from door_live_bench import Net, Zones, RULES
    from door_demo_put import put
    net = S.DoorSeg('cuda'); net.load(torch.load(CK, map_location='cpu', weights_only=False)['model'])
    net.yolo.eval(); net.off.eval(); net.yolo.fuse(verbose=False)
    M = Net(net, 'eager'); asm = F.GPUChannels(S.background(tag.split('_')[1], 'cam1'))
    tracker, zones = Tracker(Config()), Zones()
    bank = DR.Bank(OUT / ('demo2_views_' + tag))
    cap = cv2.VideoCapture(str(ROOT / 'data/sam31_door' / tag / 'cam1/video.mp4'))
    prev = np.zeros((S.H, S.W), np.uint8)
    frames, obs, rows = [], {}, []
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        for tick, ctx in contexts(cap, transform=lambda f: cv2.resize(f, (S.W, S.H), interpolation=cv2.INTER_AREA)):
            pred = M.predict(asm(ctx, prev)); prev = F.previous_union([p for p in pred if p[2] >= .4])
            out = [(pid, p[0][:S.H // 4]) for pid, p in tracker.update(tick, pred) if p[0][:S.H // 4].any()]
            frames.append(cv2.imencode('.jpg', ctx[0], [cv2.IMWRITE_JPEG_QUALITY, 90])[1])
            obs[tick] = out
            if tick % 3 == 0 and out:
                rows += zones.rows(tick, out)
                for pid, m in out:
                    ys, xs = np.nonzero(m); bx = [xs.min() * 4, ys.min() * 4, xs.max() * 4 + 4, ys.max() * 4 + 4]
                    sc = DR.score(int(m.sum()) * 16 * (2176 / 1280) ** 2, True, bx[1] <= 2 or bx[3] >= 716)
                    if bank.wants(pid, sc):
                        full = cv2.resize(m.astype(np.uint8), (S.W, S.H), interpolation=cv2.INTER_NEAREST) > 0
                        bank.offer(pid, sc, lambda: SM.crop_masked(ctx[0], full, bx), bx)
    events, side = Z.run_tag({'rows': rows}, dict(Z.BASE, **RULES))
    person = dict(Z.LAST_PERSON)                       # track -> person (inheritance chain)
    roles = DR.Roles(bank, DR.Worker()); role = {}
    members = {}
    for p_, w in person.items():
        members.setdefault(w, []).append(int(p_))
    for w, ps in members.items():
        role[w] = roles.role(ps).get('role')
    roles.worker.p.kill()
    print('events', [(e['kind'], round(e['t'], 1), e['w'], role.get(e['w'])) for e in events], flush=True)
    Zs = json.load(open(ROOT / 'data/door_v2/door_lines_owner.json', encoding='utf-8'))['lines']
    path = OUT / ('%s_new.mp4' % tag)
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 12.5, (S.W, S.H))
    last, shown = {}, []
    for k, buf in enumerate(frames):
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        for z in Zs:
            cv2.polylines(img, [np.array(z['pts'], np.int32)], z['kind'] == 'poly', (200, 200, 200), 1)
        labels = []
        for pid, m in obs.get(k, []):
            kk = k - k % 3
            if (str(pid), kk) in side:
                last[str(pid)] = side[(str(pid), kk)]
            s = last.get(str(pid))
            col = (60, 200, 60) if s == 1 else (200, 60, 200) if s == 0 else (170, 170, 170)
            full = cv2.resize(m.astype(np.uint8), (S.W, S.H), interpolation=cv2.INTER_NEAREST)
            img[full > 0] = (0.55 * img[full > 0] + 0.45 * np.array(col)).astype(np.uint8)
            cn, _ = cv2.findContours(full, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cn, -1, col, 2)
            w = person.get(str(pid), str(pid)); r = role.get(w)
            ys, xs = np.nonzero(m)
            labels.append(((xs.min() * 4, max(0, ys.min() * 4 - 28)), '#%s %s%s' % (w, 'ВНУТРИ' if s == 1 else 'СНАРУЖИ' if s == 0 else '?',
                           '  сотрудник' if r == 'staff' else '  покупатель' if r == 'customer' else ''), col))
        for e in events:
            if abs(e['t'] * 12.5 - k) < .5:
                shown.append((k, e))
        ban = [('%s  #%s  %s' % ('ВХОД' if e['kind'] == 'in' else 'ВЫХОД', e['w'], {'staff': 'сотрудник', 'customer': 'покупатель'}.get(role.get(e['w']), '')),
                (60, 220, 60) if e['kind'] == 'in' else (255, 140, 30)) for k0, e in shown if k - k0 < 30]
        img = put(img, labels, ban, 'новая система: doorseg_c + зоны   %.1f с' % (k * .08))
        wr.write(img)
    wr.release()
    print('video', path, flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
