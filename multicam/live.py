"""Live labelling (the owner's plan of 01.10): the model picks the frames it is least sure of, the owner corrects them
on /fix, the model learns from what he corrected, and picks again.

One round:
  1. sample  SAMPLE frames at random from what he has not seen: ticks of the SAM 3.1 training windows and SAM 3.1's
             single varied frames (never 18.09 or 23.09)
  2. rank    the current model looks at each: people it is unsure of (confidence 0.15-0.5), SAM's people it does not
             find, things it calls people that SAM does not -- the most of these first; the top SHOW go to /fix as a
             new batch "живая N", in that order, with SAM's outline as the draft
  3. train   as soon as he has answered ANSWERED of a batch: the model is fine-tuned softly (train_v2 det-only, rate
             2e-5) on his taken training frames only -- none of the 13 thousand drafts -- and scored on his whole test
             after every epoch; better weights become current; only then the next batch is picked, by the new model
             (the owner's order: the hard frames come from the model that has learned his last answers)
The state (/fix shows it): data/fix/live.json. Stop: data/live/STOP.

usage: live.py loop [--start CKPT]      (the GPU: started only when the owner says so)
       live.py round                    one sample-rank-batch, nothing trained"""
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
LIVE = ROOT / 'data' / 'live'
STATE = ROOT / 'data' / 'fix' / 'live.json'
SAMPLE, SHOW, ANSWERED = 600, 50, 0.9
SKIP_DAYS = ('20260918', '20260923')
SPACE_S = 10.0
START = 'runs/v2_j_owner/best.pt'
STEPS, EPOCH, LR = 300, 150, 0.00002
COMMON = ['--heldout', '20260923_21601', '20260923_25201', '20260923_01801', '--test', 'fix', '--track-every', '1000',
          '--owner-exam-every', '1000', '--keep-every', '1000', '--workers', '1', '--far-workers', '1', '--draft-workers', '3',
          '--far', '0', '--T', '1', '--clips', '3', '--ema', '0.999', '--vit-lr-mult', '0.26667', '--draft-items', '3',
          '--draft-only-steps', '0', '--freeze-steps', '0', '--freeze-vit-only', '--backbone', 'repvit_m0_9.dist_450e_in1k',
          '--decoder-layers', '6', '--det-only', '--drafts', '1.0', '--lr-floor', '0.1']


def log(*a):
    LIVE.mkdir(parents=True, exist_ok=True)
    line = time.strftime('%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    with open(LIVE / 'live.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    print(line, flush=True)


def state():
    return json.load(open(STATE, encoding='utf-8')) if STATE.exists() else {'rounds': [], 'current': START, 'best_f1': None}


def save_state(st):
    from storage import atomic_json
    atomic_json(STATE, st)


# ---------------------------------------------------------------- candidates

def candidates(seed):
    """SAMPLE frames at random: half window ticks with somebody (spaced from what /fix already holds), half single
    frames. -> [dict(kind='win', tag, cam, tick) | dict(kind='still', id, img, cam)]"""
    import fix
    import rate
    import v2_data as VD
    rng = random.Random(seed)
    have = fix.manifest()['items']
    seen = {}
    for it in have:
        if it.get('tag'):
            seen.setdefault((it['tag'], it['cam']), []).append(it['tick'])
    ids = {it['id'] for it in have}
    wins = []
    for tag, cam in VD.window_cams():
        if tag.startswith(SKIP_DAYS):
            continue
        z = np.load(VD.V2 / tag / cam / 'rows.npz')
        ticks = np.unique(z['tick']).tolist()
        near = seen.get((tag, cam), [])
        wins += [(tag, cam, int(t)) for t in ticks if all(abs(t - u) * VD.TICK >= SPACE_S for u in near)]
    still_dir = ROOT / 'data' / 'sam31_stills' / 'drafts'
    stills = [(i, str(p)) for i, (p, _) in rate.index(str(ROOT), force=True).items()
              if not i.startswith(SKIP_DAYS) and i not in ids and (still_dir / ('%s.png' % i)).exists()]
    out = [{'kind': 'win', 'tag': t, 'cam': c, 'tick': k} for t, c, k in rng.sample(wins, min(len(wins), SAMPLE // 2))]
    out += [{'kind': 'still', 'id': i, 'img': p, 'cam': i.split('_')[1]} for i, p in rng.sample(stills, min(len(stills), SAMPLE - len(out)))]
    return out


def frame_of(c, wins, drafts):
    """-> (image BGR, bg_long, bg_now, SAM's people on the stride-4 grid) of a candidate."""
    import v2_data as VD
    if c['kind'] == 'win':
        k = (c['tag'], c['cam'])
        if k not in wins:
            for w in wins.values():
                if w.cap is not None:
                    w.cap.release(); w.cap = None
            wins[k] = VD.Window(*k)
        w = wins[k]
        img = w.frame(c['tick'])
        if img is None:
            return None
        return img, w.bg_long(c['tick']), np.array(w.bg_now(c['tick'])), w.targets(c['tick'])['masks']
    img = cv2.imread(c['img'])
    lab = cv2.imread(str(ROOT / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % c['id'])), cv2.IMREAD_UNCHANGED)
    parts = c['id'].split('_')
    bg = drafts.bg(parts[0], parts[1], '_'.join(parts[2:4]))
    if img is None or lab is None or bg is None:
        return None
    return img, bg, bg, VD.draft_targets(lab, c['cam'], None)['masks']


def rank(ckpt, cands):
    """The current model on each candidate -> the same list with 'hard' (unsure + SAM's missed + its own extra)."""
    import torch
    import slot_v2 as V
    import train_slots as TS
    import v2_data as VD
    import v2_eval
    from scipy.optimize import linear_sum_assignment
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = V.load(ROOT / ckpt).to(dev).eval()
    asm = VD.Assembler(dev, getattr(model, 'rgb_size', None))
    wins, drafts = {}, VD.Drafts(0)
    out = []
    for c in cands:
        got = frame_of(c, wins, drafts)
        if got is None:
            continue
        img, bgl, bgn, G = got
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=dev == 'cuda'):
            rgb, bgv, sta, cam_ids = asm([{'img': img, 'bg_long': bgl, 'bg_now': bgn}], (c['cam'],), False)
            bg_sem = model.body.vit(bgv)[0]
            m = model.maps(rgb, sta, cam_ids, bg_sem, asm.last_world)
            r = model.decode(m, cam_ids)
            r, _ = model.cross_cameras(r, v2_eval.empty_like(r))
            P = (model.full_masks(m, r)[0] > 0).cpu().numpy()
            p = r['obj'][0].float().sigmoid().cpu().numpy()
            pad = r['pad'][0].cpu().numpy()
        P = P[:, :VD.GH, :VD.GW].reshape(len(P), -1)
        ok = [s for s in range(len(p)) if not pad[s] and P[s].sum() >= 20 and p[s] >= 0.15]
        ok = [ok[i] for i in TS.dedup([P[j] for j in ok], [p[j] for j in ok])]
        unsure = sum(1 for j in ok if p[j] < 0.5)
        keep = [j for j in ok if p[j] >= 0.3]
        G = G.reshape(len(G), -1)
        hit = 0
        if len(G) and keep:
            Gf, Pf = G.astype(np.float32), P[keep].astype(np.float32)
            inter = Gf @ Pf.T
            iou = inter / np.maximum(Gf.sum(1)[:, None] + Pf.sum(1)[None] - inter, 1)
            rr, cc = linear_sum_assignment(-iou)
            hit = int(sum(iou[i, j] >= 0.5 for i, j in zip(rr, cc)))
        c = dict(c, hard=unsure + (len(G) - hit) + (len(keep) - hit), unsure=unsure, sam=len(G), model=len(keep), both=hit)
        out.append(c)
    for w in wins.values():
        if w.cap is not None:
            w.cap.release()
    return out


def add_batch(n, picked):
    """The picked candidates -> /fix batch n "живая n", ordered by 'rank' (the hardest first)."""
    import fix
    from storage import atomic_json
    name = 'живая %d' % n
    fix.add([(c['tag'], c['cam'], c['tick']) for c in picked if c['kind'] == 'win'], n, name, 'train', log)
    out = fix.folder()
    m = fix.manifest()
    have = {it['id'] for it in m['items']}
    for c in picked:
        if c['kind'] != 'still' or c['id'] in have:
            continue
        img = cv2.imread(c['img'])
        lab = cv2.imread(str(ROOT / 'data' / 'sam31_stills' / 'drafts' / ('%s.png' % c['id'])), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(out / ('%s.jpg' % c['id'])), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(out / ('%s_init.png' % c['id'])), lab)
        m['items'].append({'id': c['id'], 'batch': n, 'set': 'train', 'src': 'still', 'cam': c['cam'], 'tag': None, 'tick': None,
                           'w': int(img.shape[1]), 'h': int(img.shape[0]), 'people': int(len([v for v in np.unique(lab) if v])), 'persons': []})
    order = {}
    for k, c in enumerate(picked):
        order[c['id'] if c['kind'] == 'still' else '%s_%s_%05d' % (c['tag'], c['cam'], c['tick'])] = k
    for it in m['items']:
        if it['id'] in order and it['batch'] == n:
            it['rank'] = order[it['id']]
    m['batches'][str(n)] = name
    atomic_json(out / 'manifest.json', m)


def new_batch(st):
    import fix
    n = max([int(b) for b in fix.manifest()['batches']] + [1]) + 1
    t0 = time.time()
    cands = candidates(seed=n)
    scored = rank(st['current'], cands)
    scored.sort(key=lambda c: (-c['hard'], -c['unsure']))
    picked = scored[:SHOW]
    add_batch(n, picked)
    log('batch', n, 'from', len(scored), 'sampled: hardest', [c['hard'] for c in picked[:10]], '...', picked[-1]['hard'] if picked else None,
        '%.0f s' % (time.time() - t0))
    st['rounds'].append({'batch': n, 'made': time.strftime('%H:%M'), 'sampled': len(scored), 'shown': len(picked),
                         'hard_mean': round(float(np.mean([c['hard'] for c in picked])), 2) if picked else 0, 'model': st['current']})
    save_state(st)
    return n


def answered(n):
    import fix
    m, sv = fix.manifest(), fix.state()
    its = [it for it in m['items'] if it['batch'] == n]
    return sum(1 for it in its if sv.get(it['id'], {}).get('verdict')) / max(1, len(its))


def train_round(st, n):
    run = 'runs/live_r%02d' % n
    args = ['train_v2.py', '--out', run, '--init', st['current'], '--steps', str(STEPS), '--epoch-steps', str(EPOCH), '--lr', str(LR)] + COMMON
    st['phase'] = 'учусь на ваших кадрах (круг %d)' % n
    save_state(st)
    t0 = time.time()
    r = subprocess.run([sys.executable] + args, cwd=str(ROOT), env=dict(os.environ, RA_V2_GOLD='1', RA_V2_DRAFTS='sam31'),
                       capture_output=True, text=True)
    f1 = []
    p = ROOT / run / 'epochs.jsonl'
    if p.exists():
        f1 = [json.loads(l).get('test', {}).get('f1') for l in open(p)]
    best = max([x for x in f1 if x is not None] or [None]) if f1 else None
    log('trained on batch', n, 'F1 by epoch', f1, 'rc', r.returncode, '%.0f s' % (time.time() - t0), r.stderr[-300:] if r.returncode else '')
    rec = next(x for x in st['rounds'] if x['batch'] == n)
    rec.update(f1=f1, trained=time.strftime('%H:%M'), taken=frames_taken())
    if best is not None and (st.get('best_f1') is None or best >= st['best_f1']) and (ROOT / run / 'best.pt').exists():
        st['current'], st['best_f1'] = run + '/best.pt', best
        rec['kept'] = True
    st['phase'] = 'жду ваших ответов'
    save_state(st)


def frames_taken():
    import v2_teacher_test as TT
    return len(TT.fix_items('take', 'train'))


def baseline(st):
    """The starting model's score on his test, once (what each round is compared with)."""
    import slot_v2 as V
    import v2_teacher_test as TT
    import torch
    m = V.load(ROOT / st['current']).cuda()
    r = TT.Test('fix').run(m, 'cuda')
    st['best_f1'] = st['start_f1'] = r['f1']
    st['start_test'] = {k: r[k] for k in ('recall', 'precision', 'f1', 'mask_iou_median')}
    del m
    torch.cuda.empty_cache()
    save_state(st)
    log('start model', st['current'], st['start_test'])


def drop_stale(st):
    """Batches nobody has touched yet that an older model picked: out of /fix, to be picked again by the current one."""
    import fix
    from storage import atomic_json
    m = fix.manifest()
    open_ = [x for x in st['rounds'] if 'trained' not in x]
    stale = [x['batch'] for k, x in enumerate(open_) if answered(x['batch']) == 0
             and (x.get('model') != st['current'] or k > 0)]     # k > 0: picked before the batch ahead of it was learned
    if not stale:
        return
    m['items'] = [it for it in m['items'] if it['batch'] not in stale]
    for b in stale:
        m['batches'].pop(str(b), None)
    atomic_json(fix.folder() / 'manifest.json', m)
    st['rounds'] = [x for x in st['rounds'] if x['batch'] not in stale]
    save_state(st)
    log('dropped batches picked by an older model', stale)


def loop(start=None):
    st = state()
    if start:
        st['current'] = start
    if st.get('best_f1') is None:
        baseline(st)
    drop_stale(st)
    pending = [x['batch'] for x in st['rounds'] if 'trained' not in x]
    while not (LIVE / 'STOP').exists():
        if not pending:
            st['phase'] = 'подбираю трудные кадры новой моделью'
            save_state(st)
            pending = [new_batch(st)]
            st['phase'] = 'жду ваших ответов'
            save_state(st)
        n = pending[0]
        if answered(n) >= ANSWERED:
            train_round(st, n)                       # learn his answers first ...
            pending.pop(0)
            st = state()                             # ... the next batch is picked at the top, by the new model
            continue
        time.sleep(20)
    (LIVE / 'STOP').unlink()
    st['phase'] = 'остановлено'
    save_state(st)
    log('stopped')


if __name__ == '__main__':
    os.chdir(ROOT)
    if sys.argv[1] == 'loop':
        loop(sys.argv[sys.argv.index('--start') + 1] if '--start' in sys.argv else None)
    elif sys.argv[1] == 'round':
        new_batch(state())
