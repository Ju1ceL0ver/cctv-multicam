"""A queue of model variants, so the card does not sit on a plateau while nobody watches (29.09, the owner's request).

Every minute: the running variant's teacher test (runs/<run>/epochs.jsonl, F1 after each epoch). On a plateau it is
stopped the gentle way (runs/<run>/STOP: it saves last.pt and ends) and the next variant starts. A variant that dies
on its own is started again from its last.pt (twice at most), then skipped.

Plateau: the 3-epoch mean of the test F1 has not risen by more than DELTA over the last WINDOW epochs (after
min_steps), or max_steps is reached. The queue (data/v2_queue/queue.json) is read every minute: items can be added or
edited while it runs. State for the /train page: data/v2_queue/state.json; decisions: data/logs/variant_queue.log.

An item: name (-> runs/<name>), attach (already running: only watched), from ('winner' -- the better of RepViT and
ViT at the same step; 'best' -- the highest test F1 of all finished runs; 'prev' -- the previous item; or a run
name), mode ('resume': that run's last.pt goes on with new data and rates; 'init': a new model that starts from its
weights wherever names and shapes match), steps_more (resume) or steps (init), args (extra training arguments,
override the defaults), backbone / layers / slim (init; default: the source's), plateau (false: runs to the end),
min_steps, max_steps, abort_at [step, f1] (stopped at that step when its best F1 is lower).

usage: variant_queue.py            (bg.spawn('variantq', ['variant_queue.py']))"""
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
Q = ROOT / 'data' / 'v2_queue'
RUNS = ROOT / 'runs'
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
WINDOW, DELTA = 6, 0.005
COMMON = ['--heldout', '20260923_21601', '20260923_25201', '20260923_01801', '--test', 'v2b', '--track-every', '3',
          '--track-ticks', '500', '--owner-exam-every', '6', '--keep-every', '5', '--epoch-steps', '300', '--workers', '3',
          '--far-workers', '1', '--draft-workers', '2', '--far', '0.3', '--T', '4', '--ema', '0.9995', '--vit-lr-mult', '0.26667',
          '--draft-items', '3']
VIT_RUN = 'v2_b'                       # the ViT run of the morning, stopped at step 7000


def say(*a):
    line = time.strftime('%m-%d %H:%M:%S ') + ' '.join(str(x) for x in a)
    print(line, flush=True)
    with open(ROOT / 'data' / 'logs' / 'variant_queue.log', 'a', encoding='utf-8') as f:
        f.write(line + '\n')
    return line


def epochs(run):
    out = []
    p = RUNS / run / 'epochs.jsonl'
    if p.exists():
        for l in open(p, encoding='utf-8'):
            try:
                e = json.loads(l)
            except ValueError:
                continue
            f1 = (e.get('test') or {}).get('f1')
            if f1 is not None:
                out.append((e['step'], f1))
    return out


def smooth(pts):
    f = [x for _, x in pts]
    return [statistics.mean(f[max(0, i - 2):i + 1]) for i in range(len(f))]


def plateau(pts):
    if len(pts) < WINDOW + 3:
        return False
    s = smooth(pts)
    return max(s[-WINDOW:]) < max(s[:-WINDOW]) + DELTA


def best(run):
    pts = epochs(run)
    return max(pts, key=lambda p: p[1]) if pts else (None, None)


def status(run):
    try:
        return json.load(open(RUNS / run / 'status.json'))
    except Exception:
        return {}


def args_of(run):
    try:
        return json.load(open(RUNS / run / 'args.json'))
    except Exception:
        return {}


def procs(run):
    """PIDs of the training launchers of this run (the venv's python.exe that started train_v2.py)."""
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { ($_.CommandLine -like '*train_v2.py*' "
          "-or $_.CommandLine -like '*yolo_baseline.py*') -and $_.CommandLine -like '*runs/%s *' } | "
          "ForEach-Object { \"$($_.ProcessId)|$($_.ParentProcessId)\" }" % run)
    out = subprocess.run(['powershell', '-NoProfile', '-Command', ps], capture_output=True, text=True).stdout.split()
    pairs = [tuple(int(x) for x in l.split('|')) for l in out if '|' in l]
    pids = {p for p, _ in pairs}
    return [p for p, parent in pairs if parent not in pids]          # the tree roots


def alive(run):
    return bool(procs(run))


def stop(run, why):
    say('stop', run, '--', why)
    (RUNS / run / 'STOP').write_text(why, encoding='utf-8')
    t = time.time()
    while alive(run) and time.time() - t < 1500:                       # a test in progress can take a few minutes
        time.sleep(20)
    for pid in procs(run):                                             # did not stop: the whole tree, not by name
        say('kill tree', run, pid)
        subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True)
    (RUNS / run / 'STOP').unlink(missing_ok=True)


def step_of(ckpt):
    import torch
    return int(torch.load(ckpt, map_location='cpu', weights_only=False).get('step', 0))


def arch_args(src_args, item):
    bb = item.get('backbone', src_args.get('backbone') or '')
    layers = item.get('layers', src_args.get('decoder_layers', 6))
    slim = item.get('slim', src_args.get('slim', False))
    return (['--backbone', bb] if bb else []) + ['--decoder-layers', str(layers)] + (['--slim'] if slim else [])


def source(item, st):
    f = item.get('from', 'prev')
    if f == 'winner':
        return st.get('winner') or VIT_RUN
    if f == 'best':                                                 # a borrowed machine's run counts once its best.pt is here
        runs = [i['name'] for i in st['items'] if i.get('state') in ('done', 'stopped') and (RUNS / i['name'] / 'best.pt').exists()] + [VIT_RUN]
        scored = [(best(r)[1] or -1, r) for r in runs]
        return max(scored)[1]
    if f == 'prev':
        done = [i['name'] for i in st['items'] if i.get('state') in ('done', 'stopped') and i.get('where') != 'kaggle']
        return done[-1] if done else VIT_RUN
    return f


def command(item, st, resume_own=False):
    """The training command of an item (train_v2.py and its arguments; kind 'yolo': the yardstick, yolo_baseline.py)."""
    name = item['name']
    run = 'runs/%s' % name
    if item.get('kind') == 'yolo':
        return ['yolo_baseline.py', '--out', run, '--model', item.get('model', 'runs/student_seg_all_s/weights/best.pt'),
                '--epochs', str(item.get('epochs', 10)), '--imgsz', str(item.get('imgsz', 1088)), '--batch', str(item.get('batch', 6)),
                '--test', 'v2b']
    cmd = ['train_v2.py', '--out', run] + COMMON
    if resume_own:                                                     # a crash: go on from its own last.pt
        a = args_of(name)
        cmd += ['--resume', '%s/last.pt' % run, '--steps', str(a.get('steps')), '--lr', str(a.get('lr')), '--lr-floor', str(a.get('lr_floor', 0.0)),
                '--drafts', str(a.get('drafts', 0.15)), '--draft-only-steps', str(a.get('draft_only_steps', 0)),
                '--freeze-steps', str(a.get('freeze_steps', 0))] + (['--freeze-vit-only'] if a.get('freeze_vit_only') else [])
        if a.get('fresh_sched'):
            cmd += ['--fresh-sched', '--sched-from', str(item.get('sched_from', -1))]
        cmd += arch_args(a, {})
    else:
        src = source(item, st)
        item['source'] = src
        sa = args_of(src)
        if item.get('mode') == 'resume':
            ck = RUNS / src / 'last.pt'
            if not ck.exists():
                ck = RUNS / src / 'best.pt'                          # a run from the borrowed machine sends only its best
            s0 = step_of(ck)
            item['sched_from'] = s0
            cmd += ['--resume', 'runs/%s/%s' % (src, ck.name), '--steps', str(s0 + item.get('steps_more', 3000)), '--fresh-sched',
                    '--lr', str(item.get('lr', 1.5e-4)), '--lr-floor', '0', '--drafts', str(item.get('drafts', 0.15)),
                    '--draft-only-steps', '0', '--freeze-steps', '0'] + arch_args(sa, {})
        else:
            ck = RUNS / src / 'best.pt'
            if not ck.exists():
                ck = RUNS / src / 'last.pt'
            cmd += ['--init', 'runs/%s/%s' % (src, ck.name), '--steps', str(item.get('steps', 6000)), '--lr', str(item.get('lr', 3e-4)),
                    '--lr-floor', '0.05', '--drafts', str(item.get('drafts', 0.15)), '--draft-only-steps', str(item.get('draft_only', 0)),
                    '--freeze-steps', str(item.get('freeze_steps', 0)), '--freeze-vit-only'] + arch_args(sa, item)
    cmd += item.get('args', [])
    return cmd


def launch(item, st, resume_own=False):
    import bg
    name = item['name']
    cmd = command(item, st, resume_own)
    bg._python = lambda: RF
    say('start', name, 'from', item.get('source', name), ':', ' '.join(cmd[1:]))
    bg.spawn(name, cmd, wait=2)
    item.update(state='running', started=time.strftime('%H:%M'), launches=item.get('launches', 0) + 1)


def pick_winner(st):
    """RepViT against ViT at the same step (the 3-epoch mean of the test F1); the lighter one wins a tie."""
    r = epochs('v2_c_repvit')
    v = epochs(VIT_RUN)
    if not r:
        return VIT_RUN
    rs = smooth(r)[-1]
    step = r[-1][0]
    at = [i for i, (s, _) in enumerate(v) if s <= step] or [0]
    vs = smooth(v)[at[-1]]
    w = 'v2_c_repvit' if rs >= vs - DELTA else VIT_RUN
    say('winner', w, ': RepViT %.4f at step %d, ViT %.4f at step %d' % (rs, step, vs, v[at[-1]][0]))
    return w


def save(st):
    Q.mkdir(parents=True, exist_ok=True)
    for it in st['items']:
        b = best(it['name'])
        pts = epochs(it['name'])
        it['best_f1'], it['best_step'] = b[1], b[0]
        it['last_f1'], it['last_step'] = (pts[-1][1], pts[-1][0]) if pts else (None, None)
    st['updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    tmp = Q / 'state.tmp.json'
    json.dump(st, open(tmp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    os.replace(tmp, Q / 'state.json')


def main():
    say('queue up')
    st = json.load(open(Q / 'state.json', encoding='utf-8')) if (Q / 'state.json').exists() else {'items': []}
    while True:
        queue = json.load(open(Q / 'queue.json', encoding='utf-8'))
        known = {i['name']: i for i in st['items']}
        st['items'] = [dict(q, **{k: v for k, v in known.get(q['name'], {}).items() if k not in q}) for q in queue]
        remote = json.load(open(Q / 'remote.json', encoding='utf-8')) if (Q / 'remote.json').exists() else {}
        for it in st['items']:
            if it.get('where') == 'kaggle' and it['name'] in remote:
                it.update(remote[it['name']])
                seen = RUNS / it['name'] / 'remote.json'
                t = json.load(open(seen)).get('t', 0) if seen.exists() else 0
                if it.get('state') == 'running' and time.time() - t > 1800:
                    it.update(state='failed', reason='Kaggle молчит больше 30 минут')
        cur = next((i for i in st['items'] if i.get('state') == 'running' and i.get('where') != 'kaggle'), None)
        if cur is None:
            att = next((i for i in st['items'] if i.get('attach') and i.get('state') is None), None)
            if att is not None and alive(att['name']):
                att.update(state='running', started='(attached)')
                cur = att
                say('watching', att['name'])
        if cur is None:
            nxt = next((i for i in st['items'] if i.get('state') is None and not i.get('attach') and i.get('where') != 'kaggle'), None)
            if nxt is None:
                st['current'] = None
                save(st)
                time.sleep(60)
                continue
            if st.get('winner') is None and any(i.get('name') == 'v2_c_repvit' and i.get('state') in ('done', 'stopped') for i in st['items']):
                st['winner'] = pick_winner(st)
            try:
                launch(nxt, st)
            except Exception as e:
                nxt.update(state='failed', reason='launch: ' + repr(e)[:200])
                say('launch failed', nxt['name'], repr(e)[:200])
            st['current'] = nxt['name']
            save(st)
            time.sleep(120)
            continue
        name = cur['name']
        st['current'] = name
        pts = epochs(name)
        s = status(name)
        steps_total = args_of(name).get('steps') or s.get('steps') or 0
        last_step = pts[-1][0] if pts else 0
        why = None
        if cur.get('abort_at') and last_step >= cur['abort_at'][0] and (best(name)[1] or 0) < cur['abort_at'][1]:
            why = 'best F1 %.3f < %.3f at step %d' % (best(name)[1] or 0, cur['abort_at'][1], last_step)
        elif cur.get('plateau', True) and last_step >= cur.get('min_steps', 3000) and plateau(pts):
            sm = smooth(pts)
            why = 'plateau: 3-epoch mean %.4f, best before the last %d epochs %.4f' % (max(sm[-WINDOW:]), WINDOW, max(sm[:-WINDOW]))
        elif cur.get('max_steps') and last_step >= cur['max_steps']:
            why = 'max steps %d' % cur['max_steps']
        if why:
            stop(name, why)
            cur.update(state='stopped', reason=why, ended=time.strftime('%H:%M'))
        elif not alive(name):
            if steps_total and max(s.get('step', 0), last_step) >= steps_total - (20 if steps_total > 1000 else 0):
                cur.update(state='done', reason='all %d steps' % steps_total, ended=time.strftime('%H:%M'))
                say('done', name)
            elif s.get('phase') == 'stopped':
                cur.update(state='stopped', reason=cur.get('reason') or 'stopped by hand', ended=time.strftime('%H:%M'))
                say('stopped by hand', name)
            elif cur.get('launches', 1) < 3 and (RUNS / name / 'last.pt').exists():
                say('died at step', s.get('step'), '-- again from its last.pt', name)
                try:
                    launch(cur, st, resume_own=True)
                except Exception as e:
                    cur.update(state='failed', reason='relaunch: ' + repr(e)[:200])
                time.sleep(120)
            else:
                cur.update(state='failed', reason='died at step %s' % s.get('step'), ended=time.strftime('%H:%M'))
                say('failed', name)
        save(st)
        time.sleep(60)


if __name__ == '__main__':
    main()
