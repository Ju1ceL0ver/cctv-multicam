"""One command on a borrowed machine with video cards (Kaggle, GPU T4 x2, internet on): the shop's training there.

  speed test, the small kit through the tunnel (kit_build.py):
    !curl -sfL "SERVER/api/jobs/kit?key=KEY" -o kit_run.py && python kit_run.py SERVER KEY [STEPS] [BACKBONE]

  a variant of the queue, the full kit from an encrypted archive (kit_build.py full):
    !curl -sfL "SERVER/api/jobs/kit?key=KEY" -o kit_run.py && python kit_run.py SERVER KEY --archive SRC --password PW
    SRC: a folder with kit_full.7z.001 ... (an attached Kaggle dataset: /kaggle/input/NAME) or their links, space-separated

With --archive it takes the next queue item marked "where": "kaggle" (data/v2_queue/queue.json on the shop machine),
trains it on every card (torchrun, fp16: a T4 has no bf16), sends its logs and tests to the server every 30 s (the
/train page and the queue table show them), stops it on a plateau by the queue's rule, then sends its best.pt back
and copies the run to /kaggle/working. For a long run: Save Version -> Save & Run All (runs without the browser)."""
import concurrent.futures as cf
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

ARGV = sys.argv[1:]
SERVER, KEY = ARGV[0].rstrip('/'), ARGV[1]


def flag(name, default=None):
    return ARGV[ARGV.index(name) + 1] if name in ARGV else default


ARCHIVE, PASSWORD = flag('--archive'), flag('--password')
POS = [a for i, a in enumerate(ARGV[2:], 2) if not a.startswith('--') and ARGV[i - 1] not in ('--archive', '--password')]
STEPS = int(POS[0]) if POS else 300
BACKBONE = POS[1] if len(POS) > 1 else 'repvit_m0_9.dist_450e_in1k'
KAGGLE = os.path.isdir('/kaggle/working')
HOME = os.environ.get('KIT_HOME', ('/tmp/kit' if ARCHIVE else '/kaggle/working/kit') if KAGGLE else os.path.abspath('kit'))
KEEP = '/kaggle/working' if KAGGLE else os.path.abspath('kit_out')
WHO = os.environ.get('KIT_WHO', 'kaggle')
FRAMES = {'draft': 6, 'clip': 8, 'far': 4}           # frames one step of each kind trains on (per card)
WINDOW, DELTA = 6, 0.005                             # the queue's plateau rule (variant_queue.py)
KAGGLE_WORKERS = ['--workers', '2', '--far-workers', '1', '--draft-workers', '1']   # 4 processor cores for two cards
CODE = ['train_v2.py', 'slot_v2.py', 'slot_model.py', 'deimv2_vit.py', 'train_slots.py', 'v2_data.py', 'v2_fast.py', 'v2_eval.py',
        'v2_track.py', 'v2_teacher_test.py', 'sam31_reid.py', 'gold.py']      # fetched fresh each time: the archive keeps the data


def url(path, **q):
    q['key'] = KEY
    return SERVER + path + '?' + urllib.parse.urlencode(q)


def call(path, body=None, data=None, timeout=60, **q):
    headers = {'Content-Type': 'application/json'} if body is not None else {'Content-Type': 'application/octet-stream'}
    payload = json.dumps(body).encode() if body is not None else data
    req = urllib.request.Request(url(path, **q), data=payload, headers=headers)
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read() or b'{}')


def report(**kw):
    kw.update(who=WHO, t=round(time.time(), 1))
    try:
        call('/api/jobs/kit/report', kw, timeout=30)
    except Exception as e:
        print('report failed:', repr(e)[:120], flush=True)


# ---------------------------------------------------------------- the small kit through the tunnel
def fetch(item):
    dst = os.path.join(HOME, item['dst'])
    if os.path.exists(dst) and os.path.getsize(dst) == item['size']:
        return 0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + '.part'
    for attempt in range(10):
        have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        req = urllib.request.Request(url('/api/jobs/kit/file', p=item['src']), headers={'Range': 'bytes=%d-' % have} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                with open(tmp, 'ab' if have and r.status == 206 else 'wb') as f:     # 200: the server sent it all again
                    shutil.copyfileobj(r, f, 1 << 20)
            if os.path.getsize(tmp) == item['size']:
                os.replace(tmp, dst)
                return item['size']
        except Exception as e:
            print('retry', item['dst'], repr(e)[:100], flush=True)
        time.sleep(min(30, 3 * (attempt + 1)))
    raise RuntimeError('could not fetch ' + item['dst'])


def download_small():
    man = json.load(urllib.request.urlopen(url('/api/jobs/kit/manifest'), timeout=60))
    items = man['files']
    total = sum(i['size'] for i in items)
    print('kit: %d files, %.2f GB (windows %s, %d drafts)' % (len(items), total / 1e9, man['windows'], man['drafts']), flush=True)
    t0, done, lock = time.time(), [0], threading.Lock()

    def one(it):
        n = fetch(it)
        with lock:
            done[0] += it['size']
        return n
    last = 0
    with cf.ThreadPoolExecutor(6) as ex:
        for f in cf.as_completed([ex.submit(one, it) for it in sorted(items, key=lambda i: -i['size'])]):
            f.result()
            if time.time() - last > 15:
                last = time.time()
                mb_s = done[0] / 1e6 / max(1e-6, time.time() - t0)
                print('  %.0f %% (%.2f of %.2f GB), %.1f MB/s' % (100 * done[0] / total, done[0] / 1e9, total / 1e9, mb_s), flush=True)
                report(phase='download', done_gb=round(done[0] / 1e9, 3), total_gb=round(total / 1e9, 3), mb_s=round(mb_s, 2))
    s = time.time() - t0
    print('downloaded in %.0f s (%.1f MB/s)' % (s, total / 1e6 / max(1e-6, s)), flush=True)
    return total, s


# ---------------------------------------------------------------- the full kit from the encrypted archive
def seven_zip():
    for exe in ('7z', '7za', '7zz'):
        if shutil.which(exe):
            return shutil.which(exe)
    subprocess.run(['apt-get', 'install', '-y', '-q', 'p7zip-full'], capture_output=True)
    for exe in ('7z', '7za'):
        if shutil.which(exe):
            return shutil.which(exe)
    return None


def get_archive(src, password):
    t0 = time.time()
    if os.path.isdir(src):
        vols = sorted(glob.glob(os.path.join(src, '**', 'kit_full.7z.0*'), recursive=True))
    else:
        links = src.replace(',', ' ').split()
        os.makedirs('/tmp/kitarc', exist_ok=True)
        vols = [os.path.join('/tmp/kitarc', 'kit_full.7z.%03d' % (k + 1)) for k in range(len(links))]

        def get(k):
            if not os.path.exists(vols[k]):
                print('downloading volume', k + 1, flush=True)
                subprocess.run(['curl', '-sfL', '--retry', '10', '-o', vols[k] + '.part', links[k]], check=True)
                os.replace(vols[k] + '.part', vols[k])
        with cf.ThreadPoolExecutor(3) as ex:
            list(ex.map(get, range(len(links))))
    if not vols:
        raise SystemExit('no kit_full.7z.001 ... in ' + src)
    gb = sum(os.path.getsize(v) for v in vols) / 1e9
    print('archive: %d volumes, %.2f GB (%.0f s)' % (len(vols), gb, time.time() - t0), flush=True)
    report(phase='archive', volumes=len(vols), gb=round(gb, 2), s=round(time.time() - t0))
    os.makedirs(HOME, exist_ok=True)
    z = seven_zip()
    if z:
        r = subprocess.run([z, 'x', '-y', '-p' + password, '-o' + HOME, vols[0]], capture_output=True, text=True)
        if r.returncode:
            raise SystemExit('7z: ' + (r.stdout + r.stderr)[-600:])
    else:                                              # no 7-Zip: the pure-Python reader (AES and all)
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'py7zr', 'multivolumefile'], check=True)
        import multivolumefile
        import py7zr
        with multivolumefile.open(vols[0][:-4], mode='rb') as vol:
            with py7zr.SevenZipFile(vol, 'r', password=password) as a:
                a.extractall(HOME)
    if not os.path.isdir(src):
        shutil.rmtree('/tmp/kitarc', ignore_errors=True)            # the disk is needed for the masks
    print('unpacked into %s (%.0f s)' % (HOME, time.time() - t0), flush=True)
    report(phase='unpacked', s=round(time.time() - t0))


# ---------------------------------------------------------------- running and watching
def gpus():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout
        return [l.strip() for l in out.strip().splitlines() if l.strip()]
    except Exception:
        return []


def lines(path, start=0):
    if not os.path.exists(path):
        return [], start
    with open(path, encoding='utf-8') as f:
        ls = f.read().splitlines()
    out = []
    for l in ls[start:]:
        try:
            out.append(json.loads(l))
        except ValueError:
            pass
    return out, len(ls)


def summary(rs, world):
    """Wall-clock seconds per step over the logged rows (a row every 20 steps), per kind of step."""
    rs = [r for r in rs if 'step' in r and 't' in r]
    if len(rs) < 2:
        return {}
    per = [(b['t'] - a['t']) / max(1, b['step'] - a['step']) for a, b in zip(rs, rs[1:]) if b['step'] > a['step'] and b['t'] > a['t']]
    kinds = {}
    for r in rs[1:]:
        kinds.setdefault(r.get('kind', '?'), []).append(r['gpu_s'] + r['data_s'])
    med = statistics.median(per[-10:]) if per else None
    mix = sum(FRAMES.get(r.get('kind'), 6) for r in rs) / len(rs)
    return {'step': rs[-1]['step'], 's_per_step': round(med, 2) if med else None,
            'frames_per_s': round(mix * world / med, 2) if med else None,
            'kind_s': {k: round(statistics.median(v), 2) for k, v in kinds.items()},
            'data_wait_s': round(statistics.median([r['data_s'] for r in rs[1:]]), 2), 'mem_gb': rs[-1].get('mem_gb')}


def plateau(eps, item):
    pts = [(e['step'], e['test']['f1']) for e in eps if (e.get('test') or {}).get('f1') is not None]
    if not pts:
        return None
    last = pts[-1][0]
    if item.get('max_steps') and last >= item['max_steps']:
        return 'max steps %d' % item['max_steps']
    if not item.get('plateau', True) or last < item.get('min_steps', 3000) or len(pts) < WINDOW + 3:
        return None
    f = [x for _, x in pts]
    s = [statistics.mean(f[max(0, i - 2):i + 1]) for i in range(len(f))]
    if max(s[-WINDOW:]) < max(s[:-WINDOW]) + DELTA:
        return 'plateau: 3-epoch mean %.4f, best before the last %d epochs %.4f' % (max(s[-WINDOW:]), WINDOW, max(s[:-WINDOW]))
    return None


def upload(run, path):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    off, part = 0, 48 << 20                                   # under the tunnel's 100 MB a request
    with open(path, 'rb') as f:
        while off < size:
            f.seek(off)
            chunk = f.read(part)
            for attempt in range(8):
                try:
                    r = call('/api/jobs/kit/upload', data=chunk, timeout=300, run=run, file=name, offset=off, final=int(off + len(chunk) >= size))
                    break
                except urllib.error.HTTPError as e:
                    if e.code == 409:                         # the server has a different amount: go on from there
                        have = json.loads(e.read() or b'{}').get('have', 0)
                        off = have
                        chunk = None
                        break
                    print('  upload: HTTP %s at %d' % (e.code, off), flush=True)
                    time.sleep(5 * (attempt + 1))
                except Exception as e:
                    print('  upload: %s at %d' % (repr(e)[:150], off), flush=True)
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError('upload failed at %d' % off)
            if chunk is not None:
                off += len(chunk)
            print('  sent %s: %.0f %%' % (name, 100 * off / size), flush=True)


def train(args, run, world, item=None):
    """torchrun train_v2.py; every 30 s: speed to the report, logs and tests mirrored, the plateau rule."""
    out = os.path.join(HOME, 'runs', run)
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node', str(world), 'train_v2.py'] + args
    print(' '.join(cmd), flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=HOME)
    off = {'log.jsonl': 0, 'epochs.jsonl': 0}
    allrows, eps, sent_args = [], [], False
    reason = None
    last = 0
    while True:
        done = proc.poll() is not None
        if done or time.time() - last >= 30:
            last = time.time()
            new_log, off['log.jsonl'] = lines(os.path.join(out, 'log.jsonl'), off['log.jsonl'])
            new_eps, off['epochs.jsonl'] = lines(os.path.join(out, 'epochs.jsonl'), off['epochs.jsonl'])
            allrows += new_log
            eps += new_eps
            s = summary(allrows, world)
            if s:
                print('step %(step)s: %(s_per_step)s s/step, %(frames_per_s)s frames/s on all cards, data wait %(data_wait_s)s s, by kind %(kind_s)s' % s, flush=True)
            for e in new_eps:
                t = e.get('test') or {}
                print('EPOCH %s step %s: teacher test F1 %s, found %s, precision %s' % (e.get('epoch'), e.get('step'), t.get('f1'), t.get('recall'), t.get('precision')), flush=True)
            if item is not None:
                body = {'log': new_log, 'epochs': new_eps}
                try:
                    body['status'] = json.load(open(os.path.join(out, 'status.json')))
                except Exception:
                    pass
                if not sent_args and os.path.exists(os.path.join(out, 'args.json')):
                    body['args'] = json.load(open(os.path.join(out, 'args.json')))
                    sent_args = True
                try:
                    call('/api/jobs/kit/mirror', body, run=run)
                except Exception as e:
                    print('mirror failed:', repr(e)[:120], flush=True)
                if reason is None and not done:
                    reason = plateau(eps, item)
                    if reason:
                        print('STOP:', reason, flush=True)
                        open(os.path.join(out, 'STOP'), 'w').write(reason)
            report(phase='training' if not done else 'done', run=run, gpus=gpus(), world=world, elapsed_s=round(time.time() - t0),
                   returncode=proc.returncode, **s)
        if done:
            break
        time.sleep(5)
    return proc.returncode, reason, out


def main():
    g = gpus()
    world = max(1, len(g))
    report(phase='start', gpus=g, cpus=os.cpu_count(), archive=bool(ARCHIVE))
    try:
        import timm  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'timm'], check=True)
    if not ARCHIVE:                                   # the speed test
        os.makedirs(HOME, exist_ok=True)
        os.chdir(HOME)
        download_small()
        bb = [] if BACKBONE == 'vit' else ['--backbone', BACKBONE]
        if bb:
            subprocess.run([sys.executable, '-c', 'import timm; timm.create_model(%r, pretrained=True, features_only=True)' % BACKBONE], check=True)
        shutil.rmtree(os.path.join(HOME, 'runs', 'kit'), ignore_errors=True)
        args = ['--out', 'runs/kit', '--weights', '', '--steps', str(STEPS), '--amp', 'fp16', '--lr', '3e-4', '--freeze-steps', '0',
                '--drafts', '0.35', '--draft-only-steps', '0', '--draft-items', '3', '--far', '0.3', '--T', '4',
                '--epoch-steps', '100000', '--owner-exam-every', '100000'] + KAGGLE_WORKERS + bb
        code, _, _ = train(args, 'kit', world)
        print('\nDONE (code %s)' % code, flush=True)
        return
    if not PASSWORD:
        raise SystemExit('--password is needed for the archive')
    if not os.path.exists(os.path.join(HOME, 'train_v2.py')):
        get_archive(ARCHIVE, PASSWORD)
    os.chdir(HOME)
    for f in CODE:                                    # today's code, not the archive's
        with urllib.request.urlopen(url('/api/jobs/kit/src', f=f), timeout=60) as r, open(os.path.join(HOME, f), 'wb') as o:
            o.write(r.read())
    got = call('/api/jobs/kit/claim')
    if not got:
        print('the queue has no item for Kaggle now ("where": "kaggle" in data/v2_queue/queue.json)', flush=True)
        report(phase='idle')
        return
    item, args = got['item'], got['args']
    run = item['name']
    print('variant %s: %s' % (run, item.get('note', '')), flush=True)
    bb = args[args.index('--backbone') + 1] if '--backbone' in args else None
    if bb:
        subprocess.run([sys.executable, '-c', 'import timm; timm.create_model(%r, pretrained=True, features_only=True)' % bb], check=True)
    code, reason, out = train(args + ['--amp', 'fp16'] + KAGGLE_WORKERS, run, world, item)
    keep = os.path.join(KEEP, run)
    os.makedirs(keep, exist_ok=True)
    for f in ('best.pt', 'last.pt', 'log.jsonl', 'epochs.jsonl', 'args.json', 'status.json'):
        if os.path.exists(os.path.join(out, f)):
            shutil.copy2(os.path.join(out, f), keep)
    best = os.path.join(out, 'best.pt')
    if os.path.exists(best):
        print('sending best.pt to the shop', flush=True)
        try:
            upload(run, best)
        except Exception as e:
            print('upload failed (it is in %s):' % keep, repr(e)[:200], flush=True)
    state = 'stopped' if reason else ('done' if code == 0 else 'failed')
    call('/api/jobs/kit/done', {'state': state, 'reason': reason or ('all steps' if code == 0 else 'exit code %s' % code)}, run=run)
    print('\nDONE %s: %s; the run is in %s' % (run, reason or state, keep), flush=True)


if __name__ == '__main__':
    main()
