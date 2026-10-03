"""The shop machine as the server of SAM 3.1 work: it cuts 15-minute windows of both cameras into jobs of
SESSION frames, hands them to any worker that asks (itself, Google Colab, a friend's PC), takes the masks back,
and when a camera's window is complete, puts it together and joins the pieces of one person by ReID.

  prepare  sam31_segment.prepare: the video 2176x1224, the student's live ticks; the live stretches cut into
           sessions of SESSION frames overlapping by OVERLAP; each session -> jobs/<id>.json + inputs/<id>.mp4
           (its frames at 1008x1008, what SAM squeezes any frame to itself)
  serve    /api/jobs/claim, /<id>/input, /<id>/result, /<id>/fail, /status, /code (the worker's source),
           /weights -- with the workers' key (_workers_key.txt next to _labelers_key.txt), not the labellers'
  merge    all jobs of a camera done -> data/sam31_seg/<tag>/<cam>/chunks.npz + info.json at 2176x1224, as
           sam31_segment.py writes them, then sam31_reid.py
  hub      the loop: keeps jobs ready, merges finished cameras, keeps the machine's own worker running
  stills   single frames (the drafts, the owner's painted frames) as jobs of unrelated frames -> data/sam31_stills

usage: sam31_jobs.py hub [DAY ...] | stills      (data/sam31_jobs/)"""
import json
import os
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
JOBS = ROOT / 'data' / 'sam31_jobs'
SEG = ROOT / 'data' / 'sam31_seg'
KEY_FILE = ROOT.parent.parent / '_workers_key.txt'
SESSION = int(os.environ.get('RA_S31_SESSION', '160'))
OVERLAP = 8
SPARE_S = 8 * 60                     # a claimed job not back within this is given to another worker too
KEEP_READY = 20                     # prepare the next window when fewer open jobs than this
S3 = r'C:\Users\ArykovAA\cctv_ai\venv_sam3\Scripts\python.exe'
RF = r'C:\Users\ArykovAA\cctv_ai\venv_rfdetr\Scripts\python.exe'
EXAM = '20260918'                   # the exam day: never labelled for training
DAYS = ['20260922', '20260917', '20260920', '20260921', '20260923', '20260919']
_lock = threading.Lock()


def key():
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_urlsafe(16))
    return KEY_FILE.read_text().strip()


def jobs():
    (JOBS / 'jobs').mkdir(parents=True, exist_ok=True)
    return sorted((JOBS / 'jobs').glob('*.json'))


def read(p):
    return json.load(open(p))


def write(p, d):
    tmp = str(p) + '.tmp'
    json.dump(d, open(tmp, 'w'), indent=1)
    os.replace(tmp, p)


# ---------------------------------------------------------------- preparing

def sessions(live):
    out, i, n = [], 0, len(live)
    while i < n:
        if not live[i]:
            i += 1; continue
        j = i
        while j < n and live[j]:
            j += 1
        s, shared = i, 0
        while s < j:
            e = min(j, s + SESSION)
            out.append((s, e, shared))
            if e >= j:
                break
            s, shared = e - OVERLAP, OVERLAP
        i = j
    return out


def prepare(day, t0, seconds=900.0):
    import sam31_segment as SG
    import day_proxy
    tag = '%s_%05d' % (day, int(t0))
    n = int(round(seconds / SG.TICK))
    for cam in ('cam1', 'cam2'):
        out = SEG / tag / cam
        if (out / 'info.json').exists() or any(read(p).get('tag') == tag and read(p).get('cam') == cam for p in jobs()):
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / 'preparing').write_text(time.strftime('%H:%M'))
        live = SG.prepare(day, cam, t0, n, out)
        (JOBS / 'inputs').mkdir(parents=True, exist_ok=True)
        for s, e, shared in sessions(live):
            jid = '%s_%s_%05d' % (tag, cam, s)
            mp4 = JOBS / 'inputs' / (jid + '.mp4')
            subprocess.run([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-framerate', '12.5', '-start_number', str(s),
                            '-i', str(out / 'sam_in' / '%05d.jpg'), '-frames:v', str(e - s), '-c:v', 'libx264', '-preset', 'fast',
                            '-crf', '14', '-pix_fmt', 'yuv420p', str(mp4)], check=True)
            write(JOBS / 'jobs' / (jid + '.json'), {'id': jid, 'tag': tag, 'day': day, 'film_start': t0, 'seconds': seconds,
                                                     'cam': cam, 'start': s, 'stop': e, 'shared': shared, 'status': 'open',
                                                     'made': time.time()})
        shutil.rmtree(out / 'sam_in', ignore_errors=True)
        (out / 'preparing').unlink()
        print(time.strftime('%H:%M'), 'prepared', tag, cam, 'live', int(live.sum()), flush=True)


def window(day, rank=0):
    import sam31_night
    return sam31_night.window(day, rank)


# ---------------------------------------------------------------- merging

def merge(tag, cam):
    """All jobs of a camera back -> chunks.npz + info.json in the segment format, then ReID."""
    import cv2
    import sam31_segment as SG
    js = sorted((read(p) for p in jobs() if read(p).get('tag') == tag and read(p).get('cam') == cam), key=lambda j: j['start'])
    if js and js[0].get('stills'):
        return merge_stills(js[0])
    W, H = SG.W, SG.H
    sx, sy = W / 1008.0, H / 1008.0
    rows, buf, offs = [], [], [0]
    for j in js:
        with np.load(JOBS / 'results' / (j['id'] + '.npz')) as z:     # closed: the files are moved to done/ after
            R, B, O = z['rows'], z['buf'], z['offs']
        for r in range(len(R)):
            kl, i, p, x1, y1, x2, y2 = R[r]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            m = np.unpackbits(B[O[r]:O[r + 1]])[:(y2 - y1) * (x2 - x1)].reshape(y2 - y1, x2 - x1)
            X1, Y1 = int(round(x1 * sx)), int(round(y1 * sy))
            X2, Y2 = max(X1 + 1, int(round(x2 * sx))), max(Y1 + 1, int(round(y2 * sy)))
            M = cv2.resize(m, (X2 - X1, Y2 - Y1), interpolation=cv2.INTER_NEAREST).astype(bool)
            bits = np.packbits(M)
            rows.append((j['start'], j['start'] + kl, i, p, X1, Y1, X2, Y2)); buf.append(bits); offs.append(offs[-1] + len(bits))
    out = SEG / tag / cam
    np.savez_compressed(out / 'chunks.npz', rows=np.array(rows, dtype=np.float64).reshape(-1, 8),
                        buf=np.concatenate(buf) if buf else np.zeros(0, np.uint8), offs=np.array(offs, dtype=np.int64))
    live = json.load(open(out / 'ticks.json'))['live']
    info = {'day': js[0]['day'], 'cam': cam, 'film_start': js[0]['film_start'], 'seconds': js[0]['seconds'], 'ticks': len(live),
            'tick': SG.TICK, 'size': [W, H], 'sam_input': 1008, 'live_ticks': int(sum(live)),
            'sessions': [[j['start'], j['stop'], j['shared']] for j in js], 'planned_sessions': len(js), 'partial': False,
            'overlap': OVERLAP, 'workers': sorted({j.get('worker', '?') for j in js}),
            'sam_s': round(sum(j.get('sam_s', 0) for j in js)),
            's_per_live_tick': round(sum(j.get('sam_s', 0) for j in js) / max(1, sum(j['stop'] - j['start'] for j in js)), 3)}
    json.dump(info, open(out / 'info.json', 'w'), indent=1)
    r = subprocess.run([RF, 'sam31_reid.py', '%s/%s' % (tag, cam), '0.35'], cwd=str(ROOT), capture_output=True, text=True,
                       env=dict(os.environ, RA_S31_ROOT=str(SEG), RA_S31_VIDEO='0'))
    for j in js:                                        # done: the inputs go, the job files move to done/
        (JOBS / 'inputs' / (j['id'] + '.mp4')).unlink(missing_ok=True)
        (JOBS / 'done').mkdir(exist_ok=True)
        os.replace(JOBS / 'jobs' / (j['id'] + '.json'), JOBS / 'done' / (j['id'] + '.json'))
    print(time.strftime('%H:%M'), 'merged', tag, cam, len(rows), 'masks; reid', r.returncode, r.stdout[-300:], r.stderr[-500:] if r.returncode else '', flush=True)


# ---------------------------------------------------------------- single frames (the drafts, the owner's frames)

STILLS_PER_JOB = 64


def prepare_stills():
    """Every painted frame and every draft frame not labelled yet -> jobs of STILLS_PER_JOB unrelated frames
    (all key frames; the order of sam31_stills.jobs: the owner's frames first). Each job is its own 'camera',
    so it is written out the moment it comes back."""
    import cv2
    import day_proxy
    import sam31_stills as ST
    have = {read(p)['id'] for p in jobs()}
    todo = [(kind, ident, path) for kind, ident, path in ST.jobs() if not (ST.OUT / kind / ('%s.png' % ident)).exists()]
    if any(i.startswith('stills_') for i in have):
        return 0                                   # a batch is already out: finish it first
    (JOBS / 'inputs').mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%m%d%H%M')
    made = 0
    for b in range(0, len(todo), STILLS_PER_JOB):
        part = todo[b:b + STILLS_PER_JOB]
        jid = 'stills_%s_%04d' % (stamp, b // STILLS_PER_JOB)
        tmp = JOBS / 'inputs' / (jid + '_frames')
        tmp.mkdir(exist_ok=True)
        items = []
        for kind, ident, path in part:
            img = cv2.imread(str(path))
            if img is None:
                continue
            cv2.imwrite(str(tmp / ('%05d.jpg' % len(items))), cv2.resize(img, (1008, 1008), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 97])
            items.append([kind, ident])
        subprocess.run([day_proxy._ffmpeg(), '-y', '-loglevel', 'error', '-framerate', '12.5', '-i', str(tmp / '%05d.jpg'),
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '12', '-g', '1', '-pix_fmt', 'yuv420p',
                        str(JOBS / 'inputs' / (jid + '.mp4'))], check=True)
        shutil.rmtree(tmp, ignore_errors=True)
        write(JOBS / 'jobs' / (jid + '.json'), {'id': jid, 'tag': 'stills', 'cam': jid, 'stills': True, 'items': items, 'start': 0,
                                                 'stop': len(items), 'shared': 0, 'status': 'open', 'made': time.time()})
        made += 1
    print(time.strftime('%H:%M'), 'stills:', len(todo), 'frames in', made, 'jobs', flush=True)
    return made


def merge_stills(j):
    import cv2
    import sam31_stills as ST
    with np.load(JOBS / 'results' / (j['id'] + '.npz')) as z:
        R, B, O = z['rows'], z['buf'], z['offs']
    per = {}
    for r in range(len(R)):
        kl, i, p, x1, y1, x2, y2 = R[r]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        m = np.zeros((1008, 1008), np.uint8)
        m[y1:y2, x1:x2] = np.unpackbits(B[O[r]:O[r + 1]])[:(y2 - y1) * (x2 - x1)].reshape(y2 - y1, x2 - x1)
        per.setdefault(int(kl), []).append((float(p), cv2.resize(m, (ST.W, ST.H), interpolation=cv2.INTER_NEAREST).astype(bool)))
    for k, (kind, ident) in enumerate(j['items']):
        (ST.OUT / kind).mkdir(parents=True, exist_ok=True)
        ST.write(kind, ident, [x for x in per.get(k, []) if x[1].any()])
    (JOBS / 'inputs' / (j['id'] + '.mp4')).unlink(missing_ok=True)
    (JOBS / 'done').mkdir(exist_ok=True)
    os.replace(JOBS / 'jobs' / (j['id'] + '.json'), JOBS / 'done' / (j['id'] + '.json'))
    print(time.strftime('%H:%M'), 'stills', j['id'], len(j['items']), 'frames by', j.get('worker'), '%.2f s/frame' % (j.get('sam_s', 0) / max(1, len(j['items']))), flush=True)


def stills_hub():
    """Cuts the single frames into jobs, writes each one out as it returns; after 21:02 the machine's own card
    joins in; when nothing is left: the scores against the owner's frames (sam31_stills.report) and out."""
    import bg
    import sam31_stills as ST
    prepare_stills()
    while True:
        for tag, cam in finished_cameras():
            if tag == 'stills':
                try:
                    merge(tag, cam)
                except Exception as e:
                    print(time.strftime('%H:%M'), 'merge failed', cam, repr(e)[:300], flush=True)
        left = [p for p in jobs() if read(p).get('stills')]
        if not left:
            break
        if ST.night():
            own_worker()
        time.sleep(30)
    for line in bg.running('sam31_worker.py'):          # the machine's own worker: its card back before the morning
        if 'shop-3060' in line:
            subprocess.run(['taskkill', '/PID', line.split('|')[0], '/F', '/T'], capture_output=True)
    print(time.strftime('%H:%M'), 'stills done', json.dumps(ST.report()), flush=True)


def pick(now, stills=True):
    """stills: the worker said it knows single-frame jobs (sam31_worker.py of 30.09). The next job for a worker: windows are finished one at a time, oldest first. Within the oldest
    unfinished window: an open job, else one whose worker is silent for SPARE_S (given again: the first
    result back wins -- a vanished Colab no longer holds a whole window). Only then the next window."""
    by = {}
    for p in jobs():
        j = read(p)
        if j.get('stills') and not stills:        # a worker with the old code would track through unrelated frames
            continue
        by.setdefault(j['tag'], []).append((p, j))
    order = sorted(by, key=lambda t: min(j.get('made', 0) for _, j in by[t]))
    for tag in order:
        js = sorted(by[tag], key=lambda pj: (pj[1]['cam'], pj[1]['start']))
        for p, j in js:
            if j['status'] == 'open':
                return p, j
        late = [(p, j) for p, j in js if j['status'] == 'claimed' and now - j.get('claimed_at', 0) > SPARE_S]
        if late:
            return min(late, key=lambda pj: pj[1].get('claimed_at', 0))
    return None, None


def finished_cameras():
    by = {}
    for p in jobs():
        j = read(p)
        by.setdefault((j['tag'], j['cam']), []).append(j['status'])
    return [k for k, v in by.items() if all(s == 'done' for s in v)      # a camera still being cut into jobs by the
            and not (SEG / k[0] / k[1] / 'preparing').exists()]          # prep process must not be merged half-way


# ---------------------------------------------------------------- serving

def register(app, root):
    from flask import jsonify, request, send_file, abort, Response

    def ok():
        k = request.args.get('key') or request.headers.get('X-Key', '')
        if not secrets.compare_digest(k, key()):
            abort(401)
        return request.args.get('worker', '?')[:40]

    @app.get('/api/jobs/claim')
    def jobs_claim():
        who = ok()
        now = time.time()
        with _lock:
            p, j = pick(now, stills=request.args.get('stills') == '1')
            if j is not None:
                j.update(status='claimed', worker=who, claimed_at=now, claims=j.get('claims', 0) + 1)
                write(p, j)
                return jsonify({'id': j['id'], 'frames': j['stop'] - j['start'], 'stills': bool(j.get('stills'))})
        return jsonify({})

    @app.get('/api/jobs/<jid>/input')
    def jobs_input(jid):
        ok()
        p = JOBS / 'inputs' / (os.path.basename(jid) + '.mp4')
        if not p.exists():
            abort(404)
        return send_file(str(p), mimetype='video/mp4')

    @app.post('/api/jobs/<jid>/result')
    def jobs_result(jid):
        who = ok()
        jid = os.path.basename(jid)
        p = JOBS / 'jobs' / (jid + '.json')
        if not p.exists():
            abort(404)
        if read(p)['status'] == 'done':               # a spare copy came back second: the first one stands
            return jsonify({'ok': True, 'duplicate': True})
        (JOBS / 'results').mkdir(exist_ok=True)
        data = request.get_data()
        tmp = JOBS / 'results' / (jid + '.npz.tmp')
        tmp.write_bytes(data)
        with np.load(tmp) as z:                           # a broken upload is refused, not stored;
            z['rows'], z['offs']                          # closed before the rename: Windows won't move an open file
        os.replace(tmp, JOBS / 'results' / (jid + '.npz'))
        with _lock:
            j = read(p)
            j.update(status='done', worker=who, done_at=time.time(), frames=int(request.args.get('frames', 0)),
                     sam_s=float(request.args.get('sam_s', 0)))
            write(p, j)
        return jsonify({'ok': True})

    @app.post('/api/jobs/<jid>/fail')
    def jobs_fail(jid):
        who = ok()
        p = JOBS / 'jobs' / (os.path.basename(jid) + '.json')
        if p.exists():
            with _lock:
                j = read(p)
                if j['status'] != 'done':
                    j.update(status='open', fails=j.get('fails', 0) + 1,
                             last_error=('%s: %s' % (who, request.get_data()[:1500].decode('utf-8', 'replace'))))
                    write(p, j)
        return jsonify({'ok': True})

    # the training kit for a borrowed machine (kit_build.py makes it, kit_run.py fetches it and reports the speed)
    kit = ROOT / 'data' / 'kit'

    @app.get('/api/jobs/kit')
    def jobs_kit():
        ok()
        return Response((ROOT / 'kit_run.py').read_text(encoding='utf-8'), mimetype='text/plain; charset=utf-8')

    @app.get('/api/jobs/kit/manifest')
    def jobs_kit_manifest():
        ok()
        return send_file(str(kit / 'manifest.json'), mimetype='application/json')

    @app.get('/api/jobs/kit/file')
    def jobs_kit_file():
        ok()
        p = request.args.get('p', '')
        if p not in {i['src'] for i in json.load(open(kit / 'manifest.json'))['files']}:     # only what the kit lists
            abort(404)
        return send_file(str(ROOT / p), conditional=True)                                  # Range: a broken download resumes

    @app.post('/api/jobs/kit/report')
    def jobs_kit_report():
        who = ok()
        body = request.get_json(force=True, silent=True) or {}
        with open(ROOT / 'data' / 'logs' / 'kit_report.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(body, got=time.strftime('%Y-%m-%d %H:%M:%S'), worker=who)) + '\n')
        return jsonify({'ok': True})

    # the variant queue's items for a borrowed machine ("where": "kaggle" in data/v2_queue/queue.json): claimed by
    # kit_run.py --queue, its logs and tests mirrored into runs/<name>/ (the /train page and the queue read them),
    # its best checkpoint sent back in parts (a tunnel request is at most 100 MB)
    qdir = ROOT / 'data' / 'v2_queue'

    def remote_state():
        p = qdir / 'remote.json'
        return json.load(open(p, encoding='utf-8')) if p.exists() else {}

    def remote_save(d):
        tmp = qdir / 'remote.tmp.json'
        json.dump(d, open(tmp, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
        os.replace(tmp, qdir / 'remote.json')

    def run_dir():
        run = os.path.basename(request.args.get('run', ''))
        if not run.startswith(('v2_', 'k_')):
            abort(400)
        d = ROOT / 'runs' / run
        d.mkdir(parents=True, exist_ok=True)
        return run, d

    @app.get('/api/jobs/kit/claim')
    def jobs_kit_claim():
        who = ok()
        import variant_queue as VQ
        with _lock:
            queue = json.load(open(qdir / 'queue.json', encoding='utf-8'))
            rem = remote_state()
            item = next((i for i in queue if i.get('where') == 'kaggle' and i['name'] not in rem), None)
            if item is None:
                return jsonify({})
            st = json.load(open(qdir / 'state.json', encoding='utf-8')) if (qdir / 'state.json').exists() else {'items': []}
            cmd = VQ.command(dict(item), st)
            rem[item['name']] = {'state': 'running', 'worker': who, 'started': time.strftime('%H:%M')}
            remote_save(rem)
        return jsonify({'item': item, 'args': cmd[1:]})

    @app.get('/api/jobs/kit/src')
    def jobs_kit_src():
        ok()
        f = os.path.basename(request.args.get('f', ''))
        if not f.endswith('.py') or not (ROOT / f).exists():
            abort(404)
        return Response((ROOT / f).read_text(encoding='utf-8'), mimetype='text/plain; charset=utf-8')

    @app.post('/api/jobs/kit/mirror')
    def jobs_kit_mirror():
        who = ok()
        run, d = run_dir()
        body = request.get_json(force=True, silent=True) or {}
        for key, name in (('log', 'log.jsonl'), ('epochs', 'epochs.jsonl')):
            if body.get(key):
                with open(d / name, 'a', encoding='utf-8') as f:
                    for r in body[key]:
                        f.write(json.dumps(r) + '\n')
        for key in ('status', 'args'):
            if body.get(key):
                json.dump(body[key], open(d / (key + '.json'), 'w', encoding='utf-8'), indent=1)
        json.dump({'worker': who, 't': time.time()}, open(d / 'remote.json', 'w'))
        return jsonify({'ok': True})

    @app.post('/api/jobs/kit/upload')
    def jobs_kit_upload():
        ok()
        run, d = run_dir()
        name = request.args.get('file', '')
        if name not in ('best.pt', 'last.pt'):
            abort(400)
        part = d / (name + '.part')
        off = int(request.args.get('offset', 0))
        if off == 0 and part.exists():
            part.unlink()
        have = part.stat().st_size if part.exists() else 0
        if have != off:
            return jsonify({'ok': False, 'have': have}), 409
        with open(part, 'ab') as f:
            shutil.copyfileobj(request.stream, f, 1 << 20)
        have = part.stat().st_size
        if request.args.get('final') == '1':
            os.replace(part, d / name)
        return jsonify({'ok': True, 'have': have})

    @app.post('/api/jobs/kit/done')
    def jobs_kit_done():
        who = ok()
        run, d = run_dir()
        body = request.get_json(force=True, silent=True) or {}
        with _lock:
            rem = remote_state()
            r = rem.get(run, {'worker': who})
            r.update(state=body.get('state', 'done'), reason=body.get('reason', ''), ended=time.strftime('%H:%M'))
            rem[run] = r
            remote_save(rem)
        return jsonify({'ok': True})

    @app.get('/api/jobs/code')
    def jobs_code():
        ok()
        return Response((ROOT / 'sam31_worker.py').read_text(encoding='utf-8'), mimetype='text/plain; charset=utf-8')

    @app.get('/api/jobs/weights')
    def jobs_weights():
        ok()
        return send_file(str(ROOT / 'data' / 'weights' / 'sam3' / 'sam3.1_multiplex.pt'), as_attachment=True)

    @app.get('/api/jobs/status')
    def jobs_status():
        ok()
        return jsonify(status())

    @app.get('/sam31')                                  # the dashboard: labellers' key, like every page
    def sam31_page():
        from flask import render_template
        return render_template('sam31.html')

    @app.get('/api/sam31/status')
    def sam31_status():
        return jsonify(status(detail=True))


def status(detail=False):
    js = [read(p) for p in jobs()] + [read(p) for p in sorted((JOBS / 'done').glob('*.json'))] if (JOBS / 'done').exists() else [read(p) for p in jobs()]
    by_worker, by_cam = {}, {}
    for j in js:
        w = by_worker.setdefault(j.get('worker', '-'), {'done': 0, 'frames': 0, 'sam_s': 0.0})
        if j['status'] == 'done':
            w['done'] += 1; w['frames'] += j.get('frames', 0); w['sam_s'] += j.get('sam_s', 0)
        c = by_cam.setdefault('%s/%s' % (j['tag'], j['cam']), {'open': 0, 'claimed': 0, 'done': 0})
        c[j['status']] += 1
    for w in by_worker.values():
        w['s_per_frame'] = round(w['sam_s'] / max(1, w['frames']), 2)
    out = {'jobs': {s: sum(1 for j in js if j['status'] == s) for s in ('open', 'claimed', 'done')},
           'workers': by_worker, 'cameras': by_cam}
    if detail:                                          # times for the dashboard: speed, ETA, who holds what
        now = time.time()
        out['now'] = now
        out['done_at'] = sorted(j['done_at'] for j in js if j['status'] == 'done' and j.get('done_at'))
        out['claimed'] = [{'id': j['id'], 'worker': j.get('worker', '?'), 'cam': '%s/%s' % (j['tag'], j['cam']),
                           'age': now - j.get('claimed_at', now), 'frames': j['stop'] - j['start']}
                          for j in js if j['status'] == 'claimed']
        for j in js:
            w = by_worker.get(j.get('worker', '-'))
            if w is not None and j.get('done_at'):
                w['last'] = max(w.get('last', 0), j['done_at'])
            c = by_cam['%s/%s' % (j['tag'], j['cam'])]
            c['frames'] = c.get('frames', 0) + j['stop'] - j['start']
            c['fails'] = c.get('fails', 0) + j.get('fails', 0)
            if j['status'] == 'done':
                c['frames_done'] = c.get('frames_done', 0) + j['stop'] - j['start']
                c['first'] = min(c.get('first', now), j.get('claimed_at', now))
                c['last'] = max(c.get('last', 0), j.get('done_at', 0))
        out['merged'] = sorted('%s/%s' % (p.parent.name, p.name) for p in SEG.glob('*/cam*') if (p / 'chunks.npz').exists())
    return out


# ---------------------------------------------------------------- the hub

def own_worker():
    import bg
    if os.environ.get('RA_S31_LOCAL_WORKER') == '0' or bg.running('sam31_worker.py'):   # 0: the card is for
        return                                                                           # preparing and merging
    bg._python = lambda: S3
    bg.spawn('sam31worker_local', ['sam31_worker.py', 'http://127.0.0.1:5070', key(), 'shop-3060'],
             env={'SAM31_CKPT': str(ROOT / 'data' / 'weights' / 'sam3' / 'sam3.1_multiplex.pt')})


def hub(days, role='all'):
    """role 'all' -- merge, prepare and keep the local worker; 'merge' -- no preparing; 'prep' -- only preparing,
    so the next windows are ready while the workers are still on this one (RA_S31_READY open jobs ahead)."""
    queue = list(days)
    ready = int(os.environ.get('RA_S31_READY', 150 if role == 'prep' else KEEP_READY))
    queue = [d for d in queue if not d.startswith(EXAM)]   # 'DAY' or 'DAY:RANK' (RANK-th busiest file of the day)
    while True:
        if role != 'prep':
            for tag, cam in finished_cameras():
                try:
                    merge(tag, cam)
                except Exception as e:
                    print(time.strftime('%H:%M'), 'merge failed', tag, cam, repr(e)[:300], flush=True)
        st = status()['jobs']
        if role != 'merge' and st['open'] < ready and queue:
            day, _, rank = queue.pop(0).partition(':')
            t0, name = window(day, int(rank or 0))
            if t0 is not None and role == 'prep' and (SEG / ('%s_%05d' % (day, int(t0)))).exists():
                continue                                      # this window is already there
            if t0 is not None:
                try:
                    prepare(day, t0)
                except Exception as e:
                    print(time.strftime('%H:%M'), 'prepare failed', day, repr(e)[:300], flush=True)
        if role != 'prep':
            own_worker()
        elif not queue:
            print(time.strftime('%H:%M'), 'nothing left to prepare', flush=True)
            return
        time.sleep(60)


if __name__ == '__main__':
    if sys.argv[1:2] in (['hub'], ['prep'], ['merge']):
        hub(sys.argv[2:] or DAYS, {'hub': 'all'}.get(sys.argv[1], sys.argv[1]))
    elif sys.argv[1:2] == ['stills']:
        stills_hub()
    elif sys.argv[1:2] == ['status']:
        print(json.dumps(status(), indent=1))
