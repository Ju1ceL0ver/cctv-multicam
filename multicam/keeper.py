"""Everything this box does, kept running without anyone watching it.

The day has two modes and they must never fight over the video card: from 10:00 the
live counter serves the shop, and from 21:00 until 09:45 the teachers label whole
recorded days for review. Around both, three things must simply always be true --
both cameras are being recorded, the review site is reachable, and nothing that died
stays dead.

Started by the Windows task `cctv-keeper` at logon and re-checked every 15 minutes, so
a reboot at closing time (which happened on 17.09 and cost a night of recording) no
longer needs a person."""
import os, re, sys, glob, time, json, shutil, subprocess
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.abspath(__file__)); os.chdir(ROOT); sys.path.insert(0, ROOT)
from bg import spawn, running
from rawsource import RAW
from auto_label import windows
from clip_registry import clip_key, clip_ready, load_failures, set_aside
from storage import atomic_json, file_lock

HOME = os.path.dirname(os.path.dirname(ROOT))   # the box's own home: as SYSTEM, expanduser points at system32
PY = sys.executable
CONDA = os.path.join(os.path.dirname(PY), 'python.exe')
LOGS = os.path.join(ROOT, 'data', 'logs')
LOG = os.path.join(LOGS, 'keeper.log')
LOCK = os.path.join(LOGS, 'keeper.heartbeat')
URLS = os.path.join(LOGS, 'urls.json')
CLOUDFLARED = r'C:\Program Files (x86)\cloudflared\cloudflared.exe'

NIGHT_START = os.environ.get('RA_NIGHT_START', '21:00')
NIGHT_STOP = os.environ.get('RA_NIGHT_STOP', '09:45')
MINUTES = int(os.environ.get('RA_WINDOW', '10'))
WORKERS = [('_am', '00:00-14:30'), ('_pm', '14:30-23:59')]   # two passes fill the card better than one
TRAIN_FROM = os.environ.get('RA_TRAIN_FROM', '05:00')   # the night's last hours go to the student
MARGIN = int(os.environ.get('RA_MARGIN', '45'))          # auto_label will not start a window it cannot finish before that
RESPAWN_GAP = 300                                        # seconds; a pass that dies at once is not restarted every minute
KEEP_DAYS = int(os.environ.get('RA_KEEP_DAYS', '14'))        # raw footage older than this may go
FREE_FLOOR_GB = float(os.environ.get('RA_FREE_GB', '150'))


def log(*a):
    line = '%s %s' % (datetime.now().strftime('%m-%d %H:%M:%S'), ' '.join(str(x) for x in a))
    os.makedirs(LOGS, exist_ok=True)   # stdout already lands in this same file
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def beat():
    open(LOCK, 'w').write('%d %s' % (os.getpid(), datetime.now().isoformat(timespec='seconds')))


# ---------------------------------------------------------------- the services


def ensure(name, pattern, start):
    if running(pattern):
        return False
    start()
    log('%s was not running, started it' % name)
    return True


def pid_alive(pid):
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          'Get-Process -Id %d -ErrorAction SilentlyContinue | %%{ $_.Id }' % pid],
                         capture_output=True, text=True).stdout
    return str(pid) in out


def ensure_recorders():
    """Both cameras, endlessly. The two processes share a script name and differ only by
    an environment variable, so each one's pid file is what says whether it is alive."""
    for cam in ('cam1', 'cam2'):
        pf = os.path.join(HOME, '_raw_recorder_%s.pid' % cam)
        try:
            pid = int(open(pf).read().strip())
        except Exception:
            pid = 0
        if pid and pid_alive(pid):
            continue
        spawn('raw_%s' % cam, [os.path.join(HOME, '_raw_recorder.py')],
              env={'RA_RAW_CAMERA': cam, 'RA_RAW_DAYS': '3650', 'RA_RAW_MANAGED': '1',
                   'RA_RAW_FREE_GB': str(FREE_FLOOR_GB)}, cwd=HOME)
        log('recorder %s was not running, started it' % cam)
        time.sleep(3)


def start_live():
    spawn('run_live', [os.path.join(HOME, 'cctv_ai', 'retail_analytics', 'scripts', 'run_live.py'),
                       '--crm', '--hours', '10-21'],
          cwd=os.path.join(HOME, 'cctv_ai', 'retail_analytics'))


def ensure_jupyter():
    """The way this box is reached from outside. Nothing else starts it, so after a
    reboot it has to come back on its own -- together with its tunnel."""
    if running('jupyter-lab') or running('jupyter.exe'):
        return
    tf = os.path.join(HOME, '_jupyter_token.txt')
    token = open(tf).read().strip() if os.path.exists(tf) else ''
    args = [os.path.join(os.path.dirname(PY), 'Scripts', 'jupyter-lab.exe'),
            '--ip=127.0.0.1', '--port=8888', '--no-browser',
            '--ServerApp.allow_remote_access=True', '--ServerApp.trust_xheaders=True']
    if token:
        args.append('--IdentityProvider.token=%s' % token)
    subprocess.Popen(['powershell', '-NoProfile', '-Command',
                      "Start-Process -FilePath '%s' -ArgumentList %s -WorkingDirectory '%s' "
                      "-RedirectStandardOutput '%s' -RedirectStandardError '%s' -WindowStyle Hidden"
                      % (args[0], ','.join("'%s'" % a for a in args[1:]), HOME,
                         os.path.join(LOGS, 'jupyter.log'), os.path.join(LOGS, 'jupyter.log.err'))],
                     close_fds=True, stdin=subprocess.DEVNULL,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log('jupyter was not running, started it')
    time.sleep(10)


def start_labeler():
    spawn('labeler', ['label_pieces.py'])


def tunnel_up(port, name):
    """A quick-tunnel address changes every restart, so whatever it is now is written
    down where the review site can show it -- that is how the owner gets the new link."""
    out = subprocess.run(['powershell', '-NoProfile', '-Command',
                          "Get-CimInstance Win32_Process -Filter \"name='cloudflared.exe'\" | "
                          "%{ $_.CommandLine }"], capture_output=True, text=True).stdout
    if (':%d' % port) in out:
        return False
    logf = os.path.join(LOGS, 'cf_%s.log' % name)
    subprocess.Popen('"%s" tunnel --no-autoupdate --url http://127.0.0.1:%d > "%s" 2>&1'
                     % (CLOUDFLARED, port, logf), shell=True, cwd=ROOT)
    log('tunnel for %s was down, started it' % name)
    time.sleep(25)
    return True


def answers(url, port):
    """An address is only worth writing down if it actually reaches this box."""
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(url, timeout=12)
        return True
    except urllib.error.HTTPError:
        return True                    # 401 from the review site still proves it is there
    except Exception:
        return False


def read_urls():
    found = {}
    for name, port, extra in (('labeler', 5070, ('cf_label.log', '_cf_pairs.log', '_cf_annotate.log')),
                              ('jupyter', 8888, ('cf_jupyter.log',))):
        cands, seen = [], set()
        for c in [os.path.join(LOGS, 'cf_%s.log' % name)] + [os.path.join(base, e) for base in (LOGS, HOME) for e in extra]:
            if not os.path.exists(c):
                continue
            for u in re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',
                                open(c, encoding='utf-8', errors='replace').read()):
                if u not in seen:
                    seen.add(u); cands.append((os.path.getmtime(c), u))
        for _, u in sorted(cands, reverse=True):
            if answers(u, port):
                found[name] = u
                break
    if found:
        old = json.load(open(URLS)) if os.path.exists(URLS) else {}
        updated = {**old, **found}
        if updated != old:
            atomic_json(URLS, updated)
            log('addresses now:', json.dumps(updated))


# ---------------------------------------------------------------- night work


def recorded_days():
    days = set()
    for cam in ('cam1', 'cam2'):
        d = os.path.join(RAW, cam)
        if os.path.isdir(d):
            days |= {x for x in os.listdir(d) if x.isdigit() and len(x) == 8}
    return sorted(days)


def coverage(day, rng=None):
    """(windows ready, windows of the day or of one worker's part of it, windows set aside).
    A set-aside window is one that failed again and again: it must not hold the queue."""
    try:
        todo = windows(day, MINUTES)
    except Exception:
        return 0, 0, 0
    if rng:
        a, b = [datetime.strptime(x, '%H:%M').time() for x in rng.split('-')]
        todo = [t for t in todo if a <= t.time() < b]
    clips = os.path.join(ROOT, 'data', 'raw_clips')
    failures = load_failures(ROOT)
    done = held = 0
    for t in todo:
        key = clip_key(day, t, clips)
        if clip_ready(os.path.join(clips, key)):
            done += 1
        elif set_aside(failures.get(key)):
            held += 1
    return done, len(todo), held


def still_recording(day):
    last = 0.0
    for cam in ('cam1', 'cam2'):
        for p in glob.glob(os.path.join(RAW, cam, day, '*.mp4')):
            last = max(last, os.path.getmtime(p))
    return last and (time.time() - last) < 3600


def next_day(rng=None):
    """Oldest day that still has windows to label in this worker's part of it."""
    for day in recorded_days():
        if still_recording(day):
            continue
        done, total, held = coverage(day, rng)
        if total and done + held < total:
            return day, done, total, held
    return None, 0, 0, 0


def pass_alive(tag):
    """Fresh heartbeat AND a live process: a pass that finished used to keep its slot for
    45 minutes, so the next day started an hour late and the card sat idle."""
    hb = os.path.join(LOGS, 'autolabel%s.heartbeat' % tag)
    if not os.path.exists(hb) or time.time() - os.path.getmtime(hb) >= 2700:
        return False
    try:
        pid = int(open(hb).read().split()[0])
    except Exception:
        return True                    # cannot tell whose it is: keep the patient rule
    return pid_alive(pid)


def window_fits(now):
    """auto_label refuses to start a window it cannot finish before the student's hours;
    do not launch a pass only to hear that."""
    end = datetime.strptime(TRAIN_FROM, '%H:%M').time()
    stop = datetime.combine(now.date(), end)
    if stop <= now:
        stop += timedelta(days=1)
    return now + timedelta(minutes=MARGIN) < stop


def clear_stale(tag):
    """A pass that stopped reporting is killed with its children before a new one starts:
    two workers grinding the same window would waste the night and race on its files."""
    hb = os.path.join(LOGS, 'autolabel%s.heartbeat' % tag)
    if not os.path.exists(hb):
        return
    try:
        pid = int(open(hb).read().split()[0])
    except Exception:
        pid = 0
    if pid and pid_alive(pid):
        os.system('taskkill /PID %d /F /T > nul 2>&1' % pid)
        log('pass%s stopped reporting, killed it (pid %d) before starting a new one' % (tag, pid))
    os.remove(hb)


def training_hours(now):
    """The tail of the night: teachers stop labelling and the student learns from what
    the owner reviewed during the day."""
    a = datetime.strptime(TRAIN_FROM, '%H:%M').time()
    b = datetime.strptime(NIGHT_STOP, '%H:%M').time()
    return a <= now.time() < b


def in_night(now):
    a = datetime.strptime(NIGHT_START, '%H:%M').time()
    b = datetime.strptime(NIGHT_STOP, '%H:%M').time()
    return now.time() >= a or now.time() < b


def stop_teachers():
    stopped = []
    for pat in ('auto_label.py', 'detect_raw.py'):   # only what holds the card; cutting pictures is free
        for l in running(pat):
            os.system('taskkill /PID %s /F /T > nul 2>&1' % l.split('|')[0].strip())
            stopped.append(pat)
    if stopped:
        log('shop hours: teachers stopped, the card is the counter\'s')
    for f in glob.glob(os.path.join(LOGS, 'autolabel*.heartbeat')):
        os.remove(f)


# ---------------------------------------------------------------- disk


def prune_raw():
    """Only remove fully labelled days older than KEEP_DAYS when disk space is low."""
    free = shutil.disk_usage(ROOT).free / 2**30
    if free > FREE_FLOOR_GB:
        return
    cutoff = (datetime.now() - timedelta(days=KEEP_DAYS)).strftime('%Y%m%d')
    for day in recorded_days():
        if free > FREE_FLOOR_GB:
            return
        done, total, _ = coverage(day)       # strict: a set-aside window was never labelled
        covered = total and done >= total
        if day >= cutoff or not covered:
            continue
        for cam in ('cam1', 'cam2'):
            d = os.path.join(RAW, cam, day)
            if os.path.isdir(d):
                shutil.rmtree(d, ignore_errors=True)
        log('disk at %.0f GB: removed raw %s (%s)' % (free, day, 'labelled' if covered else 'older than %d days' % KEEP_DAYS))
        free = shutil.disk_usage(ROOT).free / 2**30


# ---------------------------------------------------------------- loop


def main():
    try:
        singleton = file_lock(os.path.join(LOGS, 'keeper.instance.lock'), timeout=0)
        singleton.__enter__()
    except TimeoutError:
        return
    log('keeper up: nights %s-%s, %d-minute windows, two teachers' % (NIGHT_START, NIGHT_STOP, MINUTES))
    last_disk = 0
    last_spawn = {}
    while True:
        try:
            beat()
            ensure_recorders()
            ensure('live counter', 'run_live.py', start_live)
            ensure('review site', 'label_pieces.py', start_labeler)
            ensure_jupyter()
            tunnel_up(5070, 'labeler')
            tunnel_up(8888, 'jupyter')
            read_urls()

            now = datetime.now()
            if training_hours(now):
                stop_teachers()
                from train_student_seg import should_train
                if not running('train_student_seg.py') and should_train(now):
                    log('student training hours: starting')
                    spawn('train_student', ['train_student_seg.py'])
                    time.sleep(30)
            elif in_night(now):
                if window_fits(now):
                    for tag, rng in WORKERS:      # each worker takes the oldest day with work left in its own half
                        if pass_alive(tag) or time.time() - last_spawn.get(tag, 0) < RESPAWN_GAP:
                            continue
                        day, done, total, held = next_day(rng)
                        if not day:
                            continue
                        clear_stale(tag)
                        log('night work on %s%s (%d/%d windows covered%s)' % (
                            day, tag, done, total, ', %d set aside after repeated failures' % held if held else ''))
                        spawn('autolabel%s' % tag, ['auto_label.py', day, str(MINUTES), TRAIN_FROM],
                              env={'RA_RANGE': rng, 'RA_WORKER': tag})
                        last_spawn[tag] = time.time()
                        time.sleep(20)
            else:
                stop_teachers()
                for l in running('train_student_seg.py'):
                    os.system('taskkill /PID %s /F /T > nul 2>&1' % l.split('|')[0].strip())
                    log('shop hours: student training stopped')

            if time.time() - last_disk > 1800:
                prune_raw(); last_disk = time.time()
        except Exception:
            import traceback
            log('keeper error\n' + traceback.format_exc())
        time.sleep(60)


if __name__ == '__main__':
    main()
