"""Record the camera's RTSP stream verbatim, alongside the live pipeline.

Why this exists: the pipeline's own recordings are written from inside its
main loop, so they contain only the frames it managed to process -- measured
at ~12.3 fps against a 25 fps camera, tagged as 25 and therefore also played
back at double speed. Half of the motion information is discarded before the
recorder ever sees it, which is precisely the half a tracker needs.

This process does not decode anything. It remuxes RTSP packets straight to
disk (``-c copy``), so it costs no GPU and almost no CPU, and what lands on
disk is the camera's own 2560x1440 @ 25 fps -- twice the temporal resolution
and twice the linear resolution of anything we have today. Measured on this
camera: 4.1 Mbit/s, 1.8 GB per hour, ~20 GB per trading day.

Verified before writing this: the camera serves two simultaneous clients
without either losing frames (both got 301/301 frames over 12 s), so the live
service is not disturbed.
"""

import os, sys, json, time, shutil, signal, subprocess
from datetime import datetime, timedelta

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
# One process per camera. Each owns runs/raw/<camera>/ and applies its disk cap
# only there, so one camera's recorder can never prune the other's footage (or
# the first two cam1 days, which live directly under runs/raw/<day>).
CAMERA = os.environ.get('RA_RAW_CAMERA', 'cam1')
OUT = os.path.join(ROOT, 'runs', 'raw', CAMERA)
LOG = os.path.join(HOME, '_raw_recorder_%s.log' % CAMERA)
PIDFILE = os.path.join(HOME, '_raw_recorder_%s.pid' % CAMERA)

HOURS = (10, 21)              # same trading window as the live service
SEGMENT_SECONDS = 900         # 15-minute files: small enough to copy/inspect
RECORD_DAYS = int(os.environ.get('RA_RAW_DAYS', '5'))
MAX_GIGABYTES = 120.0         # per camera; oldest segments are dropped first
DISK_FREE_FLOOR_GB = 200.0    # never let the volume get tighter than this


def log(*a):
    line = '%s %s' % (time.strftime('%Y-%m-%d %H:%M:%S'), ' '.join(str(x) for x in a))
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def tree_size(path):
    total = 0
    for dp, _, fn in os.walk(path):
        for f in fn:
            try:
                total += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return total


def enforce_disk_cap():
    """Drop oldest segments first. Today's own files are fair game here --
    unlike the pipeline's retention, this is scratch data for one experiment,
    and running the volume dry would take the live service down with it."""
    if not os.path.isdir(OUT):
        return
    files = []
    for dp, _, fn in os.walk(OUT):
        for f in fn:
            if f.endswith('.mp4'):
                p = os.path.join(dp, f)
                try:
                    files.append((os.path.getmtime(p), p))
                except OSError:
                    pass
    files.sort()
    limit = MAX_GIGABYTES * 1024 ** 3
    size = sum(os.path.getsize(p) for _, p in files if os.path.exists(p))
    freed = 0
    while files and (size > limit or
                     shutil.disk_usage(OUT).free < DISK_FREE_FLOOR_GB * 1e9):
        _, victim = files.pop(0)
        try:
            s = os.path.getsize(victim)
            os.remove(victim)
            size -= s
            freed += s
        except OSError:
            break
    if freed:
        log('disk cap: freed %.1f GB' % (freed / 1e9))


def within_hours(now=None):
    now = now or datetime.now()
    return HOURS[0] <= now.hour < HOURS[1]


def next_window_start(now=None):
    now = now or datetime.now()
    start = now.replace(hour=HOURS[0], minute=0, second=0, microsecond=0)
    if start <= now:
        start += timedelta(days=1)
    return start


def record_one_day(ff, url, day_end):
    """Run ffmpeg until the window closes, restarting it if the stream drops."""
    day_dir = os.path.join(OUT, datetime.now().strftime('%Y%m%d'))
    os.makedirs(day_dir, exist_ok=True)
    attempt = 0
    while datetime.now() < day_end:
        remaining = int((day_end - datetime.now()).total_seconds())
        if remaining < 30:
            break
        pattern = os.path.join(day_dir, '%s_%%04d.mp4' % datetime.now().strftime('%H%M%S'))
        cmd = [ff, '-hide_banner', '-loglevel', 'warning',
               '-rtsp_transport', 'tcp', '-i', url,
               '-t', str(remaining), '-c', 'copy', '-an',
               '-f', 'segment', '-segment_time', str(SEGMENT_SECONDS),
               '-reset_timestamps', '1', '-strftime', '0', pattern]
        attempt += 1
        log('ffmpeg start (attempt %d), %d s remaining in window' % (attempt, remaining))
        started = time.time()
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        last_check = time.time()
        while proc.poll() is None:
            time.sleep(5)
            if datetime.now() >= day_end:
                log('window closed; stopping ffmpeg')
                proc.terminate()
                try:
                    proc.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    proc.kill()
                break
            if time.time() - last_check > 300:
                last_check = time.time()
                enforce_disk_cap()
                log('recording, %.1f GB on disk' % (tree_size(OUT) / 1e9))
        err = b''
        try:
            err = proc.stderr.read() or b''
        except Exception:
            pass
        ran = time.time() - started
        if datetime.now() >= day_end:
            break
        log('ffmpeg exited after %.0f s: %s' % (ran, err.decode('utf-8', 'replace')[-300:]))
        # A stream drop should not spin: back off, but stay responsive.
        time.sleep(10)


def main():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    import imageio_ffmpeg
    from retail_analytics.config import load_config, load_dotenv, CameraConfig

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    if CAMERA == 'cam1':
        url = load_config().camera.rtsp_url()
    else:
        # .env carries only cam1's credentials; the owner confirmed both cameras
        # share them, so cam2 falls back to cam1's password unless its own is set.
        load_dotenv()
        env = os.environ
        tag = CAMERA.upper()
        url = CameraConfig(
            name=CAMERA,
            ip=env.get('RA_CAM_%s_IP' % tag, '192.168.99.242'),
            user=env.get('RA_CAM_%s_USER' % tag) or env.get('RA_CAM_CAM1_USER', 'admin'),
            password=env.get('RA_CAM_%s_PASSWORD' % tag) or env.get('RA_CAM_CAM1_PASSWORD', ''),
        ).rtsp_url()
    os.makedirs(OUT, exist_ok=True)
    with open(PIDFILE, 'w') as f:
        f.write(str(os.getpid()))
    log('raw recorder %s up (pid %d); window %02d:00-%02d:00, %d day(s), cap %.0f GB'
        % (CAMERA, os.getpid(), HOURS[0], HOURS[1], RECORD_DAYS, MAX_GIGABYTES))
    log('ffmpeg:', ff)

    for day in range(RECORD_DAYS):
        if not within_hours():
            start = next_window_start()
            log('waiting until %s' % start.isoformat(timespec='seconds'))
            while datetime.now() < start:
                time.sleep(30)
        day_end = datetime.now().replace(hour=HOURS[1], minute=0, second=0, microsecond=0)
        log('day %d/%d: recording until %s' % (day + 1, RECORD_DAYS,
                                               day_end.strftime('%H:%M')))
        enforce_disk_cap()
        record_one_day(ff, url, day_end)
        log('day %d done; %.1f GB on disk' % (day + 1, tree_size(OUT) / 1e9))

    log('all requested days recorded; exiting')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        log('FATAL\n' + traceback.format_exc())
    finally:
        try:
            os.remove(PIDFILE)
        except OSError:
            pass
