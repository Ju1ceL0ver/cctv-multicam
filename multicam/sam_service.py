"""Persistent, bounded SAM proposal jobs. This module never changes human labels."""
import math
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from storage import atomic_json, file_lock, read_json

ROOT = Path(__file__).resolve().parent
FPS = 25
MAX_FRAMES = 250
MAX_RUN_SECONDS = 480
MAX_QUEUE_SECONDS = 24 * 3600
MODELS = {
    'small': {'checkpoint': 'sam2.1_s.pt', 'source': 'sam2.1_small', 'memory_fraction': .23, 'free_mb': 3584},
    'large': {'checkpoint': 'sam2.1_l.pt', 'source': 'sam2.1_large', 'memory_fraction': .30, 'free_mb': 4608},
}
TERMINAL = {'ready', 'failed', 'cancelled'}
LOCAL_TZ = timezone(timedelta(hours=7))


def night_window(now=None):
    """The production counter owns the GPU from 09:45 through 20:59 UTC+7."""
    now = now or datetime.now(LOCAL_TZ)
    if now.tzinfo is not None:
        now = now.astimezone(LOCAL_TZ)
    minute = now.hour * 60 + now.minute
    return minute < 9 * 60 + 45 or minute >= 21 * 60


def job_dir(folder, job_id):
    if not isinstance(job_id, str) or not re.fullmatch(r'[0-9a-f]{32}', job_id):
        raise ValueError('Некорректный номер задания SAM')
    return Path(folder) / 'sam_jobs' / job_id


def _integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError('%s должен быть целым числом' % name)
    return value


def _numbers(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError('Некорректные координаты: ' + name)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value):
        raise ValueError('Координаты должны быть конечными числами')
    return [float(x) for x in value]


def validate_request(folder, request):
    """Normalize native-pixel prompts; frame numbers are relative to clip start."""
    if not isinstance(request, dict):
        raise ValueError('Ожидается описание исправления SAM')
    cam = request.get('cam')
    if cam not in ('cam1', 'cam2'):
        raise ValueError('Неизвестная камера')
    model = request.get('model', 'small')
    if model not in MODELS:
        raise ValueError('Модель должна быть small или large')
    frame = _integer(request.get('frame'), 'frame')
    end = _integer(request.get('end_frame', frame), 'end_frame')
    if frame < 0 or end < frame or end - frame + 1 > MAX_FRAMES:
        raise ValueError('Выберите от 1 до 250 последовательных кадров (не более 10 секунд)')
    revision = _integer(request.get('review_revision'), 'review_revision')
    if revision < 0:
        raise ValueError('Некорректная версия разметки')
    folder = Path(folder)
    meta = read_json(folder / 'meta_yolo26x-seg.json')
    if not meta:
        raise ValueError('Метаданные клипа не найдены')
    if end >= math.ceil(float(meta['seconds']) * FPS):
        raise ValueError('Интервал выходит за границу клипа')
    datetime.fromisoformat(meta['start'])
    cached = read_json(folder / 'review_media' / 'source.json', {})
    dims = cached.get('dimensions', {}).get(cam)
    if not isinstance(dims, (list, tuple)) or len(dims) != 2 or any(
        isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in dims
    ):
        raise ValueError('Сначала откройте исходное видео: неизвестно разрешение камеры')
    width, height = dims
    normalized = {'cam': cam, 'frame': frame, 'end_frame': end, 'model': model,
                  'review_revision': revision, 'dimensions': [width, height], 'fps': FPS}
    box = request.get('box')
    points = request.get('points')
    if box is not None:
        box = _numbers(box, 4, 'box')
        if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
            raise ValueError('Рамка выходит за границу исходного кадра')
        normalized['box'] = box
    if points is not None:
        if not isinstance(points, list) or not 1 <= len(points) <= 32:
            raise ValueError('Укажите от 1 до 32 точек')
        points = [_numbers(p, 2, 'points') for p in points]
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            raise ValueError('Точка выходит за границу исходного кадра')
        labels = request.get('point_labels')
        if not isinstance(labels, list) or len(labels) != len(points) or any(
            type(x) is not int or x not in (0, 1) for x in labels
        ):
            raise ValueError('Для каждой точки нужна метка 1 (человек) или 0 (фон)')
        if box is None and 1 not in labels:
            raise ValueError('Нужна хотя бы одна точка на человеке')
        normalized.update(points=points, point_labels=labels)
    if box is None and points is None:
        raise ValueError('Выберите человека рамкой или точками')
    normalized['source_snapshot'] = {k: meta[k] for k in ('day', 'start', 'seconds')}
    return normalized


def _launch_worker(folder, job_id):
    """On Windows Start-Process transfers ownership away from the Jupyter kernel."""
    target = job_dir(folder, job_id)
    args = [str(ROOT / 'sam_worker.py'), str(Path(folder).resolve()), job_id]
    if os.name == 'nt':
        def quote(s):
            return "'" + str(s).replace("'", "''") + "'"
        command = ('Start-Process -FilePath %s -ArgumentList %s -WorkingDirectory %s '
                   '-RedirectStandardOutput %s -RedirectStandardError %s -WindowStyle Hidden') % (
                       quote(sys.executable), quote(subprocess.list2cmdline(args)), quote(ROOT),
                       quote(target / 'worker.log'), quote(target / 'worker.err'))
        subprocess.Popen(['powershell', '-NoProfile', '-NonInteractive', '-Command', command],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        # Supports local API fixtures; actual GPU inference is Windows-only below.
        with (target / 'worker.log').open('ab') as log:
            subprocess.Popen([sys.executable, '-u'] + args, cwd=ROOT, stdin=subprocess.DEVNULL,
                             stdout=log, stderr=log, start_new_session=True, close_fds=True)


def enqueue(folder, request):
    folder = Path(folder).resolve()
    normalized = validate_request(folder, request)
    with file_lock(folder / 'sam_jobs' / 'enqueue.lock'):
        active = []
        for path in (folder / 'sam_jobs').glob('*/job.json'):
            saved = read_json(path, {})
            if saved.get('status') not in TERMINAL:
                active.append(get_job(folder, path.parent.name))
        if sum(j.get('status') not in TERMINAL for j in active) >= 4:
            raise ValueError('Уже есть четыре задания SAM; отмените ненужные или дождитесь результата')
        job_id = uuid.uuid4().hex
        now = time.time()
        job = {'id': job_id, 'status': 'queued', 'request': normalized,
               'review_revision': normalized['review_revision'],
               'source': MODELS[normalized['model']]['source'],
               'source_snapshot': normalized['source_snapshot'],
               'created_at': now, 'updated_at': now, 'expires_at': now + MAX_QUEUE_SECONDS,
               'frame_count': normalized['end_frame'] - normalized['frame'] + 1,
               'processed_frames': 0, 'cancel_requested': False,
               'message': 'Ожидает свободной видеопамяти' if night_window() else 'В очереди до 21:00 (UTC+7)'}
        atomic_json(job_dir(folder, job_id) / 'job.json', job)
    try:
        _launch_worker(folder, job_id)
    except Exception as exc:
        update_job(folder, job_id, status='failed', message='Не удалось запустить обработчик', error=str(exc)[:1000])
    return get_job(folder, job_id)


def update_job(folder, job_id, **changes):
    target = job_dir(folder, job_id)
    with file_lock(target / 'state.lock'):
        job = read_json(target / 'job.json')
        if job is None:
            raise FileNotFoundError('Задание SAM не найдено')
        # A cancelled job must never be resurrected by a late worker result.
        if job.get('cancel_requested') and changes.get('status') == 'ready':
            changes.update(status='cancelled', message='Задание отменено')
        job.update(changes, updated_at=time.time())
        atomic_json(target / 'job.json', job)
        return job


def get_job(folder, job_id):
    target = job_dir(folder, job_id)
    job = read_json(target / 'job.json')
    if job is None:
        raise FileNotFoundError('Задание SAM не найдено')
    if job['status'] not in TERMINAL and time.time() > job['expires_at']:
        job = update_job(folder, job_id, status='failed', message='Истекло время ожидания SAM', cancel_requested=True)
    elif job['status'] == 'queued' and time.time() > job['updated_at'] + 180:
        job = update_job(folder, job_id, status='failed', message='Обработчик очереди SAM перестал отвечать; запустите задание снова', cancel_requested=True)
    elif job['status'] == 'running' and time.time() > job.get('started_at', time.time()) + MAX_RUN_SECONDS + 60:
        job = update_job(folder, job_id, status='failed', message='Обработчик SAM перестал отвечать', cancel_requested=True)
    if job['status'] == 'ready':
        proposal = read_json(target / 'proposals.json')
        if not proposal:
            return update_job(folder, job_id, status='failed', message='Результат SAM не найден')
        job = dict(job, frames=proposal['frames'])
    return job


def cancel_job(folder, job_id):
    job = get_job(folder, job_id)
    if job['status'] in TERMINAL:
        return job
    return update_job(folder, job_id, cancel_requested=True, status='cancelled', message='Задание отменено')


def job_path(folder, job_id):
    """Validated job directory, e.g. for an API-level accept.lock."""
    return job_dir(folder, job_id)


def record_acceptance(folder, job_id, revision, frames, accepted_label=None):
    """Record successful canonical transactions; never applies annotations itself.

    Callers serialize acceptance with accept.lock and first commit the canonical
    annotation transaction using accepted_revision (or original review_revision).
    This record lets successive accepted frames share one inference job.
    """
    revision = _integer(revision, 'revision')
    if accepted_label is not None and (not isinstance(accepted_label, str) or not accepted_label
                                       or len(accepted_label) > 60 or accepted_label == 'NEW'):
        raise ValueError('Нужна сохранённая метка человека, а не команда NEW')
    if not isinstance(frames, (list, tuple)):
        raise ValueError('Ожидается список принятых кадров')
    frames = [_integer(frame, 'frame') for frame in frames]
    target = job_dir(folder, job_id)
    with file_lock(target / 'state.lock'):
        job = read_json(target / 'job.json')
        if not job or job.get('status') != 'ready':
            raise ValueError('Для принятия нужен готовый результат SAM')
        previous = job.get('accepted_revision', job['review_revision'])
        if revision <= previous:
            raise ValueError('Версия сохранённой разметки должна увеличиться')
        req = job['request']
        if not frames or any(frame < req['frame'] or frame > req['end_frame'] for frame in frames):
            raise ValueError('Принятый кадр не входит в предложение SAM')
        accepted = sorted(set(job.get('accepted_frames', [])) | set(frames))
        job.update(accepted_revision=revision, accepted_frames=accepted, updated_at=time.time())
        if accepted_label is not None:
            job['accepted_label'] = accepted_label
        atomic_json(target / 'job.json', job)
        return job
