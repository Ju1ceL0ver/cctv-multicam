"""Detached SAM supervisor and bounded GPU child. No proposals are auto-approved."""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from sam_service import (FPS, MAX_RUN_SECONDS, MODELS, ROOT, TERMINAL, get_job,
                         job_dir, night_window, update_job)
from storage import atomic_json, file_lock, read_json


class Cancelled(RuntimeError):
    pass


def _check_abort(folder, job_id):
    job = read_json(job_dir(folder, job_id) / 'job.json', {})
    if job.get('cancel_requested') or job.get('status') in ('cancelled', 'failed'):
        raise Cancelled('Задание отменено')
    if not night_window():
        raise RuntimeError('Наступило дневное окно боевого счётчика; SAM остановлен')


def _gpu_free_mb():
    result = subprocess.run(['nvidia-smi', '--id=0', '--query-gpu=memory.free', '--format=csv,noheader,nounits'],
                            capture_output=True, text=True, timeout=10, check=True)
    return int(result.stdout.strip().splitlines()[0])


def run_supervisor(folder, job_id):
    folder = Path(folder).resolve()
    target = job_dir(folder, job_id)
    try:
        with file_lock(target / 'worker.lock', timeout=0):
            while True:
                job = get_job(folder, job_id)
                if job['status'] in TERMINAL:
                    return
                if not night_window():
                    update_job(folder, job_id, status='queued', message='В очереди до 21:00 (UTC+7)', next_check_at=time.time()+30)
                    time.sleep(30)
                    continue
                try:
                    with file_lock(ROOT / 'data' / 'logs' / 'sam_gpu.lock', timeout=0):
                        required = MODELS[job['request']['model']]['free_mb']
                        free = _gpu_free_mb()
                        if free >= required:
                            _run_child(folder, job_id)
                            return
                        update_job(folder, job_id, status='queued', message='Ожидает свободной видеопамяти',
                                   free_vram_mb=free, required_free_vram_mb=required, next_check_at=time.time()+30)
                except TimeoutError:
                    update_job(folder, job_id, status='queued', message='Ожидает завершения другого задания SAM', next_check_at=time.time()+30)
                time.sleep(30)
    except TimeoutError:
        return  # Another supervisor already owns this specific job.
    except Exception as exc:
        update_job(folder, job_id, status='failed', message='Не удалось выполнить SAM', error=str(exc)[:1500])


def _stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def _run_child(folder, job_id):
    target = job_dir(folder, job_id)
    _check_abort(folder, job_id)
    update_job(folder, job_id, status='running', started_at=time.time(), message='SAM обрабатывает выбранный интервал')
    args = [sys.executable, '-u', str(ROOT / 'sam_worker.py'), str(folder), job_id, '--infer']
    with (target / 'inference.log').open('ab') as log:
        child = subprocess.Popen(args, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                                 close_fds=True, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        started = time.monotonic()
        try:
            while child.poll() is None:
                _check_abort(folder, job_id)
                if time.monotonic() - started > MAX_RUN_SECONDS:
                    raise TimeoutError('SAM превысил лимит 8 минут; попробуйте более короткий интервал')
                time.sleep(1)
            if child.returncode:
                job = read_json(target / 'job.json', {})
                if job.get('status') not in TERMINAL:
                    raise RuntimeError('Обработчик SAM завершился с кодом %s; подробности в inference.log' % child.returncode)
            elif read_json(target / 'job.json', {}).get('status') not in TERMINAL:
                raise RuntimeError('Обработчик завершился без результата SAM')
        except Cancelled:
            _stop_child(child)
            update_job(folder, job_id, status='cancelled', message='Задание отменено')
        except Exception as exc:
            _stop_child(child)
            update_job(folder, job_id, status='failed', message=str(exc)[:500])
        finally:
            _stop_child(child)


def encode_rle(mask):
    """Lossless uncompressed COCO RLE: column-major, first run is background."""
    import numpy as np
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.ndim != 2:
        raise ValueError('Ожидается двумерная маска')
    flat = mask.astype(bool).ravel(order='F')
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    runs = np.diff(np.concatenate(([0], changes, [len(flat)]))).tolist()
    if len(flat) and flat[0]:
        runs.insert(0, 0)
    return {'size': [int(mask.shape[0]), int(mask.shape[1])], 'counts': runs}


def prompt_kwargs(request):
    # A flat Nx2 list means N objects in Ultralytics. The extra dimension makes
    # all positive/negative clicks refer to ONE person, not separate objects.
    result = {}
    if request.get('box') is not None:
        result['bboxes'] = [request['box']]
    if request.get('points') is not None:
        result['points'] = [request['points']]
        result['labels'] = [request['point_labels']]
    return result


def prepare_video(folder, request, destination, check_abort=lambda: None):
    """Losslessly encode all requested raw frames; fail on gaps or wrong seek."""
    import cv2
    from rawsource import Stream
    meta = read_json(Path(folder) / 'meta_yolo26x-seg.json')
    if {k: meta[k] for k in ('day', 'start', 'seconds')} != request['source_snapshot']:
        raise RuntimeError('Исходный клип изменился после создания задания')
    start = datetime.fromisoformat(meta['start'])
    first = request['frame']
    expected_count = request['end_frame'] - first + 1
    width, height = request['dimensions']
    stream = Stream(request['cam'], meta['day'])
    writer = None
    try:
        stream.seek(start + timedelta(seconds=first / FPS))
        writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'FFV1'), FPS, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Не удалось открыть lossless FFV1-кодер')
        for offset in range(expected_count):
            check_abort()
            actual_time, image = stream.read()
            expected_time = start + timedelta(seconds=(first + offset) / FPS)
            if image is None or actual_time is None or abs((actual_time - expected_time).total_seconds()) > .5 / FPS:
                raise RuntimeError('Разрыв записи или неточное позиционирование исходного кадра %s' % (first + offset))
            if image.shape[:2] != (height, width):
                raise RuntimeError('Разрешение исходного видео изменилось')
            writer.write(image)
    finally:
        if writer is not None:
            writer.release()
        if stream.cap is not None:
            stream.cap.release()
    check = cv2.VideoCapture(str(destination))
    try:
        if not check.isOpened() or int(check.get(cv2.CAP_PROP_FRAME_COUNT)) != expected_count:
            raise RuntimeError('Не удалось сохранить все исходные кадры SAM')
    finally:
        check.release()
    return expected_count


def mask_proposal(mask, request, frame):
    import cv2
    import numpy as np
    width, height = request['dimensions']
    mask = np.asarray(mask, dtype=np.uint8)
    if mask.shape != (height, width):
        raise RuntimeError('SAM вернул маску другого разрешения; результат не будет масштабирован наугад')
    ys, xs = np.nonzero(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = [cv2.approxPolyDP(c, 1., True).reshape(-1, 2).tolist()
                for c in sorted(contours, key=cv2.contourArea, reverse=True)]
    polygon = polygons[0] if polygons else []
    return {'cam': request['cam'], 'frame': frame,
            'box': [int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1] if len(xs) else None,
            'polygon': polygon, 'polygons': polygons, 'mask_rle': encode_rle(mask), 'visible': bool(len(xs)),
            'source': MODELS[request['model']]['source']}


def infer(folder, job_id):
    # Importing service/worker on the Mac for validation never initializes CUDA.
    if os.name != 'nt':
        raise RuntimeError('Эксперименты SAM выполняются только на Windows-машине с GPU')
    import numpy as np
    import torch
    from ultralytics.models.sam import SAM2VideoPredictor
    target = job_dir(folder, job_id)
    job = read_json(target / 'job.json')
    request = job['request']
    config = MODELS[request['model']]
    _check_abort(folder, job_id)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA недоступна')
    free, _ = torch.cuda.mem_get_info(0)
    if free < config['free_mb'] * 1024**2:
        raise RuntimeError('Свободная видеопамять закончилась до запуска SAM; повторите позже')
    torch.cuda.set_per_process_memory_fraction(config['memory_fraction'], 0)
    torch.cuda.reset_peak_memory_stats(0)
    video = target / 'input.avi'
    count = prepare_video(folder, request, video, lambda: _check_abort(folder, job_id))
    update_job(folder, job_id, message='SAM переносит выбранного человека по видео', native_dimensions=request['dimensions'])
    predictor = SAM2VideoPredictor(overrides={'conf': .25, 'task': 'segment', 'mode': 'predict',
        'imgsz': 1024, 'model': config['checkpoint'], 'device': 0, 'save': False,
        'verbose': False, 'vid_stride': 1, 'batch': 1})
    proposals = []
    started = time.monotonic()
    for offset, result in enumerate(predictor(source=str(video), stream=True, **prompt_kwargs(request))):
        _check_abort(folder, job_id)
        if offset >= count:
            raise RuntimeError('SAM вернул лишние кадры')
        masks = result.masks
        if masks is None or len(masks.data) == 0:
            mask = np.zeros((request['dimensions'][1], request['dimensions'][0]), dtype=np.uint8)
        else:
            if len(masks.data) != 1:
                raise RuntimeError('SAM вернул несколько объектов вместо одного; уточните точки')
            mask = masks.data[0].detach().cpu().numpy() > .5
        proposals.append(mask_proposal(mask, request, request['frame'] + offset))
        if offset % 5 == 0:
            update_job(folder, job_id, processed_frames=offset + 1)
    if len(proposals) != count:
        raise RuntimeError('SAM обработал %s из %s кадров; неполный результат отклонён' % (len(proposals), count))
    _check_abort(folder, job_id)
    if not proposals or not proposals[0]['visible']:
        raise RuntimeError('SAM не нашёл человека на выбранном кадре; уточните рамку или точки')
    atomic_json(target / 'proposals.json', {'frames': proposals, 'review_revision': job['review_revision'],
        'source_snapshot': job['source_snapshot'], 'model': config['checkpoint'], 'frame_step': 1,
        'first_source_frame': request['frame'], 'last_source_frame': request['end_frame']})
    update_job(folder, job_id, status='ready', message='Предложение готово: проверьте перед сохранением',
               processed_frames=len(proposals), elapsed_inference_s=round(time.monotonic()-started, 3),
               peak_allocated_mb=round(torch.cuda.max_memory_allocated(0)/1024**2),
               peak_reserved_mb=round(torch.cuda.max_memory_reserved(0)/1024**2),
               requires_human_review=True)
    video.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('job_id')
    parser.add_argument('--infer', action='store_true')
    args = parser.parse_args()
    if args.infer:
        try:
            infer(args.folder, args.job_id)
        except Cancelled:
            update_job(args.folder, args.job_id, status='cancelled', message='Задание отменено')
        except Exception as exc:
            update_job(args.folder, args.job_id, status='failed', message='SAM не завершил обработку', error=str(exc)[:1500])
            raise
    else:
        run_supervisor(args.folder, args.job_id)


if __name__ == '__main__':
    main()
