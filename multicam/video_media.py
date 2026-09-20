"""Exact raw-frame images and clean video proxies for the correction editor.

Every proxy carries an explicit output-frame -> raw-frame map. Human coordinates
always refer to the native camera image, never to a CSS size or proxy timestamp.
"""
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import cv2

from rawsource import FPS, Stream, segments
from storage import atomic_json, file_lock, read_json


def metadata(folder):
    data = read_json(Path(folder) / 'meta_yolo26x-seg.json')
    if not data or not math.isfinite(float(data.get('seconds', 0))) or float(data['seconds']) <= 0:
        raise ValueError('Нет исходных параметров записи')
    return data


def validate_frame(folder, cam, frame):
    if cam not in ('cam1', 'cam2'):
        raise ValueError('Неизвестная камера')
    if isinstance(frame, bool) or int(frame) != float(frame):
        raise ValueError('Нужен целый номер исходного кадра')
    frame = int(frame)
    meta = metadata(folder)
    if not 0 <= frame < math.ceil(float(meta['seconds']) * FPS):
        raise ValueError('Кадр находится вне записи')
    return meta, frame


def source_identity(folder, cam, frame):
    meta, frame = validate_frame(folder, cam, frame)
    when = datetime.fromisoformat(meta['start']) + timedelta(seconds=frame / FPS)
    for name, start in segments(cam, meta['day']):
        if start <= when < start + timedelta(seconds=900):
            path = Path(name)
            stat = path.stat()
            index = int(round((when - start).total_seconds() * FPS))
            return {'cam': cam, 'day': meta['day'], 'file': path.name,
                    'frame_index': index, 'clip_frame': frame, 'fps': FPS,
                    'pts_seconds': index / FPS, 'bytes': stat.st_size,
                    'mtime_ns': stat.st_mtime_ns, 'start': meta['start']}
    raise ValueError('Исходная запись этого кадра недоступна')


def read_frame(folder, cam, frame):
    meta, frame = validate_frame(folder, cam, frame)
    when = datetime.fromisoformat(meta['start']) + timedelta(seconds=frame / FPS)
    stream = Stream(cam, meta['day'])
    try:
        stream.seek(when)
        stamp, image = stream.read()
        if image is None or stamp is None:
            raise ValueError('Исходный кадр не найден')
        actual = round((stamp - datetime.fromisoformat(meta['start'])).total_seconds() * FPS)
        if actual != frame:
            raise ValueError('В записи пропуск: запрошенный кадр отсутствует')
        return image
    finally:
        if stream.cap is not None:
            stream.cap.release()


def remember_dimensions(folder, cam, image):
    path = Path(folder) / 'review_media/source.json'
    with file_lock(path.with_suffix('.lock')):
        state = read_json(path, {'dimensions': {}})
        dimensions = [int(image.shape[1]), int(image.shape[0])]
        if state.setdefault('dimensions', {}).get(cam) == dimensions:
            return
        state['dimensions'][cam] = dimensions
        atomic_json(path, state)


def frame_image(folder, cam, frame):
    identity = source_identity(folder, cam, frame)
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(folder) / 'review_media/frames' / (key + '.jpg')
    with file_lock(path.with_suffix('.lock')):
        if not path.exists():
            image = read_frame(folder, cam, frame)
            remember_dimensions(folder, cam, image)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(key + '.tmp.jpg')
            if not cv2.imwrite(str(tmp), image, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                raise RuntimeError('Не удалось сохранить кадр')
            os.replace(tmp, path)
    return path, identity


def display_image(folder, cam, frame, *, preview=False, roi=None):
    """Small overview or native-resolution crop; annotations keep raw coordinates.

    The original cached frame is retained for model input and detailed inspection.
    Display derivatives never become training images or model inputs.
    """
    original, identity = frame_image(folder, cam, frame)
    if not preview and roi is None:
        return original, identity
    dimensions = read_json(Path(folder) / 'review_media/source.json', {}).get('dimensions', {}).get(cam)
    if dimensions is None:
        image = cv2.imread(str(original))
        if image is None:
            raise ValueError('Не удалось прочитать исходный кадр')
        remember_dimensions(folder, cam, image)
        dimensions = [image.shape[1], image.shape[0]]
    if roi is not None:
        if (not isinstance(roi, (list, tuple)) or len(roi) != 4 or
                any(type(v) is not int for v in roi)):
            raise ValueError('Нужна область из четырёх целых координат')
        x1, y1, x2, y2 = roi
        if not (0 <= x1 < x2 <= dimensions[0] and 0 <= y1 < y2 <= dimensions[1]):
            raise ValueError('Область просмотра выходит за границы исходного кадра')
    key = hashlib.sha256(json.dumps([identity, 'display-v1', bool(preview), roi], sort_keys=True).encode()).hexdigest()[:24]
    target = Path(folder) / 'review_media/frames' / (key + '.webp')
    with file_lock(target.with_suffix('.lock')):
        if not target.exists():
            image = cv2.imread(str(original))
            if image is None:
                raise ValueError('Не удалось прочитать исходный кадр')
            if roi is not None:
                image = image[y1:y2, x1:x2]
                quality = 90
            else:
                height, width = image.shape[:2]
                if width > 960:
                    image = cv2.resize(image, (960, round(height * 960 / width)), interpolation=cv2.INTER_AREA)
                quality = 70
            temporary = target.with_name(key + '.tmp.webp')
            if not cv2.imwrite(str(temporary), image, [cv2.IMWRITE_WEBP_QUALITY, quality]):
                raise ValueError('Не удалось подготовить кадр для просмотра')
            os.replace(temporary, target)
    return target, identity


def video_window(folder, cam, frame, seconds=8):
    meta, frame = validate_frame(folder, cam, frame)
    length = max(1, min(10, float(seconds)))
    total = math.ceil(float(meta['seconds']) * FPS)
    count = min(total, int(round(length * FPS)))
    start = max(0, min(frame - count // 3, total - count))
    end = min(total, start + count)
    identities = [source_identity(folder, cam, start), source_identity(folder, cam, end - 1)]
    key = hashlib.sha256(json.dumps([identities, 1], sort_keys=True).encode()).hexdigest()[:24]
    target = Path(folder) / 'review_media' / (key + '.mp4')
    sidecar = target.with_suffix('.json')
    with file_lock(target.with_suffix('.lock'), timeout=60):
        if target.exists() and sidecar.exists():
            return read_json(sidecar)
        import imageio_ffmpeg
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(key + '.tmp.mp4')
        stream = Stream(cam, meta['day'])
        stream.seek(datetime.fromisoformat(meta['start']) + timedelta(seconds=start / FPS))
        proc = None
        raw_frames = []
        error = None
        try:
            for k in range(start, end):
                stamp, image = stream.read()
                if image is None:
                    break
                actual = int(round((stamp - datetime.fromisoformat(meta['start'])).total_seconds() * FPS))
                if actual != k:
                    raise ValueError('В видео пропуск исходных кадров; интервал недоступен')
                if (k - start) % 2:
                    continue
                if proc is None:
                    remember_dimensions(folder, cam, image)
                    h, w = image.shape[:2]
                    width = min(1280, w) // 2 * 2
                    height = round(h * width / w / 2) * 2
                    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
                           '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{width}x{height}',
                           '-r', str(FPS / 2), '-i', '-', '-an', '-c:v', 'libx264',
                           '-threads', '2', '-preset', 'veryfast', '-crf', '23',
                           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(tmp)]
                    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                proc.stdin.write(cv2.resize(image, (width, height)).tobytes())
                raw_frames.append(k)
        except Exception as exc:
            error = exc
        finally:
            if stream.cap is not None:
                stream.cap.release()
            if proc is not None:
                proc.stdin.close()
                details = proc.stderr.read().decode(errors='replace')
                if proc.wait(timeout=30) and error is None:
                    error = RuntimeError(details[-400:])
        if error is not None or not raw_frames:
            tmp.unlink(missing_ok=True)
            raise error or RuntimeError('Видеофрагмент пуст')
        result = {'name': target.name, 'cam': cam, 'start_frame': start,
                  'frames': raw_frames, 'fps': FPS / 2, 'source_fps': FPS,
                  'width': width, 'height': height, 'source': identities}
        os.replace(tmp, target)
        atomic_json(sidecar, result)
        return result
