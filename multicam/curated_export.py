"""Select corrected full frames without losing added people or human mask edits."""
import numpy as np
import cv2
from rawsource import FPS


def polygon_for_training(observation):
    """YOLO polygon labels cannot represent holes/disjoint islands losslessly.

Keep their exact RLE in the review store and skip such training frames rather than
silently filling occluders or discarding a visible body part.
"""
    rle = observation.get('mask_rle')
    if rle:
        h, w = map(int, rle['size'])
        counts = rle['counts']
        if not isinstance(counts, list) or sum(counts) != h * w:
            return None
        flat = np.zeros(h * w, np.uint8)
        offset = 0
        for i, count in enumerate(counts):
            if i % 2:
                flat[offset:offset + count] = 1
            offset += count
        mask = flat.reshape((h, w), order='F')
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) != 1 or hierarchy is None or hierarchy[0][0][2] != -1:
            return None
        poly = contours[0].reshape(-1, 2).astype(float)
    else:
        poly = np.asarray(observation.get('polygon') or [], dtype=float)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] != 2 or not np.isfinite(poly).all():
        return None
    return poly


def select_corrected_frames(folder, cam, detections, state, gap=1.0):
    from video_annotations import frame_data
    video = state.get('video', {})
    frames = set(np.rint(detections[:, 0] * FPS).astype(int).tolist())
    for obs in video.get('observations', {}).values():
        if obs.get('cam') == cam:
            frames.add(int(obs['frame']))
    for key in video.get('frame_reviews', {}):
        c, f = key.split(':', 1)
        if c == cam:
            frames.add(int(f))
    wanted, skipped, last = {}, 0, -1e9
    details = {'incomplete': 0, 'unresolved': 0, 'unrepresentable_mask': 0}
    for frame in sorted(frames):
        data = frame_data(folder, cam, frame, state=state)
        observations = data['observations']
        edited = any(o.get('cam') == cam and int(o.get('frame', -1)) == frame
                     for o in video.get('observations', {}).values())
        # A newly added/changed observation invalidates completeness. Require the
        # user's full-frame check before the image can become a training example.
        if (edited or not data.get('teacher_sampled', True)) and not data.get('frame_reviewed'):
            skipped += 1; details['incomplete'] += 1; continue
        rows = []
        failure = None
        for obs in observations:
            quality = obs.get('quality', 'unreviewed')
            if quality == 'false_positive':
                continue
            is_teacher = obs.get('source', 'teacher') in ('teacher', 'teacher_mask')
            usable = quality == 'valid' or (is_teacher and quality == 'unreviewed' and obs.get('label') not in (None, '', '?'))
            if not usable:
                failure = 'unresolved'; break
            poly = polygon_for_training(obs)
            if poly is None:
                failure = 'unrepresentable_mask'; break
            rows.append({'id': obs['id'], 'polygon': poly, 'source': obs.get('source'),
                         'mask_review_source': obs.get('mask_review_source'),
                         'frame_reviewed': bool(data.get('frame_reviewed'))})
        if failure:
            skipped += 1; details[failure] += 1; continue
        if not observations and not data.get('frame_reviewed'):
            continue
        if frame / FPS - last < gap:
            continue
        wanted[frame] = rows
        last = frame / FPS
    return wanted, skipped, details
