"""Authenticated correction API, registered on the existing labeler application."""
import math
from pathlib import Path

import numpy as np
from flask import abort, jsonify, render_template, request, send_file

from rawsource import FPS
from review_store import Conflict, read_state
from storage import read_json
from video_media import display_image, frame_image, metadata, validate_frame, video_window


def register(app, clip_folder, root):
    root = Path(root)

    def failure(exc):
        return jsonify({'error': str(exc)}), 409 if isinstance(exc, Conflict) else 400

    def body_with_revision():
        body = request.get_json(force=True)
        if not isinstance(body, dict) or 'revision' not in body:
            raise ValueError('Нужна версия разметки; обновите запись перед сохранением')
        return body

    @app.get('/video')
    def video_editor():
        return render_template('video.html')

    @app.get('/api/video/<clip>/info')
    def video_info(clip):
        folder = clip_folder(clip)
        try:
            meta = metadata(folder)
            state = read_state(folder)
            samples = {}
            with np.load(folder / 'dets_yolo26x-seg.npz') as z:
                for cam in ('cam1', 'cam2'):
                    samples[cam] = sorted(set(np.rint(z[cam][:, 0] * FPS).astype(int).tolist()))
                pieces = []
                for p in state['pieces']:
                    item = {k: v for k, v in p.items() if k != 'dets'}
                    frames = np.rint(z[p['cam']][p['dets'], 0] * FPS).astype(int)
                    if len(frames):
                        item.update(start_frame=int(frames.min()), end_frame=int(frames.max()),
                                    raw_t0=float(frames.min() / FPS), raw_t1=float(frames.max() / FPS))
                        pieces.append(item)
            grid = folder / 'grid_yolo26x-seg.npz'
            if grid.exists():
                with np.load(grid) as gz:
                    grid_frames = sorted(set(np.rint(gz['t'] * FPS).astype(int).tolist()))
                # Includes sampled empty frames, which are essential for missing-person review.
                samples = {cam: grid_frames for cam in ('cam1', 'cam2')}
            source = read_json(folder / 'review_media/source.json', {})
            dimensions = source.get('dimensions', {})
            for cam in ('cam1', 'cam2'):
                if cam not in dimensions:
                    try:
                        frame_image(folder, cam, samples[cam][0] if samples[cam] else 0)
                    except (ValueError, OSError):
                        pass
            dimensions = read_json(folder / 'review_media/source.json', {}).get('dimensions', {})
            width, height = dimensions.get('cam1', [2560, 1440])
            local = read_json(folder / 'sync_estimate.json', {})
            table = read_json(root / 'data/cam_sync.json', {})
            sync = (local if local.get('status') == 'accepted' else
                    table.get('clips', {}).get(clip) or table.get(meta['day']) or {})
            known_labels = set(state['labels'].values())
            known_labels.update(o.get('label') for o in state.get('video', {}).get('observations', {}).values())
            known_labels.update(i.get('value') for i in state.get('video', {}).get('intervals', []) if i.get('kind') == 'label')
            return jsonify({'clip': clip, 'meta': meta, 'revision': state['revision'], 'fps': FPS,
                            'width': width, 'height': height, 'dimensions': dimensions,
                            'frames': math.ceil(meta['seconds'] * FPS), 'labels': state['labels'],
                            'known_labels': sorted(v for v in known_labels if v and v != '?'),
                            'pieces': pieces, 'can_undo': state.get('undo_revision') is not None,
                            'sync_s': float(sync.get('cam1_to_cam2_s', 0)),
                            'sync_status': 'accepted' if local.get('status') == 'accepted' else 'fallback',
                            'sampled_frames': samples, 'stride': meta.get('stride', 3)})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return failure(exc)

    @app.get('/api/video/<clip>/image')
    def video_image(clip):
        try:
            roi = request.args.get('roi')
            if roi is not None:
                roi = [int(v) for v in roi.split(',')]
            path, identity = display_image(clip_folder(clip), request.args.get('cam'), request.args.get('frame', 0),
                                           preview=request.args.get('preview') == '1', roi=roi)
            response = send_file(path, conditional=True)
            response.headers['X-Raw-Frame'] = str(identity['clip_frame'])
            response.headers['X-Source-Frame'] = str(identity['frame_index'])
            response.headers['Cache-Control'] = 'private, max-age=3600'
            return response
        except (ValueError, TypeError, OSError) as exc:
            return failure(exc)

    @app.get('/api/video/<clip>/frame')
    def video_frame(clip):
        from video_annotations import frame_data
        try:
            folder = clip_folder(clip)
            _, frame = validate_frame(folder, request.args.get('cam'), request.args.get('frame', 0))
            return jsonify(frame_data(folder, request.args.get('cam'), frame))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return failure(exc)

    @app.get('/api/video/<clip>/window')
    def video_proxy(clip):
        try:
            result = video_window(clip_folder(clip), request.args.get('cam'), request.args.get('frame', 0))
            return jsonify({**result, 'url': '/video-media/%s/%s' % (clip, result['name'])})
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.get('/video-media/<clip>/<name>')
    def video_proxy_file(clip, name):
        import re
        if not re.fullmatch(r'[0-9a-f]{24}\.mp4', name):
            abort(404)
        return send_file(clip_folder(clip) / 'review_media' / name, conditional=True)

    @app.post('/api/video/<clip>/edit')
    def video_edit(clip):
        from video_annotations import video_transact
        try:
            body = body_with_revision()
            state = video_transact(clip_folder(clip), body, body['revision'])
            return jsonify({'ok': True, 'revision': state['revision'], 'can_undo': True})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return failure(exc)

    @app.post('/api/video/<clip>/sam')
    def video_sam_enqueue(clip):
        from sam_service import enqueue
        try:
            folder = clip_folder(clip)
            body = request.get_json(force=True)
            state = read_state(folder)
            supplied = body.get('review_revision', body.get('revision'))
            if supplied is not None and int(supplied) != state['revision']:
                raise Conflict('Разметка изменилась; обновите кадр перед SAM')
            body['review_revision'] = state['revision']
            return jsonify(enqueue(folder, body)), 202
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.get('/api/video/<clip>/sam/<job_id>')
    def video_sam_status(clip, job_id):
        from sam_service import get_job
        try:
            return jsonify(get_job(clip_folder(clip), job_id))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return failure(exc)

    @app.post('/api/video/<clip>/sam/<job_id>/accept')
    def video_sam_accept(clip, job_id):
        try:
            return jsonify(accept_sam_proposals(clip_folder(clip), job_id, body_with_revision()))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return failure(exc)

    @app.get('/api/video/<clip>/export')
    def video_export(clip):
        from product_export import compressed_export, export_clip
        try:
            path = export_clip(clip_folder(clip))
            if request.args.get('compress') == 'gzip':
                return send_file(compressed_export(path), as_attachment=True,
                                 download_name=clip+'-segmentation-tracks.ndjson.gz', mimetype='application/gzip')
            return send_file(path, as_attachment=True, download_name=clip+'-segmentation-tracks.ndjson', mimetype='application/x-ndjson')
        except (ValueError, KeyError, TypeError, OSError, RuntimeError) as exc:
            return failure(exc)

    @app.post('/api/video/<clip>/sam/<job_id>/cancel')
    def video_cancel_sam(clip, job_id):
        from sam_service import cancel_job
        try:
            return jsonify(cancel_job(clip_folder(clip), job_id))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            return failure(exc)


def accept_sam_proposals(folder, job_id, body):
    from sam_service import get_job, job_path, record_acceptance
    from video_annotations import effective_decision, frame_data, video_transact
    from storage import file_lock
    with file_lock(job_path(folder, job_id) / 'accept.lock'):
        job = get_job(folder, job_id)
        if job.get('status') != 'ready':
            raise ValueError('SAM ещё не подготовил предложения')
        expected = job.get('accepted_revision', job['review_revision'])
        if int(expected) != int(body['revision']):
            raise Conflict('После запуска SAM разметка изменилась; запустите предложение заново')
        state = read_state(folder)
        if state['revision'] != int(body['revision']):
            raise Conflict('Разметка изменилась в другой вкладке; обновите запись')
        wanted = body.get('frames', [])
        if not isinstance(wanted, list) or not wanted or len(wanted) > 250 or any(type(f) is not int for f in wanted):
            raise ValueError('Выберите просмотренные исходные кадры для сохранения')
        if body.get('confirm_masks') is not True:
            raise ValueError('Подтвердите просмотр и качество выбранных контуров')
        proposals = {int(p['frame']): p for p in job['frames']}
        accepted = set(job.get('accepted_frames', []))
        explicit_ids = body.get('observation_ids', {})
        if not isinstance(explicit_ids, dict):
            raise ValueError('Неверные идентификаторы наблюдений')
        piece = None
        piece_frames = {}
        if body.get('piece') is not None:
            piece = next((p for p in state['pieces'] if p['piece'] == body['piece'] and p['cam'] == job['request']['cam']), None)
            if piece is None:
                raise ValueError('Выбранный отрезок не принадлежит этой камере')
            with np.load(Path(folder) / 'dets_yolo26x-seg.npz') as z:
                for index in piece['dets']:
                    f = int(round(float(z[piece['cam']][index, 0]) * FPS))
                    piece_frames.setdefault(f, []).append('det:%s:%s' % (piece['cam'], index))
            for oid, value in state.get('video', {}).get('observations', {}).items():
                if oid.startswith('manual:') and value.get('piece') == piece['piece']:
                    piece_frames.setdefault(value['frame'], []).append(oid)
        anchor_label = None
        if body.get('anchor_id'):
            anchor_data = frame_data(folder, job['request']['cam'], job['request']['frame'], state=state)
            anchor_label = next((o.get('label') for o in anchor_data['observations'] if o['id'] == body['anchor_id']), None)
        ops, indices, protected = [], [], []
        for frame in sorted(set(wanted)):
            if frame not in proposals:
                raise ValueError('Кадр отсутствует среди предложений SAM')
            p = proposals[frame]
            if p.get('visible') is False or not p.get('box'):
                raise ValueError('SAM не видит человека в выбранном кадре; пустую маску нельзя подтвердить')
            if frame in accepted:
                continue
            current = frame_data(folder, p['cam'], frame, state=state)
            oid = explicit_ids.get(str(frame))
            if not oid and frame == job['request']['frame'] and body.get('anchor_id'):
                oid = body['anchor_id']
            if not oid and piece is not None:
                candidates = piece_frames.get(frame, [])
                if len(candidates) > 1:
                    raise ValueError('В кадре несколько наблюдений выбранного отрезка; исправьте их отдельно')
                if candidates:
                    oid = candidates[0]
            existing = next((o for o in current['observations'] if o['id'] == oid), None)
            effective_quality = effective_decision(state, piece['piece'] if piece else None, frame, oid)[1]
            if state.get('video', {}).get('observations', {}).get(oid, {}).get('deleted'):
                protected.append(frame)
                continue
            if oid and existing is None:
                raise ValueError('Наблюдение отсутствует в выбранном кадре')
            if ((existing and existing.get('quality', 'unreviewed') != 'unreviewed') or
                    (piece is not None and effective_quality != 'unreviewed')) and oid != body.get('anchor_id'):
                protected.append(frame)
                continue
            if not oid and any(
                min(o['box'][2],p['box'][2]) > max(o['box'][0],p['box'][0]) and
                min(o['box'][3],p['box'][3]) > max(o['box'][1],p['box'][1])
                for o in current['observations'] if o.get('box') and o.get('quality') != 'false_positive'
            ):
                raise ValueError('На кадре %s маска пересекает другого человека; проверьте этот кадр отдельно, чтобы не создать дубль' % frame)
            op = {'action': 'upsert_observation', 'cam': p['cam'], 'frame': frame,
                  'box': p['box'], 'quality': 'valid',
                  'source': 'sam_propagation', 'approval': 'accepted_propagation',
                  'provenance': {'job_id': job_id, 'model': job['request']['model'],
                                 'input_revision': job['review_revision']}}
            # This action accepts contours. Existing identities, including later
            # human interval corrections, are never reassigned by a SAM accept.
            if not oid and piece is not None:
                op['label'] = effective_decision(state, piece['piece'], frame)[0]
            elif not oid:
                chosen_label = job.get('accepted_label') or anchor_label or body.get('label')
                if chosen_label is not None:
                    op['label'] = chosen_label
            for key in ('polygon', 'mask_rle'):
                if p.get(key):
                    op[key] = p[key]
            if oid:
                op['id'] = oid
            if piece is not None:
                op['piece'] = piece['piece']
            if body.get('anchor_id'):
                op['provenance']['anchor_id'] = body['anchor_id']
            ops.append(op)
            indices.append(frame)
        if ops:
            state = video_transact(folder, {'action':'batch', 'operations':ops}, body['revision'])
            last = state.get('video', {}).get('observations', {}).get(state.get('_last_observation_id'), {})
            record_acceptance(folder, job_id, state['revision'], indices, accepted_label=last.get('label'))
        remaining = sorted(set(proposals) - accepted - set(indices) - set(protected))
        return {'ok': True, 'revision': state['revision'], 'accepted_frames':len(indices),
                'accepted_frame_indices': indices, 'protected_frames':protected,
                'remaining_frames':remaining}
