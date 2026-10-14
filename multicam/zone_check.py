"""13.10.2026: the disputed door moments (zone rules vs the owner's /doorside events) as GIFs for /zonecheck.
data/zonecheck/items.json (made on the Mac by zone_lab/make_items.py) -> data/zonecheck/<id>.gif; answers in
data/zonecheck/answers.json. register(app) adds /zonecheck and its API."""
import json, os, sys, time
from pathlib import Path
import numpy as np, cv2
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'data' / 'zonecheck'
W = 640


def _labels(f, texts):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(f[:, :, ::-1]); d = ImageDraw.Draw(im)
    def font(n):
        for name in ('arialbd.ttf', 'arial.ttf'):
            try:
                return ImageFont.truetype(name, n)
            except OSError:
                pass
        return ImageFont.load_default()
    big, small = font(20), font(14)
    for x, y, word, col, me, own in texts:
        fn = big if me else small
        tx = word if not me else 'система: ' + word
        y0 = max(0, y - (26 if me else 18))
        l, t, r, b = d.textbbox((x, y0), tx, font=fn)
        d.rectangle([l - 3, t - 2, r + 3, b + 2], fill=(0, 0, 0))
        d.text((x, y0), tx, fill=tuple(int(c) for c in col[::-1]), font=fn)
        if me and own in ('in', 'out'):
            t2 = 'ты: ' + ('ВНУТРИ' if own == 'in' else 'СНАРУЖИ')
            l, t, r, b = d.textbbox((x, y0 - 22), t2, font=small)
            d.rectangle([l - 3, t - 2, r + 3, b + 2], fill=(0, 0, 0))
            d.text((x, y0 - 22), t2, fill=(255, 255, 255), font=small)
    return np.asarray(im)[:, :, ::-1].copy()


def gif(item, zones):
    import door_side as S
    from PIL import Image
    ol = S.outlines(item['tag'])
    cap = cv2.VideoCapture(str(ROOT / 'data/sam31_door' / item['tag'] / 'cam1/video.mp4'))
    k0 = int(round(item['t'] * 12.5))
    a, b = max(0, k0 - 64), k0 + 64
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    hi = set(map(str, item['pieces']))
    lab = {}
    for k, piece, sysv, own in item.get('sides', []):
        lab[(int(k), str(piece))] = (sysv, own)
    last = {}
    sc = W / 1280.0
    frames = []
    for k in range(a, b + 1):
        ok, f = cap.read()
        if not ok:
            break
        if (k - a) % 2:
            continue
        f = cv2.resize(f, (W, 360), interpolation=cv2.INTER_AREA)
        for z in zones:
            p = (np.array(z['pts']) * sc).astype(np.int32)
            cv2.polylines(f, [p], z['kind'] == 'poly', (200, 200, 200), 1)
        kk = k - k % 3
        texts = []
        for piece, _, poly in ol.get(str(kk), []):
            p = (np.array(poly) * sc).astype(np.int32)
            if (kk, str(piece)) in lab:
                last[str(piece)] = lab[(kk, str(piece))]
            sysv, own = last.get(str(piece), (None, None))
            if item.get('nosys'):
                me = str(piece) in hi
                cv2.polylines(f, [p], True, (0, 230, 255) if me else (170, 170, 170), 3 if me else 1)
                continue
            col = (80, 200, 60) if sysv == 1 else (200, 60, 200) if sysv == 0 else (150, 150, 150)
            me = str(piece) in hi
            ov = f.copy(); cv2.fillPoly(ov, [p], col); f = cv2.addWeighted(ov, .45 if me else .25, f, .55 if me else .75, 0)
            cv2.polylines(f, [p], True, (0, 230, 255) if me else col, 3 if me else 1)
            word = 'ВНУТРИ' if sysv == 1 else 'СНАРУЖИ' if sysv == 0 else 'не решено'
            texts.append((int(p[:, 0].min()), int(p[:, 1].min()), word, col, me, own))
        if texts:
            f = _labels(f, texts)
        if abs(k - k0) <= 1:
            cv2.rectangle(f, (0, 0), (W - 1, 359), (40, 40, 230), 6)
        cv2.putText(f, '%+.1f s' % ((k - k0) / 12.5), (W - 90, 350), 0, .6, (255, 255, 255), 2)
        frames.append(f[:, :, ::-1])
    cap.release()
    if not frames:
        return False
    pal = Image.fromarray(frames[len(frames) // 2]).quantize(colors=96, method=Image.MEDIANCUT)
    ims = [Image.fromarray(x).quantize(palette=pal, dither=Image.Dither.NONE) for x in frames]
    tmp = OUT / (item['id'] + '.tmp.gif')
    ims[0].save(tmp, save_all=True, append_images=ims[1:], duration=160, loop=0, optimize=True)
    os.replace(tmp, OUT / (item['id'] + '.gif'))
    return True


def build():
    items = json.load(open(OUT / 'items.json', encoding='utf-8'))
    zones = json.load(open(ROOT / 'data/door_v2/door_lines_owner.json', encoding='utf-8'))['lines']
    for it in items:
        if not (OUT / (it['id'] + '.gif')).exists():
            print(it['id'], gif(it, zones), flush=True)
    print('done')


def register(app):
    from flask import jsonify, request, render_template, send_file

    @app.get('/livestream')
    def livestream_page():
        return ('<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>Дверь вживую</title><style>body{margin:0;background:#000;color:#ccc;font:14px system-ui}'
                'img{width:100%;max-width:1280px;display:block;margin:0 auto}p{text-align:center}</style></head><body>'
                '<img src="/api/livestream.mjpg" alt="поток"><p>Новая дверь: зелёный — внутри, фиолетовый — снаружи, серый — не решено. '
                'Отставание от камеры около секунды.</p></body></html>')

    @app.get('/api/livestream.mjpg')
    def livestream_mjpg():
        from flask import Response
        f = ROOT / 'data' / 'live' / 'stream' / 'latest.jpg'

        def gen():
            last = 0
            t_end = time.time() + 3600
            while time.time() < t_end:
                try:
                    m = f.stat().st_mtime
                    if m != last:
                        last = m
                        b = f.read_bytes()
                        yield b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(b)).encode() + b'\r\n\r\n' + b + b'\r\n'
                except OSError:
                    pass
                time.sleep(0.07)
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.get('/zonecheck')
    def zonecheck_page():
        return render_template('zonecheck.html')

    @app.get('/api/zonecheck')
    def zonecheck_list():
        items = json.load(open(OUT / 'items.json', encoding='utf-8')) if (OUT / 'items.json').exists() else []
        ans = json.load(open(OUT / 'answers.json', encoding='utf-8')) if (OUT / 'answers.json').exists() else {}
        return jsonify({'items': [dict(i, gif=(OUT / (i['id'] + '.gif')).exists()) for i in items], 'answers': ans})

    @app.get('/api/zonecheck/gif/<name>')
    def zonecheck_gif(name):
        p = OUT / (Path(name).stem + '.gif')
        if not p.exists():
            return ('', 404)
        return send_file(str(p), mimetype='image/gif', max_age=3600)

    @app.post('/api/zonecheck/answer')
    def zonecheck_answer():
        from storage import atomic_json
        b = request.get_json(force=True) or {}
        if str(b.get('v')) not in ('1', '2', '0', '3', 'clear'):
            return jsonify({'error': 'v'}), 400
        p = OUT / 'answers.json'
        ans = json.load(open(p, encoding='utf-8')) if p.exists() else {}
        if b['v'] == 'clear':
            ans.pop(str(b['id']), None)
        else:
            ans[str(b['id'])] = {'v': int(b['v']), 'at': time.strftime('%Y-%m-%dT%H:%M:%S')}
        atomic_json(p, ans)
        with open(OUT / 'answers.history.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(dict(b, at=time.strftime('%Y-%m-%dT%H:%M:%S')), ensure_ascii=False) + '\n')
        return jsonify(ans)


if __name__ == '__main__':
    build()
