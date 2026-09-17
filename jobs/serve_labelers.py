"""Serve both labelling tools behind a shared key, then publish them via
Cloudflare quick tunnels so they can be opened from any device.

The tools themselves have no authentication -- they were written for
localhost -- and every frame they serve shows the shop's customers. A
trycloudflare URL is unlisted, not private, so each request must carry the
key: the first visit uses ``?key=...``, which is swapped for an HttpOnly
cookie and stripped from the address bar, so the key does not linger in
history or get copied along with a screenshot of the URL.

The existing cloudflared that fronts Jupyter is left alone -- killing it would
cut off remote access to this machine.
"""
import os, re, sys, time, secrets, threading, subprocess, importlib.util

HOME = r'C:\Users\ArykovAA'
ROOT = os.path.join(HOME, 'cctv_ai', 'retail_analytics')
KEY_FILE = os.path.join(HOME, '_labelers_key.txt')
URLS_FILE = os.path.join(HOME, '_labelers_urls.txt')
LOG = os.path.join(HOME, '_serve_labelers.log')
CLOUDFLARED = r'C:\Program Files (x86)\cloudflared\cloudflared.exe'


def log(*a):
    with open(LOG, 'a', encoding='utf-8') as f:
        f.write('%s %s\n' % (time.strftime('%H:%M:%S'), ' '.join(str(x) for x in a)))


def key():
    if os.path.exists(KEY_FILE):
        return open(KEY_FILE).read().strip()
    k = secrets.token_urlsafe(18)
    open(KEY_FILE, 'w').write(k)
    return k


KEY = key()


def protect(app):
    from flask import request, redirect, make_response

    @app.before_request
    def _gate():
        supplied = request.args.get('key')
        if supplied is not None and secrets.compare_digest(supplied, KEY):
            args = {k: v for k, v in request.args.items() if k != 'key'}
            target = request.path + (('?' + '&'.join('%s=%s' % kv for kv in args.items())) if args else '')
            resp = make_response(redirect(target))
            resp.set_cookie('labeler_key', KEY, max_age=30 * 24 * 3600, httponly=True, samesite='Lax')
            return resp
        cookie = request.cookies.get('labeler_key', '')
        if cookie and secrets.compare_digest(cookie, KEY):
            return None
        return ('<meta charset="utf-8"><body style="font:16px system-ui;padding:40px">'
                'Нужен ключ доступа: откройте ссылку целиком, вместе с <code>?key=…</code></body>', 401)
    return app


def build_apps():
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    from retail_analytics.annotate.web import create_app
    annot = protect(create_app(__import__('pathlib').Path(ROOT) / 'data' / 'annotate_val'))
    spec = importlib.util.spec_from_file_location('label_pairs', os.path.join(HOME, '_label_pairs.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    pairs = protect(mod.app)
    return annot, pairs


def serve(app, port):
    from werkzeug.serving import run_simple
    run_simple('127.0.0.1', port, app, threaded=True, use_reloader=False)


def tunnel(port, name):
    logf = os.path.join(HOME, '_cf_%s.log' % name)
    open(logf, 'w').close()
    subprocess.Popen([CLOUDFLARED, 'tunnel', '--no-autoupdate', '--url', 'http://127.0.0.1:%d' % port],
                     stdout=open(logf, 'a'), stderr=subprocess.STDOUT,
                     creationflags=0x00000200 | 0x00000008 | 0x01000000)
    for _ in range(90):
        time.sleep(1)
        m = re.search(r'https://[a-z0-9-]+\.trycloudflare\.com', open(logf, errors='replace').read())
        if m:
            return m.group(0)
    return None


def main():
    open(LOG, 'w').close()
    annot, pairs = build_apps()
    threading.Thread(target=serve, args=(annot, 5050), daemon=True).start()
    threading.Thread(target=serve, args=(pairs, 5060), daemon=True).start()
    time.sleep(3)
    u1 = tunnel(5050, 'annotate')
    u2 = tunnel(5060, 'pairs')
    lines = ['annotate %s/?key=%s' % (u1, KEY), 'pairs    %s/?key=%s' % (u2, KEY)]
    open(URLS_FILE, 'w').write('\n'.join(lines) + '\n')
    log(*lines)
    while True:
        time.sleep(3600)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        log('FATAL\n' + traceback.format_exc())
