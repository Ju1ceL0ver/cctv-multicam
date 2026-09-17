#!/usr/bin/env python3
"""Minimal Jupyter Server client over HTTP/WS (for a remote server behind cloudflared).

Config via env: JUP_URL, JUP_TOKEN  (optionally JUP_CF_ID / JUP_CF_SECRET for Cloudflare Access)

Usage:
  jctl.py ls [path]
  jctl.py cat <path>
  jctl.py put <remote_path> < local_file
  jctl.py get <remote_path> <local_path>
  jctl.py run '<python code>'
  jctl.py runfile <local_file>
  jctl.py sh '<shell command>'     # runs via ! in the kernel
  jctl.py kernels
  jctl.py reset                    # drop cached kernel
"""
import json
import os
import sys
import uuid
from urllib.parse import urlparse, quote

import socket

import requests
import websocket

# websocket-client tries the first getaddrinfo entry and raises instead of
# falling back; Cloudflare resolves to IPv6 first and there is no v6 route here.
_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda *a, **k: [
    r for r in _getaddrinfo(*a, **k) if r[0] == socket.AF_INET
] or _getaddrinfo(*a, **k)

URL = os.environ.get("JUP_URL", "").rstrip("/")
TOKEN = os.environ.get("JUP_TOKEN", "")
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".jctl_kernel")

if not URL or not TOKEN:
    sys.exit("set JUP_URL and JUP_TOKEN")

HEADERS = {"Authorization": f"token {TOKEN}"}
if os.environ.get("JUP_CF_ID"):
    HEADERS["CF-Access-Client-Id"] = os.environ["JUP_CF_ID"]
    HEADERS["CF-Access-Client-Secret"] = os.environ["JUP_CF_SECRET"]


def api(method, path, **kw):
    r = requests.request(method, f"{URL}{path}", headers=HEADERS, timeout=60, **kw)
    r.raise_for_status()
    return r.json() if r.content else None


def get_kernel():
    if os.path.exists(STATE):
        kid = open(STATE).read().strip()
        try:
            api("GET", f"/api/kernels/{kid}")
            return kid
        except Exception:
            pass
    kid = api("POST", "/api/kernels", json={"name": "python3"})["id"]
    open(STATE, "w").write(kid)
    return kid


def execute(code):
    kid = get_kernel()
    p = urlparse(URL)
    scheme = "wss" if p.scheme == "https" else "ws"
    session = uuid.uuid4().hex
    ws = websocket.create_connection(
        f"{scheme}://{p.netloc}{p.path}/api/kernels/{kid}/channels?session_id={session}",
        header=[f"{k}: {v}" for k, v in HEADERS.items()],
        origin=URL,
        timeout=300,
    )
    msg_id = uuid.uuid4().hex
    ws.send(json.dumps({
        "header": {"msg_id": msg_id, "username": "jctl", "session": session,
                   "msg_type": "execute_request", "version": "5.3"},
        "parent_header": {}, "metadata": {},
        "content": {"code": code, "silent": False, "store_history": True,
                    "user_expressions": {}, "allow_stdin": False, "stop_on_error": True},
        "channel": "shell",
    }))

    status = 0
    try:
        while True:
            msg = json.loads(ws.recv())
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            t = msg["header"]["msg_type"]
            c = msg["content"]
            if t == "stream":
                (sys.stderr if c["name"] == "stderr" else sys.stdout).write(c["text"])
            elif t in ("execute_result", "display_data"):
                print(c["data"].get("text/plain", ""))
            elif t == "error":
                print("\n".join(c["traceback"]), file=sys.stderr)
                status = 1
            elif t == "status" and c["execution_state"] == "idle":
                break
    finally:
        ws.close()
    return status


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    cmd, args = sys.argv[1], sys.argv[2:]

    if cmd == "ls":
        path = args[0] if args else ""
        for it in api("GET", f"/api/contents/{quote(path)}")["content"]:
            print(f"{it['type']:9} {it['size'] or 0:>10}  {it['path']}")
    elif cmd == "cat":
        m = api("GET", f"/api/contents/{quote(args[0])}?content=1")
        print(m["content"] if m["format"] == "text" else json.dumps(m["content"], indent=1, ensure_ascii=False))
    elif cmd == "get":
        remote, local = args[0], args[1]
        meta = api("GET", f"/api/contents/{quote(remote)}?content=1")
        import base64

        if meta["format"] == "base64":
            data = base64.b64decode(meta["content"])
        else:
            data = meta["content"].encode()
        with open(local, "wb") as handle:
            handle.write(data)
        print(f"{local} ({len(data)} bytes)")
    elif cmd == "put":
        api("PUT", f"/api/contents/{quote(args[0])}",
            json={"type": "file", "format": "text", "content": sys.stdin.read()})
        print(f"wrote {args[0]}")
    elif cmd == "run":
        sys.exit(execute(args[0]))
    elif cmd == "runfile":
        sys.exit(execute(open(args[0]).read()))
    elif cmd == "sh":
        sys.exit(execute("get_ipython().system(%r)" % args[0]))
    elif cmd == "kernels":
        print(json.dumps(api("GET", "/api/kernels"), indent=1))
    elif cmd == "reset":
        if os.path.exists(STATE):
            try:
                api("DELETE", f"/api/kernels/{open(STATE).read().strip()}")
            except Exception:
                pass
            os.remove(STATE)
        print("ok")
    else:
        sys.exit(__doc__)


main()
