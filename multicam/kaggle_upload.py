"""The encrypted full kit (kit_build.py full) as a private Kaggle dataset, so a Kaggle notebook attaches it instead of
downloading 20 GB through the tunnel every session. The owner's token comes in the environment (KAGGLE_API_TOKEN) of
this process only; it is never written or printed.

usage: kaggle_upload.py [SLUG]        (default cctv-kit-v2; a new version when the dataset exists)"""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
KIT = ROOT / 'data' / 'kit_full'


def parallel(workers=1):
    """The tool sends the files one after another; one stream from the shop to Google gets ~0.25-0.7 MB/s (30.09,
    measured; a bigger send buffer did not help: the long path loses packets), so the volumes (1.4 GB each) go side by side. A file that fails stops everything: no dataset with a volume missing."""
    import concurrent.futures as cf
    from kaggle.api import kaggle_api_extended as K

    def upload_files(self, request, resources, folder, blob_type, upload_context, quiet=False, dir_mode='skip', ignore_patterns=None):
        patterns = K.DEFAULT_IGNORE_PATTERNS + (ignore_patterns or [])
        skip = [self.DATASET_METADATA_FILE, self.OLD_DATASET_METADATA_FILE, *self.DATASET_COVER_IMAGE_FILES,
                self.KERNEL_METADATA_FILE, self.MODEL_METADATA_FILE, self.MODEL_INSTANCE_METADATA_FILE]
        names = [n for n in sorted(os.listdir(folder)) if n not in skip and not K.should_ignore(n, os.path.isdir(os.path.join(folder, n)), patterns)]
        with cf.ThreadPoolExecutor(workers) as ex:
            outs = list(ex.map(lambda n: self._upload_file_or_folder(folder, n, blob_type, upload_context, dir_mode, quiet, resources, patterns), names))
        bad = [n for n, u in zip(names, outs) if u is None]
        if bad:
            raise RuntimeError('not uploaded: %s' % bad)
        for u in outs:
            if request.files is not None:
                request.files.append(self._new_file(u))
    K.KaggleApi.upload_files = upload_files


def big_send_buffer(mb=16):
    """Windows gives a socket 64 KB to send from: at ~0.23 s to Google that is ~0.28 MB/s a stream, exactly what the
    upload got (30.09). A bigger buffer lets TCP keep more in flight."""
    import socket
    import urllib3.connection
    opts = [o for o in urllib3.connection.HTTPConnection.default_socket_options if o[1] != socket.SO_SNDBUF]
    urllib3.connection.HTTPConnection.default_socket_options = opts + [(socket.SOL_SOCKET, socket.SO_SNDBUF, mb << 20)]


def timeouts(connect=30, read=180):
    """The tool sends a volume with requests and no timeout: after a reset the socket can hang forever (30.09: every
    stream stood still from 03:38 to 04:45, the process alive, nothing sent). With a timeout the stream fails, and the
    tool's resumable upload goes on from where Google has it."""
    import requests
    orig = requests.Session.request

    def request(self, method, url, **kw):
        kw.setdefault('timeout', (connect, read))
        return orig(self, method, url, **kw)
    requests.Session.request = request


def main():
    slug = sys.argv[1] if len(sys.argv) > 1 else 'cctv-kit-v2'
    timeouts()
    big_send_buffer()
    parallel()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    user = (getattr(api, 'config_values', {}) or {}).get('username') or api.get_config_value('username')
    if not user:
        raise SystemExit('the token did not give a user name')
    print('kaggle user:', user, flush=True)
    up = KIT / 'upload'
    up.mkdir(exist_ok=True)
    for old in up.iterdir():
        old.unlink()
    for f in sorted(KIT.glob('kit_full.7z.*')) + [KIT / 'kit_full.json']:
        os.link(f, up / f.name)                                   # no copy
    ref = '%s/%s' % (user, slug)
    json.dump({'title': 'CCTV shop kit v2', 'id': ref, 'licenses': [{'name': 'other'}], 'isPrivate': True},
              open(up / 'dataset-metadata.json', 'w'), indent=1)
    size = sum(p.stat().st_size for p in up.iterdir())
    print('uploading %s: %d files, %.2f GB, private' % (ref, len(list(up.iterdir())), size / 1e9), flush=True)
    t0 = time.time()
    exists = False
    try:
        exists = any(d.ref == ref for d in api.dataset_list(mine=True, search=slug))
    except Exception:
        pass
    if exists:
        r = api.dataset_create_version(str(up), 'kit ' + time.strftime('%Y-%m-%d %H:%M'), quiet=False, dir_mode='skip')
    else:
        r = api.dataset_create_new(str(up), public=False, quiet=False, convert_to_csv=False, dir_mode='skip')
    print('result:', getattr(r, 'status', r), getattr(r, 'url', ''), getattr(r, 'error', ''), flush=True)
    print('done in %.0f min (%.1f MB/s)' % ((time.time() - t0) / 60, size / 1e6 / max(1, time.time() - t0)), flush=True)


if __name__ == '__main__':
    main()
