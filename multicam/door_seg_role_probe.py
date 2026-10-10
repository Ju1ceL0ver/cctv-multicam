"""Finite CPU-only role service/profile on new TRAIN queue-probe view banks.

Starts its own worker, never connects to or stops the production role service.
No training, feedback, owner-label writes, CRM output or GPU requests.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent
TAGS = ('door_20260918_01405', 'door_20260918_15375')


def run(source, out):
    from door_role import Bank, Roles, RF
    available = psutil.virtual_memory().available
    if available < 8*2**30:
        raise RuntimeError('defer CPU teachers: less than 8GiB available')
    out.mkdir(parents=True, exist_ok=False)
    bank = Bank(out/'unused_bank')
    selected, hashes = [], {}
    owner_state = json.loads((ROOT/'data/door_side/20260918.json').read_text(encoding='utf-8'))
    for tag in TAGS:
        if not owner_state[tag].get('done') or owner_state[tag].get('role') != 'train':
            raise ValueError('completed owner TRAIN only: '+tag)
        folder = source/tag/'views'
        rows = json.loads((folder/'views.json').read_text(encoding='utf-8'))
        # Four best stored views from one person per scene: eight total.
        key = max(rows, key=lambda k: (len(rows[k]), sum(x[0] for x in rows[k])))
        views = rows[key][:4]
        name = tag+':'+key
        bank.views[name] = [(float(s), str(folder/p), b) for s,p,b in views]
        for _,p,_ in bank.views[name]:
            hashes[str(Path(p).relative_to(source))] = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        selected.append(name)
    report = {'complete': False, 'device': 'cpu', 'threads': 8, 'source': str(source),
              'input_sha256': hashes, 'free_ram_bytes': available, 'groups': {},
              'limitations': 'TRAIN service timing only; no independent role accuracy, live queue backlog or GPU speed proof.'}
    model_path = ROOT/'data/staff/current_role.pkl'
    report['role_model_sha256_before'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    lines = queue.Queue()
    start = time.perf_counter()
    worker = subprocess.Popen([RF,str(ROOT/'door_role.py'),'serve'], cwd=ROOT, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=(out/'worker.err').open('w',encoding='utf-8'), text=True,encoding='utf-8',
        env=dict(os.environ,RA_ROLE_DEVICE='cpu',RA_ROLE_THREADS='8',PYTHONIOENCODING='utf-8',
                 HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1'),
        creationflags=getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    report['own_worker_pid'] = worker.pid

    def reader():
        for line in worker.stdout:
            if line.startswith('{'):
                try:
                    lines.put(json.loads(line))
                except ValueError:
                    pass
        lines.put({'error':'owned worker closed stdout'})

    threading.Thread(target=reader,daemon=True).start()
    peak = 0
    def response(timeout):
        nonlocal peak
        result = lines.get(timeout=timeout)
        processes = [psutil.Process(worker.pid)]
        processes += processes[0].children(recursive=True)
        peak = max(peak,sum(p.memory_info().rss for p in processes if p.is_running()))
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    class Client:
        def __init__(self, threshold):
            self.threshold = threshold
            self.calls = 0
        def probs(self, paths, boxes, vectors=False):
            self.calls += 1
            worker.stdin.write(json.dumps({'crops':paths,'boxes':boxes,'vectors':vectors})+'\n')
            worker.stdin.flush()
            result = response(45)
            return (result['p'],result.get('v')) if vectors else result['p']

    try:
        ready = response(60)
        report['initialization_seconds'] = time.perf_counter()-start
        client = Client(float(ready['ready']))
        roles = Roles(bank,client)
        for key in selected:
            before = client.calls
            t = time.perf_counter(); cold = roles.role([key]); seconds = time.perf_counter()-t
            if cold.get('error') or cold.get('role') is None:
                raise RuntimeError(str(cold))
            t = time.perf_counter(); cached = roles.role([key]); cached_seconds = time.perf_counter()-t
            ps = [roles.cache[p] for _,p,_ in bank.best([key])]
            if not np.isfinite(ps).all():
                raise FloatingPointError('nonfinite role probabilities')
            report['groups'][key] = {'views':len(ps),'first_service_seconds':seconds,
                'cached_seconds':cached_seconds, 'worker_calls':client.calls-before,
                'first':cold,'cached':cached,'cached_role_equal':cold['role']==cached['role'],
                'p_staff_equal':cold['p_staff']==cached['p_staff']}
        report['worker_tree_observed_peak_rss_bytes'] = peak
        report['complete'] = True
    except Exception as exc:
        report['error'] = type(exc).__name__+': '+str(exc)
        raise
    finally:
        # First close the owned worker's input: its serve loop exits at EOF.
        try:
            own = psutil.Process(worker.pid)
            children = own.children(recursive=True) if worker.poll() is None else []
        except psutil.NoSuchProcess:
            own, children = None, []
        try:
            worker.stdin.close()
        except OSError:
            pass
        try:
            worker.wait(timeout=4)
        except subprocess.TimeoutExpired:
            for p in children+([own] if own is not None else []):
                try:
                    cmd = p.cmdline()
                    if str(ROOT/'door_role.py') in cmd and 'serve' in cmd:
                        p.terminate()
                except psutil.Error:
                    pass
            try:
                worker.wait(timeout=4)
            except subprocess.TimeoutExpired:
                report['cleanup_pending_pid'] = worker.pid
        report['elapsed_seconds'] = time.perf_counter()-start
        report['owned_worker_exit_code'] = worker.poll()
        report['role_model_sha256_after'] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        report['role_model_file_unchanged'] = report['role_model_sha256_before'] == report['role_model_sha256_after']
        (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args()
    run(a.source,a.out)
