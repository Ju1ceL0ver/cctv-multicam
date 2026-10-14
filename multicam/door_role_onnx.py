"""CPU ORT teachers for door_role; opt in with RA_ROLE_BACKEND=onnx.

Preprocessing and L2 normalisation match track_emb.teachers('cpu'). The
existing staff classifier and accumulation/gallery remain unchanged.
"""
import json
import hashlib
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/demo/role_onnx'


def teachers(require_provenance=False):
    import cv2
    import onnxruntime as ort
    opt = ort.SessionOptions()
    import os
    opt.intra_op_num_threads = int(os.environ.get('RA_ROLE_THREADS', '2'))
    opt.inter_op_num_threads = 1
    if require_provenance:
        for name in ('transreid','csci'):
            path=OUT/(name+'.onnx');meta=json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
            if meta['onnx_sha256']!=hashlib.sha256(path.read_bytes()).hexdigest():raise RuntimeError('Role ONNX hash mismatch: '+name)
    sessions = [ort.InferenceSession(str(OUT / (n + '.onnx')), sess_options=opt,
                                    providers=['CPUExecutionProvider']) for n in ('transreid', 'csci')]
    mean = np.asarray([.48145466, .4578275, .40821073], np.float32)[None, :, None, None]
    std = np.asarray([.26862954, .26130258, .27577711], np.float32)[None, :, None, None]
    def embed(crops):
        v1, v2 = [], []
        for c in crops:
            rgb = c[:, :, ::-1]
            a = np.ascontiguousarray(cv2.resize(rgb, (128, 256)).transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
            b = np.ascontiguousarray(cv2.resize(rgb, (224, 224)).transpose(2, 0, 1)[None], dtype=np.float32) / np.float32(255)
            b = (b - mean) / std
            v1.append(sessions[0].run(['v'], {'x': a})[0][0])
            v2.append(sessions[1].run(['v'], {'x': b})[0][0])
        normalise = lambda v: np.asarray(v, np.float32) / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), np.float32(1e-12))
        return normalise(v1), normalise(v2)
    return embed


def export_and_check():
    from door_trt import cpu_guard
    cpu_guard()
    import torch
    import cv2
    import onnx
    import track_emb as T
    import staff_model
    OUT.mkdir(parents=True, exist_ok=True)
    eager = T.teachers('cpu')
    tr, cs = eager.models
    class Trans(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.model = tr
        def forward(self, x):
            return self.model(x, cam_label=torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
    for name, model, shape in [('transreid', Trans().eval(), (1, 3, 256, 128)), ('csci', cs, (1, 3, 224, 224))]:
        path = OUT / (name + '.onnx')
        h=hashlib.sha256()
        for key,value in model.state_dict().items():
            h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        source_sha=h.hexdigest()
        metadata=path.with_suffix('.json')
        if path.exists() and metadata.exists():
            old=json.loads(metadata.read_text(encoding='utf-8'))
            if old.get('source_state_sha256')!=source_sha or old.get('onnx_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
                raise RuntimeError('Stale role ONNX; archive before re-export: '+name)
        if path.exists() and not metadata.exists():
            # Artifacts from an interrupted first export are never trusted by existence alone.
            import shutil
            backup=ROOT/'data/quality_backups/20261013_trt/role_initial_export'
            backup.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,backup/path.name);path.unlink()
        if not path.exists():
            tmp = OUT / (name + '.tmp.onnx')
            with torch.inference_mode():
                torch.onnx.export(model, torch.zeros(shape), str(tmp), input_names=['x'], output_names=['v'],
                                  opset_version=17, dynamo=False)
            onnx.checker.check_model(str(tmp)); tmp.replace(path)
            metadata.write_text(json.dumps(dict(source_state_sha256=source_sha,onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),input_shape=list(shape)),indent=2),encoding='utf-8')
        print('exported', name, flush=True)
    ort = teachers()
    clf = staff_model.load_current(ROOT)
    folders = sorted((ROOT / 'data/demo').glob('bench_views_*'))
    paths = [p for folder in folders for p in sorted(folder.glob('*.jpg'))]
    if not paths:
        raise RuntimeError('No saved bench views for probability equivalence')
    records = []
    # Include every saved view, with exact saved boxes where available. Old
    # benchmarks omitted views.json: a fixed box verifies backend differences
    # with identical classifier geometry, but is marked explicitly in report.
    for folder in folders:
        boxes = {}
        index = folder / 'views.json'
        if index.exists():
            for views in json.loads(index.read_text(encoding='utf-8')).values():
                boxes.update({v[1]: v[2] for v in views})
        for path in sorted(folder.glob('*.jpg')):
            crop = cv2.imread(str(path))
            if crop is None:
                raise RuntimeError('Unreadable view ' + str(path))
            a, b = eager([crop]); c, d = ort([crop])
            box = boxes.get(path.name, [0., 0., 1280., 720.])
            if not all(np.isfinite(v).all() for v in (a,b,c,d)):
                raise FloatingPointError('Nonfinite role embedding: '+str(path))
            p = float(clf.predict_proba(a, b, [box], ['cam1'])[0, 1])
            q = float(clf.predict_proba(c, d, [box], ['cam1'])[0, 1])
            if not np.isfinite([p,q]).all():raise FloatingPointError('Nonfinite role probabilities')
            records.append(dict(path=str(path.relative_to(ROOT)), p_torch=p, p_onnx=q, error=abs(p-q),
                                cloth_max_abs=float(np.abs(a-c).max()), shape_max_abs=float(np.abs(b-d).max()),
                                box_source='views.json' if path.name in boxes else 'identical_fixed_geometry'))
            if len(records) % 20 == 0:
                print('views checked', len(records), flush=True)
    report = dict(complete=True, cuda_initialized=torch.cuda.is_initialized(), views=len(records),
                  max_probability_error=max(r['error'] for r in records), records=records,
                  geometry_note='Old benchmark JPEGs may lack original boxes; same fixed geometry used for both backends.')
    report['pass'] = all(np.isfinite(r['error']) and r['error']<=.01 for r in records)
    (OUT / 'check.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}), flush=True)
    if not report['pass']:
        raise RuntimeError('Role probability error exceeds .01')


def stamp_initial_export():
    """Finalize the witnessed first export only after its full saved-view check.

    This mode is for the first job started before provenance support was added.
    It never accepts an arbitrary pre-existing model in the normal exporter.
    """
    import time
    report_path=OUT/'check.json'
    end=time.monotonic()+3*3600
    while not report_path.exists():
        if time.monotonic()>end:raise TimeoutError('Initial role check did not finish')
        time.sleep(20)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if not report.get('pass') or not report.get('complete') or not report.get('records'):
        raise RuntimeError('Initial role check failed')
    for row in report['records']:
        if not np.isfinite([row[k] for k in ('p_torch','p_onnx','error','cloth_max_abs','shape_max_abs')]).all() or row['error']>.01:
            raise FloatingPointError('Initial role check has nonfinite/failed record')
    from door_trt import cpu_guard
    cpu_guard()
    import track_emb as T
    eager=T.teachers('cpu')
    provenance={}
    for name,model in zip(('transreid','csci'),eager.models):
        h=hashlib.sha256()
        for key,value in model.state_dict().items():
            key=('model.'+key) if name=='transreid' else key
            h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
        path=OUT/(name+'.onnx')
        meta=dict(source_state_sha256=h.hexdigest(),onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  input_shape=[1,3,256,128] if name=='transreid' else [1,3,224,224],
                  provenance='First export created in this task; full 778-view equivalence checked before stamping')
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');provenance[name]=meta
    report['provenance']=provenance;report['finite_checked']=True
    import shutil
    backup=ROOT/'data/quality_backups/20261013_trt/role_initial_check.json'
    if not backup.exists():shutil.copy2(report_path,backup)
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,views=report['views'],max_probability_error=report['max_probability_error'],finite_checked=True)),flush=True)


if __name__ == '__main__':
    import sys
    if '--stamp-initial-export' in sys.argv:stamp_initial_export()
    else:export_and_check()
