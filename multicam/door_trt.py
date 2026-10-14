"""DoorSeg TensorRT 11 strongly typed backend; no speed measurements.

Export can run on CPU. Engine build/inference must pass the existing night guard.
The engine is cached against the exact ONNX bytes, TensorRT version and GPU.
"""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/demo'


def gpu_guard():
    from door_seg_night_probe import blockers
    reasons = blockers()
    if os.environ.get('RA_GPU_OWNER_OK') == '1':      # 13.10 owner: the live door stopped for this test, day or not
        reasons = [r for r in reasons if r != 'daytime' and not r.startswith('interactive session') and not r.startswith('sampling foreign')]
    if not reasons:
        import psutil
        own={os.getpid(),*(p.pid for p in psutil.Process().parents())}
        for p in psutil.process_iter(['pid','name']):
            if p.pid in own or 'python' not in (p.info['name'] or '').lower():continue
            try:cmd=' '.join(p.cmdline())
            except psutil.NoSuchProcess:continue
            if ('door_live_bench.py' in cmd and '--equivalence-only' in cmd) or ('door_trt_check.py' in cmd and '--gpu' in cmd):
                reasons.append('foreign TRT GPU check PID '+str(p.pid))
    if reasons:
        raise RuntimeError('TRT GPU unavailable: ' + ', '.join(reasons))


def cpu_guard():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    import torch
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU export must start in a fresh CPU-only process')
    def forbidden(*a, **k):
        raise RuntimeError('CPU export forbids CUDA initialization')
    torch.cuda._lazy_init = forbidden
    torch._C._cuda_init = forbidden
    torch.set_num_threads(2)


def wrap(net):
    import torch
    class Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.yolo, self.off = net.yolo, net.off
        def forward(self, x):
            o = self.yolo(x)
            return o[0][0], o[0][1], self.off(net.feat)
    return Wrap().eval()


def export(ckpt, precision='fp16'):
    """Keep ONNX types explicit: TRT11 has no FP16 builder flag."""
    cpu_guard()
    import torch
    import onnx
    import door_seg as S
    OUT.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    S.checkpoint_finite(state)
    net = S.DoorSeg('cpu'); net.load(state['model'])
    net.yolo.eval(); net.off.eval(); net.yolo.fuse(verbose=False)
    model = wrap(net)
    if precision == 'fp16':
        model.half()
    elif precision != 'fp32':
        raise ValueError(precision)
    x = torch.zeros(1, S.C, S.HP, S.W, dtype=next(model.parameters()).dtype)
    path = OUT / ('doorseg_c.onnx' if precision == 'fp16' else 'doorseg_c_fp32.onnx')
    import shutil
    backup=ROOT/'data/quality_backups/20261013_trt/exports';backup.mkdir(parents=True,exist_ok=True)
    for old in (path,path.with_suffix('.json')):
        if old.exists():
            dest=backup/(old.stem+'_'+hashlib.sha256(old.read_bytes()).hexdigest()[:12]+old.suffix)
            if not dest.exists():shutil.copy2(old,dest)
    tmp = path.with_suffix('.tmp.onnx')
    if tmp.exists():tmp.unlink()
    with torch.inference_mode():
        torch.onnx.export(model, x, str(tmp), input_names=['x'], output_names=['det', 'proto', 'off'],
                          opset_version=17, dynamo=False)
    onnx.checker.check_model(str(tmp))
    tmp.replace(path)
    meta = dict(checkpoint_sha256=hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
                onnx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), precision=precision,
                input_shape=list(x.shape), outputs=['det', 'proto', 'off'], cuda_initialized=torch.cuda.is_initialized())
    path.with_suffix('.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print(json.dumps(meta), flush=True)
    return meta


class Engine:
    def __init__(self, onnx=None, engine=None):
        gpu_guard()  # before importing TensorRT or opening CUDA
        import torch
        import tensorrt as trt
        self.torch, self.trt = torch, trt
        onnx = Path(onnx or OUT / 'doorseg_c.onnx')
        engine = Path(engine or OUT / 'doorseg_c_fp16.engine')
        if not onnx.is_file():
            raise FileNotFoundError('Run CPU export first: ' + str(onnx))
        meta_path = engine.with_suffix('.json')
        expected = dict(onnx_sha256=hashlib.sha256(onnx.read_bytes()).hexdigest(),
                        tensorrt=trt.__version__, gpu=torch.cuda.get_device_name())
        self.log = trt.Logger(trt.Logger.WARNING)
        if engine.exists():
            if not meta_path.exists() or json.loads(meta_path.read_text(encoding='utf-8')) != expected:
                raise RuntimeError('Stale/unverified engine; archive it before explicit rebuild')
            self.built = False
        else:
            builder = trt.Builder(self.log)
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            network = builder.create_network(flags)
            parser = trt.OnnxParser(network, self.log)
            if not parser.parse_from_file(str(onnx)):
                raise RuntimeError('; '.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            blob = builder.build_serialized_network(network, config)
            if blob is None:
                raise RuntimeError('TensorRT engine build returned None')
            gpu_guard()
            tmp = engine.with_suffix('.tmp.engine'); tmp.write_bytes(bytes(blob)); tmp.replace(engine)
            meta_path.write_text(json.dumps(expected, indent=2), encoding='utf-8')
            self.built = True
        self.runtime = trt.Runtime(self.log)
        self.engine = self.runtime.deserialize_cuda_engine(engine.read_bytes())
        if self.engine is None:
            raise RuntimeError('TensorRT engine deserialization failed')
        self.context = self.engine.create_execution_context()
        types = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64, trt.bool: torch.bool}
        self.input_dtype = types[self.engine.get_tensor_dtype('x')]
        self.outputs = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT:
                self.outputs[n] = torch.empty(tuple(self.engine.get_tensor_shape(n)),
                                              dtype=types[self.engine.get_tensor_dtype(n)], device='cuda')
        if set(self.outputs) != {'det', 'proto', 'off'}:
            raise RuntimeError('Unexpected engine outputs: ' + str(set(self.outputs)))
        self.calls = 0

    def __call__(self, x):
        if self.calls % 16 == 0:
            gpu_guard()
        self.calls += 1
        torch = self.torch
        x = x.to(dtype=self.input_dtype).contiguous()
        if tuple(x.shape) != tuple(self.engine.get_tensor_shape('x')):
            raise ValueError('Engine requires fixed shape ' + str(self.engine.get_tensor_shape('x')))
        stream = torch.cuda.current_stream(x.device)
        self.context.set_tensor_address('x', x.data_ptr())
        for n, t in self.outputs.items():
            self.context.set_tensor_address(n, t.data_ptr())
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError('TensorRT execute_async_v3 failed')
        # Keep the temporary half input alive until asynchronous execution finishes.
        x.record_stream(stream)
        return tuple(self.outputs[n] for n in ('det', 'proto', 'off'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(); p.add_argument('checkpoint'); p.add_argument('--precision', choices=['fp16', 'fp32'], default='fp16')
    a = p.parse_args(); export(a.checkpoint, a.precision)
