"""CPU-only receiver for checkpoint chunks written by the Colab runner."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def restore(run):
    folder = run / 'colab_upload'
    folder.mkdir(exist_ok=True)
    for manifest in sorted(folder.glob('*.json')):
        meta = json.loads(manifest.read_text())
        name = meta['target']
        if Path(name).name != name or not name.endswith('.pt'):
            raise ValueError('Invalid checkpoint name')
        if any(Path(p).name != p for p in meta['parts']):
            raise ValueError('Invalid part name')
        parts = [folder / p for p in meta['parts']]
        if not all(p.is_file() for p in parts):
            continue
        temp = run / (name + '.colab.tmp')
        sha = hashlib.sha256()
        size = 0
        with temp.open('wb') as out:
            for part in parts:
                with part.open('rb') as src:
                    while block := src.read(4 * 1024 ** 2):
                        sha.update(block)
                        size += len(block)
                        out.write(block)
        if size != meta['size'] or sha.hexdigest() != meta['sha256']:
            temp.unlink()
            raise ValueError('Checkpoint transport checksum mismatch')
        import torch
        import door_seg as S
        torch.set_num_threads(2)
        state = torch.load(temp, map_location='cpu', weights_only=False)
        S.checkpoint_finite(state)
        if 'opt' not in state or not isinstance(state.get('step'), int):
            raise ValueError('Checkpoint must contain optimizer and successful step')
        # Leave a newer local training state intact; never roll it back through transport.
        target = run / name
        if name == 'last.pt' and target.exists():
            previous = torch.load(target, map_location='cpu', weights_only=False)
            if previous['step'] > state['step']:
                temp.unlink()
                raise ValueError('Refusing to overwrite a newer checkpoint')
            del previous
        os.replace(temp, target)
        print(time.strftime('%d.%m %H:%M:%S'), 'finite checkpoint restored', name,
              'step', state['step'], 'sha256', sha.hexdigest(), flush=True)
        del state
        for p in parts:
            p.unlink()
        manifest.unlink()


def main(name, once=False):
    run = ROOT / 'runs' / name
    if not run.is_dir():
        raise FileNotFoundError(run)
    while True:
        restore(run)
        if once:
            break
        ownership = run / 'COLAB_ACTIVE.json'
        if ownership.exists() and not json.loads(ownership.read_text()).get('active'):
            break
        time.sleep(20)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run')
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    main(args.run, args.once)
