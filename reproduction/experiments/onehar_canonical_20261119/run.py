"""Use the unchanged shared split/scaler/training/artifact runner for oneHAR."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import sys

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent/'imu_continuation_20261106'
sys.path.insert(0, str(SHARED))
spec = importlib.util.spec_from_file_location('canonical_runner', SHARED/'train.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
sys.path.insert(0, str(HERE))
from onehar_adapter import build

runner.HERE = HERE
runner.SOURCE_BYTES = {p.name: p.read_bytes() for p in HERE.iterdir() if p.suffix in {'.py', '.md'}}
runner.SOURCE_BYTES['train.py'] = (SHARED/'train.py').read_bytes()
runner.SOURCE_BYTES['prepare.py'] = (SHARED/'prepare.py').read_bytes()


def construct(method, length, channels, classes, **context):
    assert method == 'onehar'
    return build(context['x'], context['train'], context['validation'], context['out'], context['a'], context['device'])


runner.build = construct

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--fold', required=True)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--memory-fraction', type=float, default=.16)
    p.add_argument('--smoke', action='store_true')
    a = p.parse_args(); a.method = 'onehar'
    if not a.smoke and a.epochs != 200:
        raise ValueError('Production uses the shared 200-epoch cap; adapter recipe caps CDNet at80. Short runs require --smoke.')
    runner.run(a)
