"""Inspect the archived state_dict without importing the Kaiwu runtime."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from agent_ppo.model.model import Model


def inspect(path):
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = Model(device=torch.device('cpu'))
    model.load_state_dict(state, strict=True)
    groups = {}
    tensors = []
    for name, tensor in state.items():
        group = name.split('.')[0]
        groups[group] = groups.get(group, 0) + tensor.numel()
        tensors.append({'name': name, 'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                        'elements': tensor.numel(), 'finite': bool(torch.isfinite(tensor).all())})
    assert all(t['finite'] for t in tensors), 'Non-finite checkpoint tensor'
    return {
        'checkpoint': path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name,
        'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'format': 'PyTorch state_dict; model weights only',
        'optimizer_state_available': False, 'strict_model_load': True,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'tensor_count': len(tensors), 'all_tensors_finite': True,
        'parameters_by_module': groups, 'tensors': tensors,
        'inspection_torch_version': torch.__version__,
        'scope': 'Structural compatibility and numeric integrity; no game performance measured.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'ckpt/model.ckpt-34055.pkl')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = inspect(args.checkpoint.resolve())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['checkpoint', 'sha256', 'parameter_count', 'tensor_count', 'strict_model_load', 'all_tensors_finite']}, indent=2))


if __name__ == '__main__':
    main()
