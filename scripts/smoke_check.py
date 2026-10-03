"""Run preprocessing and checkpoint inference on a labeled synthetic observation."""
import json
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from agent_ppo.conf.conf import Config
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from agent_ppo.algorithm.algorithm import Algorithm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    fixture = json.loads((ROOT/'data/examples/synthetic_observation.json').read_text(encoding='utf-8'))
    pre = Preprocessor()
    features = pre.feature_process(fixture, -1)
    flat = np.concatenate([x.reshape(-1) for x in features[:7]]).astype(np.float32)
    assert flat.shape == (Config.DIM_OF_OBSERVATION,) and np.isfinite(flat).all()
    model = Model(device=torch.device('cpu'))
    model.load_state_dict(torch.load(ROOT/'ckpt/model.ckpt-34055.pkl', map_location='cpu', weights_only=True), strict=True)
    model.eval()
    with torch.no_grad():
        logits, value = model(torch.from_numpy(flat).unsqueeze(0))
        probabilities = Algorithm._masked_softmax(logits, torch.tensor([features[7]], dtype=torch.float32))
    assert logits.shape == (1,16) and value.shape == (1,1)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert float(probabilities[:,8:].sum()) == 0.0
    action = int(probabilities.argmax(1).item())
    assert features[7][action] == 1
    report = {
        'scope': 'Synthetic interface check only; no simulator, historical episode or competition evaluation.',
        'fixture': 'data/examples/synthetic_observation.json',
        'observation_shape': list(flat.shape), 'map_shape': list(features[6].shape),
        'logits_shape': list(logits.shape), 'value_shape': list(value.shape),
        'finite_outputs': True, 'illegal_action_probability': 0.0,
        'greedy_action_on_synthetic_fixture': action,
        'numpy_version': np.__version__, 'torch_version': torch.__version__,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
