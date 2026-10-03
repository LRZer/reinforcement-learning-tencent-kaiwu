"""Offline checks of original numerical code; not a substitute for Kaiwu integration."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from agent_ppo.conf.conf import Config
from agent_ppo.feature.preprocessor import Preprocessor
from agent_ppo.model.model import Model
from agent_ppo.algorithm.algorithm import Algorithm


def load_definition_for_numeric_checks():
    # Only the data-record factory is replaced, inside this loader. The original
    # flatten_obs/sample_process/GAE functions execute unchanged. No platform
    # service or serialization behavior is emulated or claimed as verified.
    def create_cls(name, **fields):
        class Record:
            def __init__(self, **values):
                self.__dict__.update({k: None for k in fields})
                self.__dict__.update(values)
        return type(name, (Record,), {})
    names = ['common_python', 'common_python.utils', 'common_python.utils.common_func']
    old = {name: sys.modules.get(name) for name in names}
    try:
        for name in names:
            sys.modules[name] = types.ModuleType(name)
        sys.modules[names[-1]].create_cls = create_cls
        spec = importlib.util.spec_from_file_location('_offline_definition', ROOT/'agent_ppo/feature/definition.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name in names:
            if old[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old[name]


class OfflineChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = json.loads((ROOT/'data/examples/synthetic_observation.json').read_text(encoding='utf-8'))
        cls.definition = load_definition_for_numeric_checks()

    def test_checkpoint_matches_archived_network(self):
        model = Model(device=torch.device('cpu'))
        state = torch.load(ROOT/'ckpt/model.ckpt-34055.pkl', map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 75814)
        self.assertTrue(all(torch.isfinite(v).all() for v in state.values()))
        logits, value = model(torch.zeros(2, Config.DIM_OF_OBSERVATION))
        self.assertEqual(tuple(logits.shape), (2,16))
        self.assertEqual(tuple(value.shape), (2,1))

    def test_preprocessing_and_flatten_contract(self):
        out = Preprocessor().feature_process(self.fixture, -1)
        fields = dict(zip(['hero','monster','treasure','buff','topo','memory','map_obs'],out[:7]))
        flat = self.definition.flatten_obs(types.SimpleNamespace(**fields))
        self.assertEqual(flat.shape, (3601,))
        self.assertTrue(np.isfinite(flat).all())
        self.assertEqual(tuple(out[6].shape), (8,21,21))
        self.assertEqual(out[7], [1]*8+[0]*8)

    def test_legal_action_mask(self):
        logits = torch.tensor([[1.0,1000.0]+[0.0]*14])
        legal = torch.tensor([[1.0,0.0]+[0.0]*14])
        probs = Algorithm._masked_softmax(logits,legal)
        self.assertEqual(probs[0,0].item(), 1.0)
        self.assertEqual(probs[0,1:].sum().item(), 0.0)

    def test_bfs_detour_and_unreachable_target(self):
        pre = Preprocessor()
        grid = np.ones((21,21),dtype=np.float32)
        grid[3:18,11] = 0
        distance = pre._bfs_from_center(grid)
        self.assertGreater(distance[10,12], 2)
        grid[:,11] = 0
        distance = pre._bfs_from_center(grid)
        self.assertTrue(np.isinf(distance[10,12]))

    def test_treasure_memory_and_reset(self):
        pre = Preprocessor()
        pre.feature_process(self.fixture,-1)
        unseen = copy.deepcopy(self.fixture)
        unseen['observation']['frame_state']['organs'] = []
        pre.feature_process(unseen,0)
        self.assertGreater(pre.treasure_memory[64,67], 0.35)
        self.assertGreater(pre.buff_memory[69,64], 0.5)
        pre.reset()
        self.assertEqual(float(pre.treasure_memory.sum()), 0)
        self.assertEqual(float(pre.buff_memory.sum()), 0)

    def test_terminal_reward_branches(self):
        for terminated,truncated,steps,expected in [(True,False,40,-2.0),(False,True,1000,1.2),(False,True,40,-0.5)]:
            obs = copy.deepcopy(self.fixture)
            obs['terminated'],obs['truncated'] = terminated,truncated
            obs['observation']['env_info']['finished_steps'] = steps
            out = Preprocessor().feature_process(obs,-1)
            self.assertEqual(out[9]['reward_detail']['terminal'], expected)

    def test_gae_has_no_return_leak_across_done(self):
        samples = [types.SimpleNamespace(reward=np.array([r],dtype=np.float32),value=np.array([0.0],dtype=np.float32),done=np.array([d],dtype=np.float32)) for r,d in [(1,0),(2,1),(100,1)]]
        self.definition.sample_process(samples,gamma=0.9,lamda=0.8)
        np.testing.assert_allclose([s.advantage.item() for s in samples],[2.44,2,100],rtol=1e-6)
        self.assertEqual(samples[1].next_value.item(), 0)

    def test_ppo_update_is_finite_and_changes_parameters(self):
        torch.manual_seed(4)
        model = Model(device=torch.device('cpu'))
        optimizer = torch.optim.Adam(model.parameters(),lr=Config.INIT_LEARNING_RATE_START)
        algorithm = Algorithm(model,optimizer,torch.device('cpu'))
        obs = np.random.default_rng(4).normal(0,0.1,(4,3601)).astype(np.float32)
        with torch.no_grad():
            logits,values = model(torch.from_numpy(obs))
            probs = logits.softmax(1).numpy()
        samples = [types.SimpleNamespace(obs=obs[i],legal_action=np.ones(16,np.float32),act=np.array([i]),prob=probs[i],reward=np.array([1.0],np.float32),advantage=np.array([i-1.5],np.float32),value=values[i].numpy(),reward_sum=np.array([i*0.5],np.float32)) for i in range(4)]
        before = model.actor_head.weight.detach().clone()
        algorithm.learn(samples)
        self.assertEqual(algorithm.train_step,1)
        self.assertFalse(torch.equal(before,model.actor_head.weight))
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
