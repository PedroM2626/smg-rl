import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from smg_rl.config import load_config
from smg_rl.official_btr import OFFICIAL, load_official_agent, official_settings, train_official_btr

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/algorithm-benchmark-officialbtr-proj-20261108.yaml'


class FakeVector:
    """Random 84x84 stacks; every fifth step of every worker ends its episode."""

    def __init__(self, workers):
        self.workers, self.t, self.rng = workers, 0, np.random.default_rng(0)

    def seed(self, value):
        pass

    def _obs(self):
        return {'image': self.rng.integers(0, 255, (self.workers, 4, 84, 84), dtype=np.uint8),
                'scenario': np.ones((self.workers, 1), np.float32)}

    def reset(self):
        return self._obs()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        self.t += 1
        done = np.array([self.t % 5 == 0] * self.workers)
        infos = [{'start_kind': 'full_start', 'variation': 'sweet_sweet', 'complete': False, 'dead': bool(d),
                  'scene_nerve': 0, 'worker_episode': self.t // 5, 'episode': {'r': -1.0, 'l': 5},
                  'start_relative_route_progress': 0.01,
                  'terminal_observation': {'image': self.rng.integers(0, 255, (4, 84, 84), dtype=np.uint8)}}
                 for d in done]
        return self._obs(), np.full(self.workers, -3.0), done, infos

    def close(self):
        pass


def test_official_settings_are_the_desktop_defaults_and_validate_the_discount():
    config = load_config(CONFIG)
    settings = official_settings(config, 4)
    assert settings['batch_size'] == 16 and settings['lr'] == 5e-5 and settings['spectral'] and settings['munch']
    assert settings['min_sampling_size'] == 20000 and settings['eps_steps'] == 40000 and OFFICIAL['min_sampling_size'] == 200000
    broken = deepcopy(config)
    broken['dqn']['gamma'] = 0.99
    with pytest.raises(ValueError, match='discount'):
        official_settings(broken, 4)
    broken = deepcopy(config)
    broken['official_btr']['typo'] = 1
    with pytest.raises(ValueError, match='Unknown'):
        official_settings(broken, 4)


def test_vendored_files_are_the_unmodified_mit_licensed_sources():
    notice = (ROOT / 'third_party/btr/NOTICE.md').read_text()
    assert '093ca77f08540d646f12fd942ee5fda763e0fd8a' in notice
    assert 'MIT License' in (ROOT / 'third_party/btr/LICENSE').read_text()
    assert load_official_agent().__name__ == 'Agent'


def test_official_agent_learns_through_the_adapter_and_writes_the_run_files(tmp_path):
    config = load_config(CONFIG)
    config['training']['device'] = 'cpu'
    config['paths']['runs'] = str(tmp_path / 'runs')
    config['paths']['replay'] = str(tmp_path / 'replay')
    config['parallel']['workers'] = 2
    config['official_btr'] = {'min_sampling_size': 40, 'max_mem_size': 2000, 'batch_size': 8, 'eps_steps': 100}
    result = train_official_btr(config, 160, vector_factory=lambda *a: FakeVector(2))
    run = Path(result['run'])
    status = json.loads((run / 'status.json').read_text())
    assert result['state'] == 'completed' and status['real_decisions'] == 160 and status['optimizer_updates'] > 0
    attempts = [json.loads(line) for line in (run / 'attempts.jsonl').read_text().splitlines()]
    assert len(attempts) == 2 * 16 and all(a['start_kind'] == 'full_start' and a['decisions'] == 5 for a in attempts)
    assert (run / 'checkpoint-final/official-btr.pt').exists() and status['completed_attempts'] == len(attempts)
    assert abs(sum(status['action_share']) - 1) < 1e-3
