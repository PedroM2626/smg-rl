"""Model-free collectors deliberately import neither Torch/CUDA nor SB3.

The parent owns the Q network. This runtime preserves Gymnasium's terminal
observation and autoreset protocol using only raw project environments.
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cloudpickle
import numpy as np
import psutil


@dataclass
class LeanWorkerFactory:
    config: dict
    index: int
    run_dir: str

    def __call__(self):
        from smg_rl.env import GalaxyEnv
        from smg_rl.native import background_backend

        if self.config.get('novelty', {}).get('enabled', True) or 'world_model' in self.config:
            raise ValueError('Lean collectors require a purely model-free, curiosity-disabled profile.')
        np.random.seed(self.config['training']['seed'] + self.index)
        time.sleep(self.index * self.config['parallel'].get('startup_spacing_seconds', 1.))
        backend = background_backend(self.config, self.index)
        try:
            environment = GalaxyEnv(self.config, backend, Path(self.run_dir) / f'worker-{self.index:02d}',
                memory=backend.memory, variation=self.index % len(self.config['variations']), intrinsic=False)
            resource = {'pid': psutil.Process().pid, 'private_bytes': getattr(psutil.Process().memory_info(), 'private', None),
                        'torch_imported': 'torch' in sys.modules, 'sb3_imported': 'stable_baselines3' in sys.modules}
            (environment.run_dir / 'collector-resources.json').write_text(json.dumps(resource, indent=2))
            return environment
        except BaseException:
            backend.close()
            raise


def lean_worker(remote, parent, serialized_factory, statistics, index):
    parent.close()
    env = None
    reset_info = {}
    try:
        env = cloudpickle.loads(serialized_factory)()
        while True:
            command, data = remote.recv()
            started = time.perf_counter()
            if command == 'step':
                observation, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                info['TimeLimit.truncated'] = truncated and not terminated
                if done:
                    info['terminal_observation'] = observation
                    observation, reset_info = env.reset()
                result = (observation, reward, done, info, reset_info)
            elif command == 'reset':
                kwargs = {'seed': data[0]}
                if data[1]:
                    kwargs['options'] = data[1]
                observation, reset_info = env.reset(**kwargs)
                result = (observation, reset_info)
            elif command == 'get_spaces':
                result = (env.observation_space, env.action_space)
            elif command == 'get_attr':
                result = getattr(env, data)
            elif command == 'has_attr':
                result = hasattr(env, data)
            elif command == 'set_attr':
                setattr(env, data[0], data[1])
                result = None
            elif command == 'env_method':
                result = getattr(env, data[0])(*data[1], **data[2])
            elif command == 'is_wrapped':
                result = False  # Factories construct raw Gymnasium environments, no wrappers.
            elif command == 'render':
                result = env.render()
            elif command == 'close':
                break
            else:
                raise NotImplementedError(f'Unknown lean collector command: {command}')
            if command in ('step', 'reset'):
                statistics[index * 2] = started
                statistics[index * 2 + 1] = time.perf_counter()
            remote.send(result)
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        if env is not None:
            env.close()
        remote.close()
